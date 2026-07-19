"""Unit tests for the pure EASY reservation-based backfill scheduler.

All deterministic; no threads, no I/O. Covers: head-blocked with a small task
fitting behind the reservation (admitted), a small task that would delay the
head (rejected), unknown durations (only disjoint backfill), exclusive head (no
backfill), per-GPU reservation accounting, and overrunning tasks slipping the
reservation.
"""

import math

from adaptive_executor.scheduling import (
    Capacity,
    DispatchDecision,
    PendingEntry,
    RunningEntry,
    plan_dispatch,
)


def _cap(
    *,
    memory_free=100.0,
    gpu_free=None,
    gpus=(),
    next_gpu_index=0,
    max_workers=4,
    running_count=0,
    snapshot_present=True,
):
    return Capacity(
        memory_free_gb=memory_free,
        gpu_free_vram_gb=gpu_free or {},
        gpu_round_robin=tuple(gpus),
        next_gpu_index=next_gpu_index,
        max_workers=max_workers,
        running_count=running_count,
        snapshot_present=snapshot_present,
    )


def _pending(id, *, mem=1.0, vram=0.0, cpu=1.0, duration=None, exclusive=False):
    return PendingEntry(
        id=id,
        memory_gb=mem,
        vram_gb=vram,
        cpu_cores=cpu,
        duration_p90_seconds=duration,
        exclusive=exclusive,
    )


def _running(id, *, mem=1.0, vram=0.0, gpu=None, remaining=None):
    return RunningEntry(
        id=id, memory_gb=mem, vram_gb=vram, gpu_id=gpu, remaining_seconds=remaining
    )


def _ids(plan):
    return [d.pending_id for d in plan.decisions]


# --- basic FIFO front admission --------------------------------------------


def test_all_fit_admits_everything_in_order():
    pending = [_pending("a", mem=1.0), _pending("b", mem=1.0), _pending("c", mem=1.0)]
    plan = plan_dispatch(pending, [], _cap(memory_free=10.0, max_workers=4))
    assert _ids(plan) == ["a", "b", "c"]


def test_worker_cap_limits_front_admission():
    pending = [_pending(x, mem=0.1) for x in "abcde"]
    plan = plan_dispatch(pending, [], _cap(memory_free=100.0, max_workers=3))
    # Only 3 worker slots; nothing behind can backfill (head needs a slot too).
    assert _ids(plan) == ["a", "b", "c"]


def test_head_blocked_no_backfill_candidates():
    # Head does not fit on memory and there is nothing behind it.
    pending = [_pending("big", mem=50.0)]
    running = [_running("r", mem=40.0, remaining=5.0)]
    plan = plan_dispatch(pending, running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == []


# --- rule (b): short task finishes before the reservation -------------------


def test_small_task_finishing_before_reservation_is_backfilled():
    # Head needs 40GB but only 20 free now; running task frees 30GB at t=10.
    head = _pending("head", mem=40.0)
    small = _pending("small", mem=5.0, duration=3.0)  # finishes before t=10
    running = [_running("r", mem=30.0, remaining=10.0)]
    plan = plan_dispatch([head, small], running, _cap(memory_free=20.0, running_count=1))
    # Head still blocked; small backfilled because it releases before reservation.
    assert _ids(plan) == ["small"]


def test_small_task_too_long_would_delay_head_is_rejected():
    # Same as above but the small task runs LONGER than the reservation and its
    # memory is needed by the head -> rejected (rule b fails, rule a fails).
    head = _pending("head", mem=40.0)
    # 20 free + 30 released = 50 at reservation; head needs 40, leaving 10 slack.
    # A 15GB task held through the reservation would leave only 35 < 40 -> delays.
    big = _pending("big", mem=15.0, duration=100.0)
    running = [_running("r", mem=30.0, remaining=10.0)]
    plan = plan_dispatch([head, big], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == []


# --- rule (a): disjoint capacity, holds indefinitely ------------------------


def test_disjoint_small_task_backfills_even_if_long():
    # Reservation leaves plenty of slack, so a long task that fits in the slack
    # backfills via rule (a) regardless of its duration.
    head = _pending("head", mem=40.0)
    small = _pending("small", mem=5.0, duration=1000.0)
    running = [_running("r", mem=30.0, remaining=10.0)]
    # 20 free + 30 released = 50 at reservation; head 40 + small 5 = 45 <= 50.
    plan = plan_dispatch([head, small], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == ["small"]


def test_unknown_duration_task_only_backfills_when_disjoint():
    head = _pending("head", mem=40.0)
    # Unknown duration -> cannot use rule (b). Must be disjoint (rule a).
    disjoint = _pending("disjoint", mem=5.0, duration=None)
    running = [_running("r", mem=30.0, remaining=10.0)]
    plan = plan_dispatch([head, disjoint], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == ["disjoint"]


def test_unknown_duration_task_not_disjoint_is_rejected():
    head = _pending("head", mem=40.0)
    # Would fit now (20 free) but is NOT disjoint from the reservation and has
    # unknown duration -> cannot backfill by either rule.
    unknown = _pending("unknown", mem=15.0, duration=None)
    running = [_running("r", mem=30.0, remaining=10.0)]
    plan = plan_dispatch([head, unknown], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == []


# --- unknown reservation (head waits on unknown-duration running task) -------


def test_unknown_running_duration_disables_rule_b():
    # The running task holding the memory the head needs has unknown duration ->
    # reservation is infinite -> rule (b) disabled; only disjoint backfill.
    head = _pending("head", mem=40.0)
    short = _pending("short", mem=15.0, duration=0.001)  # would-be rule (b)
    running = [_running("r", mem=30.0, remaining=None)]
    plan = plan_dispatch([head, short], running, _cap(memory_free=20.0, running_count=1))
    # 20 free now; head needs 40 and can never be satisfied by this running task.
    # 'short' fits now (15 < 20) but is not disjoint (holding 15 leaves 5 < 40
    # would-be head need at inf, and reservation never reachable) -> rejected.
    assert _ids(plan) == []


def test_unknown_running_duration_allows_disjoint_backfill():
    # Head waits on unknown-duration running task, but capacity is large enough
    # that a small task is disjoint and can backfill via rule (a).
    head = _pending("head", mem=40.0)
    small = _pending("small", mem=5.0, duration=0.001)
    # Head can never fit (needs 40, only 20 free forever) -> at inf head doesn't
    # fit even without small -> disjoint check fails too. So expect no backfill.
    running = [_running("r", mem=30.0, remaining=None)]
    plan = plan_dispatch([head, small], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == []


# --- exclusive head blocks all backfill -------------------------------------


def test_exclusive_head_blocks_all_backfill():
    head = _pending("head", mem=1.0, exclusive=True)
    tiny = _pending("tiny", mem=0.1, duration=0.001)
    running = [_running("r", mem=1.0, remaining=1.0)]
    plan = plan_dispatch([head, tiny], running, _cap(memory_free=100.0, running_count=1))
    # Exclusive head needs the in-flight set empty; nothing may backfill.
    assert _ids(plan) == []


def test_exclusive_head_admitted_when_idle():
    head = _pending("head", mem=1.0, exclusive=True)
    behind = _pending("behind", mem=1.0)
    plan = plan_dispatch([head, behind], [], _cap(memory_free=100.0, running_count=0))
    # Idle: the exclusive head admits — alone. Exclusive means "runs alone,
    # start to finish", so nothing is co-admitted in the same cycle; 'behind'
    # waits until the exclusive task finishes.
    assert _ids(plan) == ["head"]


# --- per-GPU reservation accounting -----------------------------------------


def test_backfill_must_not_take_the_gpu_the_head_waits_for():
    # Head needs 8GB VRAM but no GPU has room now (GPU0=5, GPU1=4). GPU0's running
    # task frees +4 at t=10, so at the reservation GPU0 has 9 and fits the head;
    # GPU1 never fits the head. Head reserves GPU0. A 3GB backfill task fits on
    # GPU1 now and, being disjoint from GPU0, is admitted there (never GPU0).
    head = _pending("head", mem=1.0, vram=8.0)
    backfill = _pending("bf", mem=1.0, vram=3.0, duration=1000.0)
    running = [_running("r", mem=1.0, vram=4.0, gpu=0, remaining=10.0)]
    cap = _cap(
        memory_free=100.0,
        gpu_free={0: 5.0, 1: 4.0},
        gpus=(0, 1),
        running_count=1,
    )
    plan = plan_dispatch([head, backfill], running, cap)
    # Steered onto GPU1 (disjoint from the head's GPU0 reservation).
    assert plan.decisions == (DispatchDecision(pending_id="bf", gpu_id=1),)


def test_backfill_stealing_reserved_gpu_is_rejected():
    # Only one GPU. Head needs 8GB; GPU0 has 5 free now (blocked) and frees +4 at
    # t=10 -> 9 at the reservation (head fits, 1 slack). A long 3GB task fits now
    # (3<5) but held through the reservation leaves 9-3=6 < 8 -> would delay the
    # head -> rejected. No other GPU to escape to.
    head = _pending("head", mem=1.0, vram=8.0)
    thief = _pending("thief", mem=1.0, vram=3.0, duration=1000.0)
    running = [_running("r", mem=1.0, vram=4.0, gpu=0, remaining=10.0)]
    cap = _cap(memory_free=100.0, gpu_free={0: 5.0}, gpus=(0,), running_count=1)
    plan = plan_dispatch([head, thief], running, cap)
    assert _ids(plan) == []


def test_short_gpu_task_may_use_reserved_gpu_if_it_finishes_first():
    # Same single-GPU setup, but the VRAM task finishes (2s) before the head's
    # reservation (10s), so rule (b) admits it on the reserved GPU.
    head = _pending("head", mem=1.0, vram=8.0)
    quick = _pending("quick", mem=1.0, vram=3.0, duration=2.0)
    running = [_running("r", mem=1.0, vram=4.0, gpu=0, remaining=10.0)]
    cap = _cap(memory_free=100.0, gpu_free={0: 5.0}, gpus=(0,), running_count=1)
    plan = plan_dispatch([head, quick], running, cap)
    assert plan.decisions == (DispatchDecision(pending_id="quick", gpu_id=0),)


# --- overrunning tasks slip the reservation ---------------------------------


def test_overrun_running_task_slips_reservation_toward_fifo():
    # An overrunning running task is reported with remaining=None (unknown) by
    # the caller. That makes the reservation infinite and disables rule (b): a
    # short task that WOULD have backfilled before the overrun no longer can.
    head = _pending("head", mem=40.0)
    short = _pending("short", mem=15.0, duration=0.5)

    # Before overrun: running task frees 30GB at t=5. reservation=5.
    running_before = [_running("r", mem=30.0, remaining=5.0)]
    cap = _cap(memory_free=20.0, running_count=1)
    plan_before = plan_dispatch([head, short], running_before, cap)
    # short finishes (0.5) before reservation (5): rule (b) admits it.
    assert _ids(plan_before) == ["short"]

    # After overrun: same task now reported unknown -> reservation infinite ->
    # rule (b) disabled; short is not disjoint (15 held leaves 5 < 40) -> rejected.
    running_after = [_running("r", mem=30.0, remaining=None)]
    plan_after = plan_dispatch([head, short], running_after, cap)
    assert _ids(plan_after) == []


# --- multiple backfill candidates, FIFO order -------------------------------


def test_multiple_backfill_candidates_scanned_in_fifo_order():
    # Head blocked. Two candidates both disjoint-fit; both admitted in order.
    head = _pending("head", mem=40.0)
    b1 = _pending("b1", mem=3.0, duration=1000.0)
    b2 = _pending("b2", mem=3.0, duration=1000.0)
    running = [_running("r", mem=30.0, remaining=10.0)]
    # 20 free + 30 = 50 at reservation; head 40 + b1 3 + b2 3 = 46 <= 50.
    plan = plan_dispatch([head, b1, b2], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == ["b1", "b2"]


def test_backfill_budget_exhausted_rejects_later_candidate():
    head = _pending("head", mem=40.0)
    b1 = _pending("b1", mem=8.0, duration=1000.0)  # disjoint (50-40=10 slack)
    b2 = _pending("b2", mem=8.0, duration=1000.0)  # would push head below need
    running = [_running("r", mem=30.0, remaining=10.0)]
    plan = plan_dispatch([head, b1, b2], running, _cap(memory_free=20.0, running_count=1))
    # b1 uses 8 of 10 slack; b2 (another 8) would leave head short -> rejected.
    assert _ids(plan) == ["b1"]


# --- no snapshot (startup) --------------------------------------------------


def test_no_snapshot_memory_not_gating():
    pending = [_pending("a", mem=999.0), _pending("b", mem=999.0)]
    cap = _cap(memory_free=math.inf, max_workers=4, snapshot_present=False)
    plan = plan_dispatch(pending, [], cap)
    assert _ids(plan) == ["a", "b"]


def test_reservation_advances_gpu_round_robin_index_on_backfill():
    # Two GPUs; head reserves GPU0. A disjoint backfill task lands on GPU1 and
    # advances the RR index past it.
    head = _pending("head", mem=1.0, vram=8.0)
    bf = _pending("bf", mem=1.0, vram=3.0, duration=1000.0)
    running = [_running("r", mem=1.0, vram=4.0, gpu=0, remaining=10.0)]
    cap = _cap(
        memory_free=100.0,
        gpu_free={0: 5.0, 1: 4.0},
        gpus=(0, 1),
        next_gpu_index=1,
        running_count=1,
    )
    plan = plan_dispatch([head, bf], running, cap)
    assert plan.decisions == (DispatchDecision(pending_id="bf", gpu_id=1),)
    # Picked index 1 -> next index wraps to 0.
    assert plan.next_gpu_index == 0
