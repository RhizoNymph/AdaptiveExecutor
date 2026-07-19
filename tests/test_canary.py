"""Unit tests for the cold-start canary in the pure scheduler.

While a task's estimation profile is cold (zero observations), at most ONE task
per profile identity may run at a time, so N identical first-contact submits are
not admitted together into a mass OOM. Covered: a cold task blocks a second cold
task of the same identity (front + backfill); different identities are
unaffected; a warm profile unclamps; an explicit hint bypasses; and a
cold-blocked head still permits disjoint rule-(a) backfill without stalling
feasible non-cold work.

All deterministic; no threads, no I/O.
"""

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


def _pending(
    id, *, mem=1.0, vram=0.0, cpu=1.0, duration=None, exclusive=False,
    identity=None, cold=False,
):
    return PendingEntry(
        id=id,
        memory_gb=mem,
        vram_gb=vram,
        cpu_cores=cpu,
        duration_p90_seconds=duration,
        exclusive=exclusive,
        profile_identity=identity,
        cold=cold,
    )


def _running(id, *, mem=1.0, vram=0.0, gpu=None, remaining=None, identity=None):
    return RunningEntry(
        id=id,
        memory_gb=mem,
        vram_gb=vram,
        gpu_id=gpu,
        remaining_seconds=remaining,
        profile_identity=identity,
    )


def _ids(plan):
    return [d.pending_id for d in plan.decisions]


# --- one cold task at a time per identity -----------------------------------


def test_first_cold_task_admits_but_second_same_identity_blocked():
    # Two cold submits of the same identity arrive together with ample capacity.
    # Only the first (the canary) starts; the second must wait.
    a = _pending("a", mem=1.0, identity="m:f", cold=True)
    b = _pending("b", mem=1.0, identity="m:f", cold=True)
    plan = plan_dispatch([a, b], [], _cap(memory_free=100.0))
    assert _ids(plan) == ["a"]


def test_cold_task_blocked_by_running_same_identity_canary():
    # A canary of identity m:f is already in flight; a cold sibling must wait
    # even though there is plenty of capacity and free worker slots.
    running = [_running("canary", mem=1.0, remaining=None, identity="m:f")]
    b = _pending("b", mem=1.0, identity="m:f", cold=True)
    plan = plan_dispatch([b], running, _cap(memory_free=100.0, running_count=1))
    assert _ids(plan) == []


def test_cold_tasks_of_different_identities_are_unaffected():
    # Distinct identities each get their own canary; both admit.
    a = _pending("a", mem=1.0, identity="m:f", cold=True)
    b = _pending("b", mem=1.0, identity="m:g", cold=True)
    plan = plan_dispatch([a, b], [], _cap(memory_free=100.0))
    assert _ids(plan) == ["a", "b"]


def test_warm_profile_is_not_clamped():
    # Not cold (warm profile): several tasks of the same identity co-admit.
    a = _pending("a", mem=1.0, identity="m:f", cold=False)
    b = _pending("b", mem=1.0, identity="m:f", cold=False)
    c = _pending("c", mem=1.0, identity="m:f", cold=False)
    plan = plan_dispatch([a, b, c], [], _cap(memory_free=100.0))
    assert _ids(plan) == ["a", "b", "c"]


def test_hint_bypasses_canary():
    # The executor never marks a hinted task cold; model that here as cold=False.
    # Two same-identity hinted tasks both admit despite a shared identity.
    a = _pending("a", mem=1.0, identity="m:f", cold=False)
    b = _pending("b", mem=1.0, identity="m:f", cold=False)
    plan = plan_dispatch([a, b], [], _cap(memory_free=100.0))
    assert _ids(plan) == ["a", "b"]


def test_second_cold_blocked_in_backfill_position():
    # A cold head of one identity is blocked (resource), and a cold candidate of
    # a DIFFERENT identity backfills; but a second cold candidate sharing the
    # first candidate's identity is refused in backfill.
    head = _pending("head", mem=40.0, identity="m:head", cold=True)
    bf1 = _pending("bf1", mem=3.0, duration=1000.0, identity="m:x", cold=True)
    bf2 = _pending("bf2", mem=3.0, duration=1000.0, identity="m:x", cold=True)
    running = [_running("r", mem=30.0, remaining=10.0, identity="m:r")]
    # 20 free + 30 released = 50 at reservation; head 40 leaves 10 slack.
    plan = plan_dispatch([head, bf1, bf2], running, _cap(memory_free=20.0, running_count=1))
    # bf1 backfills (disjoint, its own identity); bf2 shares bf1's now-held
    # identity and is refused by the canary.
    assert _ids(plan) == ["bf1"]


# --- cold-blocked head does not stall feasible non-cold work ----------------


def test_cold_blocked_head_allows_disjoint_backfill():
    # The head is cold-blocked by a running canary of its identity. A non-cold
    # task behind it that fits disjointly must still be dispatched (rule a),
    # never stalled behind the canary.
    running = [_running("canary", mem=1.0, remaining=None, identity="m:f")]
    head = _pending("head", mem=1.0, identity="m:f", cold=True)
    behind = _pending("behind", mem=5.0, duration=1000.0, identity="m:g", cold=False)
    plan = plan_dispatch(
        [head, behind], running, _cap(memory_free=100.0, running_count=1)
    )
    # Head waits (canary); disjoint non-cold work backfills past it.
    assert _ids(plan) == ["behind"]


def test_cold_blocked_head_rule_b_disabled_reservation_infinite():
    # With a cold-blocked head, the reservation is infinite: a short task that is
    # NOT disjoint (would rely on rule b) cannot backfill, only disjoint work.
    running = [_running("canary", mem=1.0, remaining=None, identity="m:f")]
    head = _pending("head", mem=90.0, identity="m:f", cold=True)
    # short would finish quickly, but rule (b) is off (reservation inf) and it is
    # not disjoint from the head's 90GB need (holding 15 leaves 84 < 90).
    short = _pending("short", mem=15.0, duration=0.001, identity="m:g", cold=False)
    plan = plan_dispatch(
        [head, short], running, _cap(memory_free=95.0, running_count=1)
    )
    assert _ids(plan) == []


def test_cold_head_admits_when_identity_not_held():
    # A cold head whose identity is NOT already running is itself the canary and
    # is admitted normally.
    head = _pending("head", mem=1.0, identity="m:f", cold=True)
    plan = plan_dispatch([head], [], _cap(memory_free=100.0))
    assert _ids(plan) == ["head"]


def test_cold_backfill_candidate_becomes_its_own_canary():
    # A blocked (resource) non-cold head; a cold candidate of a fresh identity
    # backfills as its own canary.
    head = _pending("head", mem=40.0, identity="m:head", cold=False)
    cold_bf = _pending("cbf", mem=3.0, duration=1000.0, identity="m:new", cold=True)
    running = [_running("r", mem=30.0, remaining=10.0, identity="m:r")]
    plan = plan_dispatch([head, cold_bf], running, _cap(memory_free=20.0, running_count=1))
    assert _ids(plan) == ["cbf"]
    assert plan.decisions == (DispatchDecision(pending_id="cbf", gpu_id=None),)
