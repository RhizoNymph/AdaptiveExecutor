"""Unit tests for exclusive-*run* isolation in the pure scheduler.

The sibling ``test_scheduling.py`` covers exclusive *pending* semantics (an
exclusive head cannot start while anything runs and blocks all backfill). These
tests cover the complementary rule: while an exclusive task is already *in
flight*, ``plan_dispatch`` dispatches nothing at all — an exclusive task runs
alone for its whole run, not just at its start. Exclusivity lifts the moment the
task leaves the running set.

All deterministic; no threads, no I/O.
"""

from adaptive_executor.scheduling import (
    Capacity,
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
    running_count=1,
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


def _running(id, *, mem=1.0, vram=0.0, gpu=None, remaining=None, exclusive=False):
    return RunningEntry(
        id=id,
        memory_gb=mem,
        vram_gb=vram,
        gpu_id=gpu,
        remaining_seconds=remaining,
        exclusive=exclusive,
    )


def _ids(plan):
    return [d.pending_id for d in plan.decisions]


# --- a running exclusive task blocks every other dispatch -------------------


def test_running_exclusive_blocks_front_admission():
    # Tons of free capacity and idle worker slots, but an exclusive task is in
    # flight -> nothing may be dispatched alongside it.
    running = [_running("excl", mem=5.0, remaining=10.0, exclusive=True)]
    pending = [_pending("a", mem=1.0), _pending("b", mem=1.0)]
    plan = plan_dispatch(pending, running, _cap(memory_free=100.0, running_count=1))
    assert _ids(plan) == []


def test_running_exclusive_blocks_backfill():
    # A blocked head with a would-be rule (a)/(b) backfill candidate behind it,
    # but an exclusive task is running -> no backfill either.
    running = [
        _running("excl", mem=5.0, remaining=10.0, exclusive=True),
        _running("r", mem=30.0, remaining=10.0),
    ]
    head = _pending("head", mem=40.0)
    small = _pending("small", mem=1.0, duration=0.001)
    plan = plan_dispatch([head, small], running, _cap(memory_free=20.0, running_count=2))
    assert _ids(plan) == []


def test_running_exclusive_among_several_running_blocks_all():
    # The exclusive one need not be first in the running list.
    running = [
        _running("r0", mem=1.0, remaining=5.0),
        _running("excl", mem=1.0, remaining=5.0, exclusive=True),
        _running("r1", mem=1.0, remaining=5.0),
    ]
    pending = [_pending("a", mem=1.0), _pending("b", mem=1.0)]
    plan = plan_dispatch(pending, running, _cap(memory_free=100.0, running_count=3))
    assert _ids(plan) == []


def test_running_exclusive_blocks_even_an_exclusive_pending():
    # A pending exclusive task also cannot start while an exclusive one runs.
    running = [_running("excl", mem=1.0, remaining=5.0, exclusive=True)]
    pending = [_pending("also-excl", mem=1.0, exclusive=True), _pending("b", mem=1.0)]
    plan = plan_dispatch(pending, running, _cap(memory_free=100.0, running_count=1))
    assert _ids(plan) == []


# --- exclusivity lifts once the exclusive task leaves the running set --------


def test_exclusivity_lifts_when_exclusive_completes():
    pending = [_pending("a", mem=1.0), _pending("b", mem=1.0)]

    # While the exclusive task is in flight: nothing dispatches.
    running_before = [_running("excl", mem=5.0, remaining=10.0, exclusive=True)]
    blocked = plan_dispatch(
        pending, running_before, _cap(memory_free=100.0, running_count=1)
    )
    assert _ids(blocked) == []

    # Once it has completed (gone from the running set): the queue flows again.
    running_after: list[RunningEntry] = []
    freed = plan_dispatch(
        pending, running_after, _cap(memory_free=100.0, running_count=0)
    )
    assert _ids(freed) == ["a", "b"]


# --- a running *non*-exclusive task must NOT block (regression guard) --------


def test_running_non_exclusive_does_not_block():
    running = [_running("r", mem=5.0, remaining=10.0, exclusive=False)]
    pending = [_pending("a", mem=1.0), _pending("b", mem=1.0)]
    plan = plan_dispatch(pending, running, _cap(memory_free=100.0, running_count=1))
    # No exclusivity in flight -> normal front admission proceeds.
    assert _ids(plan) == ["a", "b"]
