"""Bug 1: worker-pool pinning deadlock — evict an idle mismatched worker at the
cap and spawn a correctly-pinned replacement, without stalling.
Bug 3: recycle workers after N tasks to avoid RSS baseline drift."""

import time

import pytest

import adaptive_executor.adaptive_executor as ae
from adaptive_executor.adaptive_executor import AdaptiveExecutor, WorkerSlot
from fakes import FakeProcess, FakeQueue


def _make_executor(max_workers=4, **kw):
    """Executor that never spawns real processes: _spawn_worker is stubbed."""
    ex = AdaptiveExecutor(max_workers=max_workers, **kw)

    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True),
            work_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]
    return ex


def _add_slot(ex, pinned, busy=False):
    slot = ex._spawn_worker(pinned)
    if busy:
        slot.current_work_id = f"work-{slot.worker_id}"
    return slot


# --- Bug 1 -----------------------------------------------------------------


def test_reuses_matching_idle_worker():
    ex = _make_executor(max_workers=4)
    existing = _add_slot(ex, pinned=0)
    got = ex._get_or_spawn_idle_worker(0)
    assert got is existing
    assert ex._retiring == []


def test_evicts_idle_mismatched_worker_at_cap():
    ex = _make_executor(max_workers=4)
    # Fill the pool with 4 idle CPU workers (pinned None).
    cpu_workers = [_add_slot(ex, pinned=None) for _ in range(4)]
    assert len(ex.workers) == 4

    # A GPU task (pin 0) arrives; no matching idle worker and we are at the cap.
    got = ex._get_or_spawn_idle_worker(0)

    assert got is not None
    assert got.pinned_gpu_id == 0
    # Cap preserved: one CPU worker evicted, one GPU worker spawned.
    assert len(ex.workers) == 4
    # Exactly one worker retired, given the stop sentinel, and removed from pool.
    assert len(ex._retiring) == 1
    victim = ex._retiring[0]
    assert victim in cpu_workers
    assert victim.intentionally_stopped is True
    assert victim.work_queue.items == [None]
    assert victim.worker_id not in ex.workers


def test_all_busy_pool_returns_none():
    ex = _make_executor(max_workers=3)
    for _ in range(3):
        _add_slot(ex, pinned=None, busy=True)
    # Every worker is busy: correct backpressure, no eviction.
    assert ex._get_or_spawn_idle_worker(0) is None
    assert ex._retiring == []


def test_check_workers_reaps_retiring():
    ex = _make_executor(max_workers=2)
    victim = _add_slot(ex, pinned=None)
    ex._retire_worker(victim, reason="test")
    assert victim in ex._retiring

    # Worker has now exited.
    victim.process.set_dead(exitcode=0)
    ex._check_workers()

    assert ex._retiring == []
    # It was joined (reaped) so no zombie remains.
    assert victim.process.join_timeouts != []


def test_check_workers_keeps_still_running_retiring():
    ex = _make_executor(max_workers=2)
    victim = _add_slot(ex, pinned=None)
    ex._retire_worker(victim, reason="test")
    # Still alive (hasn't processed the sentinel yet).
    ex._check_workers()
    assert ex._retiring == [victim]


# --- Bug 3 -----------------------------------------------------------------


def test_should_recycle_threshold():
    ex = _make_executor(max_workers=2, worker_recycle_after_tasks=3)
    w = _add_slot(ex, pinned=None)
    w.tasks_completed = 2
    assert ex._should_recycle(w) is False
    w.tasks_completed = 3
    assert ex._should_recycle(w) is True


def test_recycle_disabled_when_none():
    ex = _make_executor(max_workers=2, worker_recycle_after_tasks=None)
    w = _add_slot(ex, pinned=None)
    w.tasks_completed = 1000
    assert ex._should_recycle(w) is False


def test_retire_worker_moves_to_retiring():
    ex = _make_executor(max_workers=2, worker_recycle_after_tasks=1)
    w = _add_slot(ex, pinned=None)
    w.tasks_completed = 5
    ex._retire_worker(w, reason="recycle")
    assert w.worker_id not in ex.workers
    assert w in ex._retiring
    assert w.work_queue.items == [None]
    assert w.intentionally_stopped is True


# --- Bug 3 integration with real worker processes ---------------------------


def _wait(fut, timeout=30.0):
    return fut.result(timeout=timeout)


def test_recycle_spawns_fresh_workers_no_zombies():
    import task_fns

    with AdaptiveExecutor(max_workers=1, worker_recycle_after_tasks=2) as ex:
        futures = [ex.submit(task_fns.echo, i) for i in range(6)]
        results = [_wait(f) for f in futures]
    assert results == list(range(6))
    # With max_workers=1 and recycle-after-2, more than one worker must have
    # been spawned over the run.
    assert ex._next_worker_id >= 2
    # No retired worker is left alive (no zombies).
    assert all(not w.process.is_alive() for w in ex._retiring)


def test_recycle_disabled_uses_single_worker():
    import task_fns

    with AdaptiveExecutor(max_workers=1, worker_recycle_after_tasks=None) as ex:
        futures = [ex.submit(task_fns.echo, i) for i in range(6)]
        results = [_wait(f) for f in futures]
    assert results == list(range(6))
    # Recycling disabled: exactly one worker handled everything.
    assert ex._next_worker_id == 1
