"""Per-worker result queues isolate worker-kill corruption.

All workers used to ``put()`` results onto one shared ``mp.Queue``; a SIGTERM/
SIGKILL landing mid-``put`` could wedge that shared feeder state and silently
break result delivery for EVERY worker. Now each worker owns its result queue,
so killing one worker can only affect that worker's queue.

Coverage:
  Real-subprocess (survival / end-to-end delivery):
    - kill_worker timeout kills one worker while OTHER workers keep completing
      tasks and delivering results afterwards.
    - recycle drain: every result is delivered across worker recycling.
    - crash isolation: a worker that SIGKILLs itself fails only its own task;
      other workers keep delivering.
  Deterministic fake-level (collector sweep + queue lifecycle bookkeeping):
    - sweep collects from every live, non-poisoned worker queue.
    - a killed worker's queue is poisoned: never swept, discarded (not drained).
    - graceful-retire reap drains a queued final result, then discards.
    - _handle_dead_worker drains a final result for an earlier task, then
      applies crash handling to the current task.
    - discard closes + cancels the join thread and is idempotent.
    - kill escalates SIGTERM -> SIGKILL when the worker ignores SIGTERM.
"""

import time
from concurrent.futures import Future

import pytest

from adaptive_executor.adaptive_executor import AdaptiveExecutor, PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    ResourceEstimate,
    ResourceObservation,
    WorkItem,
    WorkResult,
)
from fakes import FakeProcess, FakeQueue


# --- real-subprocess tests -------------------------------------------------


def test_kill_worker_does_not_corrupt_other_result_delivery():
    """The survival property the shared-queue design could not guarantee: after
    one worker is killed on timeout, the remaining workers still deliver results,
    and freshly submitted tasks still complete."""
    import kill_task_fns
    import task_fns

    with AdaptiveExecutor(
        max_workers=3,
        task_timeout_seconds=0.4,
        on_timeout="kill_worker",
    ) as ex:
        # Short tasks running concurrently with the doomed long task.
        concurrent = [ex.submit(task_fns.add, i, i) for i in range(3)]
        long_fut = ex.submit(kill_task_fns.busy_sleep, 30)

        # The long task overruns its timeout and its worker is killed.
        with pytest.raises(TimeoutError):
            long_fut.result(timeout=15)

        # Concurrent short tasks were delivered on their own queues.
        assert [f.result(timeout=15) for f in concurrent] == [0, 2, 4]

        # And result delivery is intact AFTER the kill: new tasks complete.
        after = [ex.submit(task_fns.add, i, 100) for i in range(10)]
        assert [f.result(timeout=15) for f in after] == [i + 100 for i in range(10)]


def test_recycle_drains_all_results():
    """Recycling retires workers mid-run; the drain-at-reap path must deliver
    every result with none lost."""
    import task_fns

    with AdaptiveExecutor(max_workers=2, worker_recycle_after_tasks=2) as ex:
        futures = [ex.submit(task_fns.echo, i) for i in range(12)]
        results = [f.result(timeout=15) for f in futures]

    assert results == list(range(12))
    # Recycling actually happened (more workers than the cap were spawned).
    assert ex._next_worker_id > 2


def test_worker_crash_isolates_to_its_own_task():
    """A worker that SIGKILLs itself fails only its own task; other workers keep
    delivering results on their own queues."""
    import kill_task_fns
    import task_fns

    with AdaptiveExecutor(max_workers=3, max_resource_crash_retries=1) as ex:
        crash_fut = ex.submit(kill_task_fns.crash_now)

        # The crashing task is retried once (penalized), crashes again, then fails.
        with pytest.raises(RuntimeError):
            crash_fut.result(timeout=15)

        # Result delivery is intact for other/subsequent tasks.
        futures = [ex.submit(task_fns.add, i, i) for i in range(8)]
        assert [f.result(timeout=15) for f in futures] == [i + i for i in range(8)]


# --- deterministic fake-level tests ----------------------------------------


def _make_executor(max_workers=4, **kw):
    """Executor whose ``_spawn_worker`` builds in-process fakes (no subprocess).

    Every slot gets a ``FakeQueue`` result queue so the collector sweep and
    queue-lifecycle bookkeeping can be exercised deterministically.
    """
    ex = AdaptiveExecutor(max_workers=max_workers, **kw)

    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True, pid=1000 + wid),
            work_queue=FakeQueue(),
            result_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]
    return ex


def _make_result(work_id, worker_id, value):
    return WorkResult(
        id=work_id,
        worker_id=worker_id,
        success=True,
        result=value,
        exception=None,
        observation=ResourceObservation(0.0, 0.0, 0.0, 0.0),
    )


def _load_inflight(ex, slot, work_id, value, fn_name="add"):
    """Register an in-flight task on ``slot`` and queue its result. Returns the
    future the collector should resolve."""
    future: Future = Future()
    item = WorkItem(
        id=work_id,
        fn_module="task_fns",
        fn_name=fn_name,
        args=(),
        kwargs={},
        gpu_id=None,
    )
    pending = PendingWork(
        item=item,
        future=future,
        estimate=ResourceEstimate(memory_gb=0.1, vram_gb=0.0, cpu_cores=1.0),
        submitted_at=0.0,
        worker_id=slot.worker_id,
    )
    ex.in_flight[work_id] = pending
    slot.current_work_id = work_id
    slot.result_queue.put(_make_result(work_id, slot.worker_id, value))
    return future


def test_sweep_collects_from_every_live_worker_queue():
    ex = _make_executor()
    a = ex._spawn_worker(None)
    b = ex._spawn_worker(None)
    fa = _load_inflight(ex, a, "wa", 11)
    fb = _load_inflight(ex, b, "wb", 22)

    processed = ex._sweep_result_queues()

    assert processed is True
    assert fa.result(timeout=0) == 11
    assert fb.result(timeout=0) == 22
    assert ex.in_flight == {}
    # Queues emptied by the sweep.
    assert a.result_queue.items == []
    assert b.result_queue.items == []
    # Idle sweep reports no work.
    assert ex._sweep_result_queues() is False


def test_poisoned_queue_is_never_swept_and_discarded_without_drain():
    ex = _make_executor(task_timeout_seconds=1.0, on_timeout="kill_worker")
    victim = ex._spawn_worker(None)
    survivor = ex._spawn_worker(None)

    # Both have a result queued; the victim is about to be killed on timeout.
    victim_fut = _load_inflight(ex, victim, "vic", 1, fn_name="busy_sleep")
    survivor_fut = _load_inflight(ex, survivor, "srv", 2)
    victim_pending = ex.in_flight["vic"]
    victim_pending.started_at = 0.0

    ex._handle_timeout(victim_pending)

    # Victim poisoned + evicted to retiring; SIGTERM delivered to its process.
    assert victim.queue_poisoned is True
    assert victim.worker_id not in ex.workers
    assert victim in ex._retiring
    assert victim.process.terminated is True
    # Its task failed with a timeout (not from its poisoned queue).
    assert isinstance(victim_fut.exception(timeout=0), TimeoutError)

    # The sweep still delivers the survivor and NEVER reads the poisoned queue.
    assert ex._sweep_result_queues() is True
    assert survivor_fut.result(timeout=0) == 2
    # Poisoned queue was skipped: its queued result is untouched by the sweep.
    assert victim.result_queue.items == [_make_result("vic", victim.worker_id, 1)]

    # Reaping discards the poisoned queue WITHOUT draining it.
    victim.process.set_dead(exitcode=-15)
    ex._check_workers()
    assert ex._retiring == []
    assert victim.queue_discarded is True
    assert victim.result_queue.closed is True
    # Never drained: the stale result was discarded with the worker.
    assert victim.result_queue.items == [_make_result("vic", victim.worker_id, 1)]


def test_graceful_retire_reap_drains_final_result():
    """A recycled/evicted worker may have delivered a final result before exiting
    on its sentinel; the reap must drain it (no loss), then discard."""
    ex = _make_executor(worker_recycle_after_tasks=1)
    worker = ex._spawn_worker(None)
    final_fut = _load_inflight(ex, worker, "last", 99)

    # Retire gracefully (e.g. recycle/eviction): still in-flight result unread.
    ex._retire_worker(worker, reason="recycle")
    assert worker.queue_poisoned is False
    assert worker in ex._retiring

    # Worker exits on its sentinel; reap drains the pending result.
    worker.process.set_dead(exitcode=0)
    ex._check_workers()

    assert final_fut.result(timeout=0) == 99
    assert ex.in_flight == {}
    assert ex._retiring == []
    assert worker.queue_discarded is True
    assert worker.result_queue.closed is True
    assert worker.result_queue.items == []


def test_handle_dead_worker_drains_earlier_result_then_handles_crash():
    """A crashed worker may have delivered a result for an EARLIER task before
    dying; that result must be drained (future resolved), while the task it was
    actually running is treated as a crash loss."""
    ex = _make_executor(max_resource_crash_retries=0)
    worker = ex._spawn_worker(None)

    # An earlier task's result sits unread in the queue...
    earlier_fut = _load_inflight(ex, worker, "earlier", 7)
    # ...but the worker is now running (and crashes on) a different task.
    current_fut: Future = Future()
    current_item = WorkItem(
        id="current",
        fn_module="task_fns",
        fn_name="add",
        args=(),
        kwargs={},
        gpu_id=None,
    )
    ex.in_flight["current"] = PendingWork(
        item=current_item,
        future=current_fut,
        estimate=ResourceEstimate(memory_gb=0.1, vram_gb=0.0, cpu_cores=1.0),
        submitted_at=0.0,
        worker_id=worker.worker_id,
    )
    worker.current_work_id = "current"

    # Worker crashed (non-retryable exit code with retries disabled).
    worker.process.set_dead(exitcode=1)
    ex._check_workers()

    # Earlier task delivered via drain; current task failed as a crash loss.
    assert earlier_fut.result(timeout=0) == 7
    assert isinstance(current_fut.exception(timeout=0), RuntimeError)
    assert ex.in_flight == {}
    assert worker.queue_discarded is True
    assert worker.result_queue.closed is True


def test_discard_queue_closes_cancels_and_is_idempotent():
    ex = _make_executor()
    worker = ex._spawn_worker(None)

    ex._discard_queue(worker)
    assert worker.queue_discarded is True
    assert worker.result_queue.closed is True
    assert worker.result_queue.cancel_join_thread_called is True

    # Idempotent: a second discard does not raise and does not re-run.
    worker.result_queue.closed = False
    ex._discard_queue(worker)
    assert worker.result_queue.closed is False


def test_kill_escalates_to_sigkill_when_sigterm_ignored():
    ex = _make_executor(task_timeout_seconds=1.0, on_timeout="kill_worker")
    ex._kill_grace_seconds = 0.0  # do not actually wait in the test
    worker = ex._spawn_worker(None)
    worker.process.ignore_terminate = True  # survives SIGTERM

    fut = _load_inflight(ex, worker, "doomed", 1, fn_name="busy_sleep")
    pending = ex.in_flight["doomed"]
    pending.started_at = 0.0

    ex._handle_timeout(pending)

    assert worker.process.terminated is True
    assert worker.process.killed is True  # escalated to SIGKILL
    assert isinstance(fut.exception(timeout=0), TimeoutError)
