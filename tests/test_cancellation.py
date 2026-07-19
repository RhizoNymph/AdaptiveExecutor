"""Future.cancel() must follow standard concurrent.futures semantics.

Queued tasks are cancellable (never dispatched, queue slot freed); once a task
is dispatched the future is RUNNING and cancel() returns False. No cancelled or
already-done future may ever take a result/exception (that would raise
InvalidStateError and silently kill a background thread).
"""

import time
from concurrent.futures import Future

import pytest

from adaptive_executor import AdaptiveExecutor
from adaptive_executor.adaptive_executor import PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    ResourceEstimate,
    ResourceObservation,
    ResourceSnapshot,
    WorkItem,
    WorkResult,
)
from fakes import FakeProcess, FakeQueue, StubSystemMonitor

import task_fns


def _snapshot(*, memory_total_gb=32.0, memory_used_gb=4.0, gpus=None):
    return ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        gpus=gpus or {},
    )


def _make_executor(max_workers=4, snapshot=None, gpu_ids=None, **kw):
    """Executor wired with a stub monitor and a non-spawning worker factory.

    Mirrors the deterministic fake-executor pattern used elsewhere: no threads,
    no real monitor, no subprocesses.
    """
    if snapshot is None:
        snapshot = _snapshot()
    ex = AdaptiveExecutor(max_workers=max_workers, gpu_ids=gpu_ids, **kw)
    ex.monitor = StubSystemMonitor(snapshot=snapshot)
    ex._started = True  # bypass real start(): no threads, no real monitor

    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True, pid=1000 + wid),
            work_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]
    return ex


def _pending(estimate=None, work_id="work-1"):
    if estimate is None:
        estimate = ResourceEstimate(memory_gb=1.0, vram_gb=0.0, cpu_cores=1.0)
    item = WorkItem(
        id=work_id,
        fn_module="task_fns",
        fn_name="echo",
        args=(),
        kwargs={},
        gpu_id=None,
    )
    return PendingWork(
        item=item,
        future=Future(),
        estimate=estimate,
        submitted_at=0.0,
    )


def _observation():
    return ResourceObservation(
        memory_delta_gb=0.0, vram_delta_gb=0.0, cpu_percent=0.0, duration_seconds=0.0
    )


# --- cancel a queued task ----------------------------------------------------


def test_cancel_queued_task_is_never_dispatched():
    ex = _make_executor()
    pending = _pending()
    ex.pending.append(pending)

    assert pending.future.cancel() is True  # queued -> cancellable

    ex._maybe_dispatch()

    # Never dispatched, queue slot freed, recorded abandoned.
    assert pending.future.cancelled() is True
    assert pending.item.id not in ex.in_flight
    assert len(ex.pending) == 0
    assert pending.item.id in ex.completed_or_abandoned
    # No worker was handed the item (current_work_id stays None, no queue put).
    for worker in ex.workers.values():
        assert worker.current_work_id is None
        assert worker.work_queue.items == []


def test_dispatched_future_is_running_and_cannot_be_cancelled():
    # The set_running handshake decides exactly one winner: a task that reaches a
    # worker transitions PENDING -> RUNNING, so a later cancel() must return
    # False and the task stays in-flight (accounting is never yanked away).
    ex = _make_executor()
    pending = _pending()
    ex.pending.append(pending)

    ex._maybe_dispatch()

    assert pending.item.id in ex.in_flight
    assert pending.future.running() is True
    assert pending.future.cancel() is False
    assert pending.future.cancelled() is False


# --- background threads survive a done/cancelled future ----------------------


def test_process_result_survives_result_for_cancelled_future():
    # A result arriving for a future that is already done must be dropped, not
    # raise InvalidStateError (which would kill the collector thread).
    ex = _make_executor()
    pending = _pending()
    worker = ex._spawn_worker(None)
    worker.current_work_id = pending.item.id
    pending.worker_id = worker.worker_id
    ex.in_flight[pending.item.id] = pending

    # Simulate the future already being resolved out from under the collector.
    pending.future.set_result("already-done")

    result = WorkResult(
        id=pending.item.id,
        worker_id=worker.worker_id,
        success=True,
        result="late",
        exception=None,
        observation=_observation(),
    )

    # Must not raise; in-flight entry is popped; original value untouched.
    ex._process_result(result)

    assert pending.item.id not in ex.in_flight
    assert pending.future.result() == "already-done"


def test_process_result_survives_exception_for_done_future():
    ex = _make_executor()
    pending = _pending()
    worker = ex._spawn_worker(None)
    worker.current_work_id = pending.item.id
    pending.worker_id = worker.worker_id
    ex.in_flight[pending.item.id] = pending
    pending.future.set_result(123)

    result = WorkResult(
        id=pending.item.id,
        worker_id=worker.worker_id,
        success=False,
        result=None,
        exception=ValueError("late failure"),
        observation=_observation(),
    )

    ex._process_result(result)  # must not raise

    assert pending.item.id not in ex.in_flight
    assert pending.future.result() == 123


# --- sweep / shutdown tolerate cancelled entries -----------------------------


def test_fail_pending_futures_skips_cancelled_entry():
    ex = _make_executor()
    cancelled = _pending(work_id="cancelled")
    live = _pending(work_id="live")
    ex.pending.append(cancelled)
    ex.pending.append(live)

    assert cancelled.future.cancel() is True

    # Must not raise on the cancelled entry; the live one gets the exception.
    ex._fail_pending_futures(RuntimeError("shutting down"))

    assert cancelled.future.cancelled() is True
    assert isinstance(live.future.exception(), RuntimeError)
    assert len(ex.pending) == 0
    assert cancelled.item.id in ex.completed_or_abandoned
    assert live.item.id in ex.completed_or_abandoned


def test_infeasible_sweep_skips_cancelled_entry():
    # An entry that is both infeasible AND cancelled must be dropped as abandoned
    # without stuffing an InfeasibleTaskError into the cancelled future.
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0), memory_headroom_gb=2.0)
    infeasible = _pending(
        estimate=ResourceEstimate(memory_gb=100.0, vram_gb=0.0, cpu_cores=1.0),
        work_id="infeasible-cancelled",
    )
    ex.pending.append(infeasible)

    assert infeasible.future.cancel() is True

    ex._maybe_dispatch()  # must not raise

    # Stays cancelled — no exception was set on it — and it left the queue.
    assert infeasible.future.cancelled() is True
    assert infeasible.item.id not in ex.in_flight
    assert len(ex.pending) == 0
    assert infeasible.item.id in ex.completed_or_abandoned


# --- end-to-end smoke: cancel before dispatch, real subprocesses -------------


def test_real_subprocess_cancel_before_dispatch(tmp_path):
    profile_path = tmp_path / "profiles.json"
    # One worker: a slow task occupies it so a second task queues behind it and
    # can be cancelled before it is ever dispatched.
    with AdaptiveExecutor(max_workers=1, profile_path=str(profile_path)) as ex:
        blocker = ex.submit(task_fns.slow, 0.5)
        time.sleep(0.1)  # let the blocker take the only worker
        queued = ex.submit(task_fns.add, 2, 3)

        assert queued.cancel() is True
        assert queued.cancelled() is True

        assert blocker.result(timeout=30) == 0.5

    # The cancelled task never executed, so no observation was recorded for it.
    assert ex.profiles.get("task_fns", "add").sample_count == 0
