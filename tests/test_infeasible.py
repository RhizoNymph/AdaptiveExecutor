"""Infeasible-task detection: a task whose estimate can never fit on this
machine must fail (submit-time raise or dispatch-time future failure) instead of
silently blocking the FIFO queue forever."""

import pytest

from adaptive_executor import AdaptiveExecutor, InfeasibleTaskError
from adaptive_executor.adaptive_executor import PendingWork, WorkerSlot
from adaptive_executor.dtypes import GPUSnapshot, ResourceEstimate, ResourceSnapshot, WorkItem
from fakes import FakeProcess, FakeQueue, StubSystemMonitor

import task_fns


def _snapshot(*, memory_total_gb=32.0, memory_used_gb=4.0, gpus=None):
    return ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        gpus=gpus or {},
    )


def _gpu(device_id=0, total=16.0, used=1.0):
    return GPUSnapshot(
        device_id=device_id,
        vram_used_gb=used,
        vram_total_gb=total,
        utilization_percent=0.0,
    )


def _make_executor(max_workers=4, snapshot=None, gpu_ids=None, **kw):
    """Executor wired with a stub monitor and a non-spawning worker factory."""
    ex = AdaptiveExecutor(max_workers=max_workers, gpu_ids=gpu_ids, **kw)
    ex.monitor = StubSystemMonitor(snapshot=snapshot)
    ex._started = True  # bypass real start(): no threads, no real monitor

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


def _pending(estimate, retry_count=0):
    from concurrent.futures import Future

    item = WorkItem(
        id=f"work-{id(estimate)}",
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
        retry_count=retry_count,
    )


# --- submit-time raise ------------------------------------------------------


def test_submit_raises_infeasible_memory():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0), memory_headroom_gb=2.0)
    # 64 GB hint -> estimate 96 GB (no observations -> 1.5x safety); capacity 30.
    with pytest.raises(InfeasibleTaskError) as excinfo:
        ex.submit(task_fns.echo, 1, memory_gb=64.0)

    err = excinfo.value
    assert err.kind == "memory"
    assert err.capacity_gb == pytest.approx(30.0)
    assert err.estimate_gb > err.capacity_gb
    assert err.retry_count == 0
    # Nothing was enqueued.
    assert len(ex.pending) == 0


def test_submit_raises_infeasible_vram():
    ex = _make_executor(
        snapshot=_snapshot(gpus={0: _gpu(total=16.0)}),
        gpu_ids=[0],
        vram_headroom_gb=1.0,
    )
    # 40 GB vram hint -> 60 GB estimate; largest GPU capacity is 16 - 1 = 15.
    with pytest.raises(InfeasibleTaskError) as excinfo:
        ex.submit(task_fns.echo, 1, vram_gb=40.0)

    err = excinfo.value
    assert err.kind == "vram"
    assert err.capacity_gb == pytest.approx(15.0)
    assert err.estimate_gb > err.capacity_gb
    assert len(ex.pending) == 0


def test_submit_feasible_task_is_enqueued():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0), memory_headroom_gb=2.0)
    fut = ex.submit(task_fns.echo, 1, memory_gb=1.0)
    assert not fut.done()
    assert len(ex.pending) == 1


def test_submit_no_infeasibility_when_capacity_unknown():
    # No snapshot at all (monitor.current and .snapshot() both None): capacity is
    # unknown, so infeasibility must NOT be declared; the task is enqueued.
    ex = _make_executor(snapshot=None, memory_headroom_gb=2.0)
    fut = ex.submit(task_fns.echo, 1, memory_gb=64.0)
    assert not fut.done()
    assert len(ex.pending) == 1


# --- dispatch-time future failure -------------------------------------------


def test_dispatch_fails_infeasible_head_and_continues():
    ex = _make_executor(
        max_workers=4,
        snapshot=_snapshot(memory_total_gb=32.0),
        memory_headroom_gb=2.0,
    )
    infeasible = _pending(ResourceEstimate(memory_gb=100.0, vram_gb=0.0, cpu_cores=1.0))
    feasible = _pending(ResourceEstimate(memory_gb=1.0, vram_gb=0.0, cpu_cores=1.0))
    ex.pending.append(infeasible)
    ex.pending.append(feasible)

    ex._maybe_dispatch()

    # The infeasible head failed its future rather than blocking the queue.
    assert infeasible.future.done()
    assert isinstance(infeasible.future.exception(), InfeasibleTaskError)
    # The queue continued: the feasible task got dispatched to a worker.
    assert feasible.item.id in ex.in_flight
    assert len(ex.pending) == 0
    assert not feasible.future.done()


def test_dispatch_penalized_retry_infeasible_carries_context():
    ex = _make_executor(
        max_workers=4,
        snapshot=_snapshot(memory_total_gb=32.0),
        memory_headroom_gb=2.0,
    )
    # A crash-penalized task: retry_count=1 and estimate doubled past capacity.
    penalized = _pending(
        ResourceEstimate(memory_gb=40.0, vram_gb=0.0, cpu_cores=1.0),
        retry_count=1,
    )
    ex.pending.append(penalized)

    ex._maybe_dispatch()

    err = penalized.future.exception()
    assert isinstance(err, InfeasibleTaskError)
    assert err.kind == "memory"
    assert err.retry_count == 1
    # Message distinguishes a crash-induced penalty from a bad user hint.
    assert "crash" in str(err).lower()


def test_dispatch_no_infeasibility_when_capacity_unknown():
    # monitor.current is None -> capacity unknown -> no infeasibility declared.
    # The head stays queued (normal backpressure), future untouched.
    ex = _make_executor(max_workers=4, snapshot=None)
    huge = _pending(ResourceEstimate(memory_gb=1000.0, vram_gb=0.0, cpu_cores=1.0))
    ex.pending.append(huge)

    ex._maybe_dispatch()

    assert not huge.future.done()
    # It was admitted (capacity unknown path) rather than failed.
    assert huge.item.id in ex.in_flight
