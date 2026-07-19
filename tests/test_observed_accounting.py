"""Executor-level tests for observed-usage commitment accounting.

Driven through the real ``AdaptiveExecutor`` with fake worker spawning and an
injected snapshot; the psutil/NVML sampling seams (``_sample_rss_bytes`` /
``_sample_process_vram_gb``) are overridden so no real process or GPU is touched.

Covers:
- baseline capture at dispatch,
- observed usage computed on refresh (memory + VRAM),
- fallback to observed=0 when the process is unreadable (full estimate stays
  committed -- the safe direction),
- committed totals / per-GPU maps crediting observed usage,
- the RunningEntry *release* projection = committed remainder (lock-in test),
- increased admission: a second task is admitted once the first's observed usage
  covers its estimate, while the headroom invariant still holds.
"""

import time
import uuid

import adaptive_executor.adaptive_executor as ae
from adaptive_executor.adaptive_executor import AdaptiveExecutor, PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    GPUSnapshot,
    ResourceEstimate,
    ResourceSnapshot,
    WorkItem,
)
from adaptive_executor.scheduling import DispatchPlan
from fakes import FakeProcess, FakeQueue

_GB = 1_000_000_000


def _make_executor(max_workers=4, **kw):
    ex = AdaptiveExecutor(max_workers=max_workers, **kw)

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


def _set_snapshot(ex, *, memory_total=100.0, memory_used=0.0, gpus=None):
    ex.monitor._current = ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used,
        memory_total_gb=memory_total,
        gpus=gpus or {},
    )


def _pending_work(*, mem, vram=0.0, cpu=1.0, duration=None, exclusive=False):
    item = WorkItem(
        id=str(uuid.uuid4()),
        fn_module="m",
        fn_name="f",
        args=(),
        kwargs={},
        gpu_id=None,
    )
    est = ResourceEstimate(
        memory_gb=mem,
        vram_gb=vram,
        cpu_cores=cpu,
        confidence=1.0,
        duration_p90_seconds=duration,
    )
    return PendingWork(item=item, future=ae.Future(), estimate=est, submitted_at=time.time(), exclusive=exclusive)


def _place_in_flight(
    ex,
    work,
    *,
    elapsed=0.0,
    gpu_id=None,
    pid=None,
    rss_baseline_bytes=None,
    vram_baseline_gb=None,
):
    work.started_at = time.time() - elapsed
    work.worker_id = -1
    work.assigned_gpu_id = gpu_id
    work.worker_pid = pid
    work.rss_baseline_bytes = rss_baseline_bytes
    work.vram_baseline_gb = vram_baseline_gb
    ex.in_flight[work.item.id] = work


# --- baseline capture at dispatch ------------------------------------------


def test_baseline_captured_at_dispatch():
    ex = _make_executor(max_workers=4)
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)
    ex._sample_rss_bytes = lambda pid: 3 * _GB  # type: ignore[method-assign]

    work = _pending_work(mem=5.0)
    ex.pending.append(work)
    ex._maybe_dispatch()

    assert work.item.id in ex.in_flight
    dispatched = ex.in_flight[work.item.id]
    assert dispatched.worker_pid is not None
    assert dispatched.rss_baseline_bytes == 3 * _GB
    # No usage realized yet at dispatch time.
    assert dispatched.observed_memory_gb == 0.0


def test_baseline_none_when_pid_unavailable_keeps_full_estimate():
    ex = _make_executor(max_workers=4)
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)

    # A worker whose process exposes no pid (e.g. a bare fake): baseline stays
    # None and the full estimate remains committed.
    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True, pid=None),
            work_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]

    work = _pending_work(mem=5.0)
    ex.pending.append(work)
    ex._maybe_dispatch()

    dispatched = ex.in_flight[work.item.id]
    assert dispatched.worker_pid is None
    assert dispatched.rss_baseline_bytes is None
    committed = ex._committed_resources()
    assert committed.memory_gb == 5.0  # full estimate, no observed credit


# --- observed usage computed on refresh ------------------------------------


def test_refresh_computes_observed_memory_above_baseline():
    ex = _make_executor()
    _set_snapshot(ex)
    work = _pending_work(mem=10.0)
    _place_in_flight(ex, work, pid=1234, rss_baseline_bytes=2 * _GB)
    # Current RSS is 2GB above baseline -> observed = 2GB.
    ex._sample_rss_bytes = lambda pid: 4 * _GB  # type: ignore[method-assign]

    ex._refresh_observations(time.time())
    assert work.observed_memory_gb == 2.0


def test_refresh_clamps_observed_to_zero_below_baseline():
    ex = _make_executor()
    _set_snapshot(ex)
    work = _pending_work(mem=10.0)
    _place_in_flight(ex, work, pid=1234, rss_baseline_bytes=5 * _GB)
    ex._sample_rss_bytes = lambda pid: 3 * _GB  # below baseline  # type: ignore[method-assign]

    ex._refresh_observations(time.time())
    assert work.observed_memory_gb == 0.0


def test_refresh_unreadable_process_falls_back_to_zero():
    ex = _make_executor()
    _set_snapshot(ex)
    work = _pending_work(mem=10.0)
    _place_in_flight(ex, work, pid=1234, rss_baseline_bytes=2 * _GB)
    # Simulate a prior good reading, then the process becomes unreadable.
    work.observed_memory_gb = 3.0
    ex._sample_rss_bytes = lambda pid: None  # type: ignore[method-assign]

    ex._refresh_observations(time.time() + 1.0)
    assert work.observed_memory_gb == 0.0  # conservative fallback


def test_refresh_is_throttled_by_interval():
    ex = _make_executor()
    _set_snapshot(ex)
    ex._observation_refresh_seconds = 0.1
    work = _pending_work(mem=10.0)
    _place_in_flight(ex, work, pid=1234, rss_baseline_bytes=0)

    calls = []
    ex._sample_rss_bytes = lambda pid: calls.append(pid) or 1 * _GB  # type: ignore[method-assign]

    t0 = 100.0
    ex._refresh_observations(t0)
    ex._refresh_observations(t0 + 0.05)  # within interval -> skipped
    assert len(calls) == 1
    ex._refresh_observations(t0 + 0.2)  # past interval -> sampled again
    assert len(calls) == 2


# --- observed VRAM ----------------------------------------------------------


def test_refresh_computes_observed_vram_per_gpu():
    ex = _make_executor()
    _set_snapshot(
        ex,
        gpus={
            0: GPUSnapshot(
                device_id=0, vram_used_gb=3.0, vram_total_gb=16.0, utilization_percent=0.0
            )
        },
    )
    work = _pending_work(mem=1.0, vram=8.0)
    _place_in_flight(ex, work, gpu_id=0, pid=1234, vram_baseline_gb=1.0)
    # Current per-process VRAM on gpu0 is 6.0; baseline 1.0 -> observed 5.0.
    ex._sample_process_vram_gb = lambda pid, gpu: 6.0  # type: ignore[method-assign]

    ex._refresh_observations(time.time())
    assert work.observed_vram_gb == 5.0
    per_gpu = ex._committed_vram_per_gpu()
    # committed vram on gpu0 = max(8 - 5, 0) = 3.
    assert per_gpu == {0: 3.0}


# --- committed totals credit observed --------------------------------------


def test_committed_resources_credit_observed_memory():
    ex = _make_executor()
    a = _pending_work(mem=10.0)
    b = _pending_work(mem=6.0)
    _place_in_flight(ex, a, pid=1, rss_baseline_bytes=0)
    _place_in_flight(ex, b, pid=2, rss_baseline_bytes=0)
    a.observed_memory_gb = 4.0  # remainder 6
    b.observed_memory_gb = 8.0  # overrun -> remainder 0

    committed = ex._committed_resources()
    assert committed.memory_gb == 6.0


# --- release projection lock-in (reservation conservatism) -----------------


def test_running_entry_release_is_committed_remainder():
    """The reservation is fed each running task's *committed remainder* as the
    amount it releases when it finishes -- never the full estimate once usage is
    realized, and never a negative on overrun. This is the conservative choice:
    the realized portion already sits in ``snapshot.used`` and is NOT assumed to
    return, so the head's reservation never assumes more frees than the
    accounting guarantees.
    """
    ex = _make_executor()
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)

    under = _pending_work(mem=10.0, duration=5.0)  # observed 4 -> releases 6
    over = _pending_work(mem=10.0, vram=8.0, duration=5.0)  # observed 15/20 -> releases 0
    _place_in_flight(ex, under, pid=1, rss_baseline_bytes=0)
    _place_in_flight(ex, over, pid=2, gpu_id=0, rss_baseline_bytes=0, vram_baseline_gb=0.0)
    # Observed usage measured (above zero baselines) via the sampling seams.
    ex._sample_rss_bytes = lambda pid: {1: 4 * _GB, 2: 15 * _GB}[pid]  # type: ignore[method-assign]
    ex._sample_process_vram_gb = lambda pid, gpu: 20.0  # type: ignore[method-assign]

    captured = {}

    def capture(pending, running, capacity):
        captured["running"] = {r.id: r for r in running}
        return DispatchPlan(decisions=(), next_gpu_index=capacity.next_gpu_index)

    import adaptive_executor.adaptive_executor as mod

    orig = mod.plan_dispatch
    mod.plan_dispatch = capture  # type: ignore[assignment]
    try:
        ex._build_dispatch_plan()
    finally:
        mod.plan_dispatch = orig

    r_under = captured["running"][under.item.id]
    r_over = captured["running"][over.item.id]
    assert r_under.memory_gb == 6.0  # max(10 - 4, 0)
    assert r_over.memory_gb == 0.0  # max(10 - 15, 0)
    assert r_over.vram_gb == 0.0  # max(8 - 20, 0)


# --- increased admission (the whole point) ---------------------------------


def _admission_scenario(*, observed):
    """Tight memory: one 10GB task already realized in the snapshot, plus a new
    10GB task. Returns (executor, new_work). ``observed`` is the running task's
    attributed usage."""
    ex = _make_executor(max_workers=4)
    # total 24, headroom 2, and the running task has already allocated 10GB
    # (reflected in memory_used).
    _set_snapshot(ex, memory_total=24.0, memory_used=10.0)

    running = _pending_work(mem=10.0, duration=100.0)
    _place_in_flight(ex, running, pid=555, rss_baseline_bytes=0)
    # observed usage measured via the sampler as growth above a zero baseline.
    ex._sample_rss_bytes = lambda pid: int(observed * _GB)  # type: ignore[method-assign]

    new_work = _pending_work(mem=10.0, duration=100.0)
    ex.pending.append(new_work)
    return ex, new_work


def test_second_task_blocked_without_observed_credit():
    # Control: the running task's usage is NOT yet realized as observed (sampler
    # reports the baseline), so the full 10GB stays committed and double-counts
    # against the snapshot -> the new task cannot be admitted.
    ex, new_work = _admission_scenario(observed=0.0)
    ex._maybe_dispatch()
    assert new_work.item.id not in ex.in_flight
    assert [p.item.id for p in ex.pending] == [new_work.item.id]


def test_second_task_admitted_once_observed_covers_estimate():
    # Fixed: the running task's 10GB is now observed, so its committed remainder
    # is 0 and the previously double-counted 10GB is freed for admission.
    ex, new_work = _admission_scenario(observed=10.0)
    ex._maybe_dispatch()
    assert new_work.item.id in ex.in_flight

    # Headroom invariant still holds: committed never exceeds usable memory.
    usable = 24.0 - 10.0 - 2.0  # total - used - headroom
    committed = ex._committed_resources()
    assert committed.memory_gb <= usable + 1e-9
