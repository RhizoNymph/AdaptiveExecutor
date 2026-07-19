"""Bug 2: VRAM must be attributed to the worker's pinned GPU only, preferring
per-process attribution and falling back to the pinned device's used memory."""

import adaptive_executor.monitor as monitor_mod
from adaptive_executor.dtypes import GPUSnapshot, ResourceSnapshot
from adaptive_executor.monitor import ResourceMonitor
from adaptive_executor.worker import Worker
from fakes import FakeComputeProc, FakeMonitor, FakeQueue, make_fake_pynvml


def _worker(pinned, monitor):
    w = Worker(FakeQueue(), FakeQueue(), worker_id=0, pinned_gpu_id=pinned)
    w.monitor = monitor
    return w


def test_unpinned_worker_records_zero_and_skips_gpu():
    mon = FakeMonitor(per_process={0: 5.0}, device={0: 9.0})
    w = _worker(None, mon)
    assert w._gpu_vram_gb() == 0.0
    # Unpinned workers must not query the GPU at all.
    assert mon.per_process_calls == []
    assert mon.device_calls == []


def test_pinned_worker_uses_only_its_device_per_process():
    # Device 1 has huge usage from another task; pinned worker on 0 must ignore it.
    mon = FakeMonitor(per_process={0: 1.5, 1: 99.0}, device={0: 4.0, 1: 99.0})
    w = _worker(0, mon)
    assert w._gpu_vram_gb() == 1.5
    assert mon.per_process_calls[0][0] == 0  # queried device 0 only


def test_pinned_worker_falls_back_to_device_used():
    mon = FakeMonitor(per_process={0: None}, device={0: 3.0})
    w = _worker(0, mon)
    assert w._gpu_vram_gb() == 3.0
    assert mon.device_calls == [0]


def test_pinned_worker_returns_zero_when_no_measurement():
    mon = FakeMonitor(per_process={0: None}, device={0: None})
    w = _worker(0, mon)
    assert w._gpu_vram_gb() == 0.0


def test_execute_with_observation_unpinned_vram_zero():
    mon = FakeMonitor(per_process={0: 5.0}, device={0: 5.0})
    w = _worker(None, mon)
    result = w._execute_with_observation(lambda: 123, (), {}, "wid")
    assert result.success
    assert result.result == 123
    assert result.observation.vram_delta_gb == 0.0


def test_execute_with_observation_pinned_positive_delta(monkeypatch):
    mon = FakeMonitor()
    w = _worker(0, mon)
    values = iter([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    def fake_vram():
        try:
            return next(values)
        except StopIteration:
            return 5.0

    monkeypatch.setattr(w, "_gpu_vram_gb", fake_vram)
    result = w._execute_with_observation(task_slow, (), {}, "wid")
    assert result.observation.vram_delta_gb >= 4.0


def task_slow():
    import time

    time.sleep(0.15)
    return "done"


# --- ResourceMonitor per-process API (v3/v2/unversioned) coverage -----------


def _monitor_with_fake(monkeypatch, fake):
    monkeypatch.setattr(monitor_mod, "_import_nvml", lambda: fake)
    m = ResourceMonitor()
    m._gpu_handles = [(0, object())]
    return m


def test_per_process_sums_only_matching_pids(monkeypatch):
    fake = make_fake_pynvml(
        procs=[FakeComputeProc(111, 2_000_000_000), FakeComputeProc(222, 5_000_000_000)]
    )
    m = _monitor_with_fake(monkeypatch, fake)
    assert m.per_process_vram_gb(0, {111}) == 2.0
    assert m.per_process_vram_gb(0, {111, 222}) == 7.0
    assert m.per_process_vram_gb(0, {999}) == 0.0


def test_per_process_falls_back_to_v2(monkeypatch):
    fake = make_fake_pynvml(
        procs=[FakeComputeProc(111, 3_000_000_000)], available=("v2", "plain")
    )
    m = _monitor_with_fake(monkeypatch, fake)
    assert m.per_process_vram_gb(0, {111}) == 3.0


def test_per_process_none_when_api_missing(monkeypatch):
    fake = make_fake_pynvml(procs=[FakeComputeProc(111, 1_000_000_000)], available=())
    m = _monitor_with_fake(monkeypatch, fake)
    assert m.per_process_vram_gb(0, {111}) is None


def test_per_process_none_when_query_raises(monkeypatch):
    fake = make_fake_pynvml(procs=[FakeComputeProc(111, 1e9)], raise_procs=True)
    m = _monitor_with_fake(monkeypatch, fake)
    assert m.per_process_vram_gb(0, {111}) is None


def test_per_process_none_for_unknown_device(monkeypatch):
    fake = make_fake_pynvml(procs=[FakeComputeProc(111, 1e9)])
    m = _monitor_with_fake(monkeypatch, fake)
    assert m.per_process_vram_gb(7, {111}) is None


def test_device_vram_used_from_snapshot():
    m = ResourceMonitor()
    m._current = ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=1.0,
        memory_total_gb=8.0,
        gpus={0: GPUSnapshot(device_id=0, vram_used_gb=6.5, vram_total_gb=16.0, utilization_percent=10)},
    )
    assert m.device_vram_used_gb(0) == 6.5
    assert m.device_vram_used_gb(3) is None
