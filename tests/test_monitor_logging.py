"""Bug 5: ResourceMonitor NVML init/snapshot failures are logged at the right
levels instead of being silently swallowed."""

import logging

import adaptive_executor.monitor as monitor_mod
from adaptive_executor.monitor import ResourceMonitor
from fakes import make_fake_pynvml


def test_nvml_import_error_logs_debug(monkeypatch, caplog):
    def raise_import():
        raise ImportError("no pynvml")

    monkeypatch.setattr(monitor_mod, "_import_nvml", raise_import)
    m = ResourceMonitor()
    with caplog.at_level(logging.DEBUG, logger="adaptive_executor.monitor"):
        m._init_nvml()
    assert any("nvml unavailable" in r.message for r in caplog.records)
    assert m._gpu_handles == []


def test_nvml_init_failure_logs_warning(monkeypatch, caplog):
    fake = make_fake_pynvml(raise_init=True)
    monkeypatch.setattr(monitor_mod, "_import_nvml", lambda: fake)
    m = ResourceMonitor()
    with caplog.at_level(logging.WARNING, logger="adaptive_executor.monitor"):
        m._init_nvml()
    assert any("nvml init failed" in r.message for r in caplog.records)
    assert m._gpu_handles == []


def test_snapshot_device_failure_logs_debug_once(monkeypatch, caplog):
    fake = make_fake_pynvml(raise_mem=True)
    monkeypatch.setattr(monitor_mod, "_import_nvml", lambda: fake)
    m = ResourceMonitor()
    m._gpu_handles = [(0, object())]
    with caplog.at_level(logging.DEBUG, logger="adaptive_executor.monitor"):
        m.snapshot()
        m.snapshot()
    device_logs = [r for r in caplog.records if "gpu snapshot failed" in r.message]
    # At most once per device despite two poll cycles.
    assert len(device_logs) == 1
    assert 0 in m._warned_devices
