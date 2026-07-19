"""Bug 4: debounced persistence (N observations or T seconds), flush, atomic
writes. Bug 5: corrupt profile file and save failures are logged, not fatal."""

import json
import logging

from adaptive_executor.dtypes import ResourceObservation
from adaptive_executor.profiles import ProfileStore


def _obs(mem=1.0):
    return ResourceObservation(memory_delta_gb=mem, vram_delta_gb=0.0, cpu_percent=50.0, duration_seconds=0.1)


def _count_writes(store):
    calls = []
    orig = store._write_atomic

    def counting(data):
        calls.append(data)
        orig(data)

    store._write_atomic = counting
    return calls


def test_no_write_before_threshold(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=5)
    calls = _count_writes(store)
    for _ in range(4):
        store.record("m", "f", _obs())
    assert calls == []
    assert not path.exists()


def test_write_exactly_once_on_crossing_n(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=5)
    calls = _count_writes(store)
    for _ in range(5):
        store.record("m", "f", _obs())
    assert len(calls) == 1
    assert path.exists()


def test_time_based_debounce_with_injected_clock(tmp_path):
    path = tmp_path / "p.json"
    now = [0.0]
    store = ProfileStore(
        persist_path=path,
        save_every_n=1000,
        save_interval_seconds=5.0,
        clock=lambda: now[0],
    )
    calls = _count_writes(store)
    store.record("m", "f", _obs())  # t=0, no save
    assert calls == []
    now[0] = 6.0
    store.record("m", "f", _obs())  # interval elapsed -> save
    assert len(calls) == 1


def test_flush_forces_write(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1000)
    calls = _count_writes(store)
    store.record("m", "f", _obs())
    assert calls == []
    store.flush()
    assert len(calls) == 1
    assert path.exists()


def test_flush_noop_without_persist_path():
    store = ProfileStore(persist_path=None)
    store.record("m", "f", _obs())
    store.flush()  # must not raise


def test_atomic_write_leaves_no_tempfiles(tmp_path):
    path = tmp_path / "sub" / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)
    store.record("m", "f", _obs())
    assert path.exists()
    # Data is valid JSON and keyed by fn.
    data = json.loads(path.read_text())
    assert "m:f" in data
    # No leftover temp files.
    leftovers = list(path.parent.glob(".profiles_*"))
    assert leftovers == []


def test_load_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)
    store.record("m", "f", _obs(mem=2.5))
    store.flush()

    reloaded = ProfileStore(persist_path=path)
    prof = reloaded.get("m", "f")
    assert prof.sample_count == 1
    assert prof.observations[0].memory_delta_gb == 2.5


def test_corrupt_file_logs_warning_and_starts_empty(tmp_path, caplog):
    path = tmp_path / "p.json"
    path.write_text("{ this is not valid json ]]")
    with caplog.at_level(logging.WARNING, logger="adaptive_executor.profiles"):
        store = ProfileStore(persist_path=path)
    assert any("failed to load profiles" in r.message for r in caplog.records)
    assert store.get("m", "f").sample_count == 0


def test_save_failure_logs_error_not_raises(tmp_path, caplog):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)

    def boom(data):
        raise OSError("disk full")

    store._write_atomic = boom
    with caplog.at_level(logging.ERROR, logger="adaptive_executor.profiles"):
        # Must not raise into the record path.
        store.record("m", "f", _obs())
    assert any("profile save failed" in r.message for r in caplog.records)
