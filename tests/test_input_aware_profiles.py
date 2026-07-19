"""Input-aware resource profiles: an optional caller-supplied ``profile_key``
buckets a function's observations so an input-sensitive function (``f(small)``
vs ``f(huge)``) does not merge into one distribution.

Covers key derivation, keyed-vs-base estimation, fallback when a bucket is
empty, dual recording (base + keyed), keyed-profile persistence round-trip, and
executor-level plumbing from ``submit(..., profile_key=...)`` through to
``_process_result`` recording."""

import json

import pytest

from adaptive_executor import AdaptiveExecutor, InfeasibleTaskError
from adaptive_executor.adaptive_executor import PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    ResourceObservation,
    ResourceSnapshot,
    WorkResult,
)
from adaptive_executor.profiles import (
    LearnedProfile,
    ProfileStore,
    STORE_KEY_SEPARATOR,
    derive_store_key,
)
from fakes import FakeProcess, FakeQueue, StubSystemMonitor

import task_fns


def _obs(mem=1.0, dur=0.1):
    return ResourceObservation(
        memory_delta_gb=mem, vram_delta_gb=0.0, cpu_percent=50.0, duration_seconds=dur
    )


def _snapshot(*, memory_total_gb=32.0, memory_used_gb=4.0, gpus=None):
    return ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        gpus=gpus or {},
    )


def _make_executor(max_workers=4, snapshot=None, gpu_ids=None, **kw):
    """Executor wired with a stub monitor and a non-spawning worker factory,
    mirroring tests/test_infeasible.py's fixture."""
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


# --- store-key derivation ---------------------------------------------------


def test_derive_store_key_base_and_keyed():
    assert derive_store_key("m", "f") == "m:f"
    assert derive_store_key("m", "f", None) == "m:f"
    assert derive_store_key("m", "f", "large") == f"m:f{STORE_KEY_SEPARATOR}large"


def test_derive_store_key_is_pure():
    # Distinct keys map to distinct store keys; identical inputs are stable.
    assert derive_store_key("m", "f", "a") != derive_store_key("m", "f", "b")
    assert derive_store_key("m", "f", "a") == derive_store_key("m", "f", "a")


# --- estimation: keyed vs base, and fallback --------------------------------


def test_keyed_profile_preferred_when_it_has_observations():
    store = ProfileStore()
    # Base-only small observations (no key): base distribution stays small.
    for _ in range(30):
        store.record("m", "f", _obs(mem=1.0))
    # A few large observations in the "large" bucket (also flow into base).
    for _ in range(5):
        store.record("m", "f", _obs(mem=10.0), profile_key="large")

    base_est = store.get("m", "f").estimate()
    large_est = store.get("m", "f", profile_key="large").estimate()

    assert large_est.memory_gb > base_est.memory_gb


def test_fallback_to_base_when_keyed_bucket_empty():
    store = ProfileStore()
    for _ in range(10):
        store.record("m", "f", _obs(mem=3.0))

    keyed = store.get("m", "f", profile_key="never_seen")
    base = store.get("m", "f")

    assert keyed.sample_count == base.sample_count == 10
    assert keyed.estimate().memory_gb == base.estimate().memory_gb
    # The fallback path must not materialize an empty keyed entry.
    assert derive_store_key("m", "f", "never_seen") not in store.profiles


def test_no_key_uses_base_exactly_as_before():
    store = ProfileStore()
    for _ in range(10):
        store.record("m", "f", _obs(mem=2.0))
    assert store.get("m", "f").estimate().memory_gb == pytest.approx(2.0)


# --- dual recording ---------------------------------------------------------


def test_dual_recording_writes_base_and_keyed():
    store = ProfileStore()
    store.record("m", "f", _obs(mem=2.0), profile_key="k")

    assert derive_store_key("m", "f") in store.profiles
    assert derive_store_key("m", "f", "k") in store.profiles
    assert store.profiles[derive_store_key("m", "f")].sample_count == 1
    assert store.profiles[derive_store_key("m", "f", "k")].sample_count == 1


def test_no_key_records_base_only():
    store = ProfileStore()
    store.record("m", "f", _obs())
    assert derive_store_key("m", "f") in store.profiles
    assert not any(STORE_KEY_SEPARATOR in k for k in store.profiles)


# --- persistence round-trip -------------------------------------------------


def test_persistence_roundtrip_of_keyed_profiles(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)
    store.record("m", "f", _obs(mem=7.0), profile_key="large")
    store.flush()

    data = json.loads(path.read_text())
    assert derive_store_key("m", "f") in data
    assert derive_store_key("m", "f", "large") in data

    reloaded = ProfileStore(persist_path=path)
    keyed = reloaded.get("m", "f", profile_key="large")
    base = reloaded.get("m", "f")
    assert keyed.sample_count == 1
    assert base.sample_count == 1
    assert keyed.observations[0].memory_delta_gb == 7.0


# --- executor-level plumbing ------------------------------------------------


def test_submit_plumbs_profile_key_to_pending():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0))
    ex.submit(task_fns.echo, 1, profile_key="large")
    pending = ex.pending[0]
    assert pending.profile_key == "large"


def test_submit_dispatch_and_result_record_under_both_keys():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0))
    fut = ex.submit(task_fns.echo, 1, profile_key="large")

    ex._maybe_dispatch()
    assert len(ex.in_flight) == 1
    pending = next(iter(ex.in_flight.values()))
    assert pending.profile_key == "large"

    result = WorkResult(
        id=pending.item.id,
        worker_id=pending.worker_id,
        success=True,
        result=1,
        exception=None,
        observation=_obs(mem=2.5),
    )
    ex._process_result(result)

    assert fut.result(timeout=1) == 1
    base_key = derive_store_key("task_fns", "echo")
    keyed_key = derive_store_key("task_fns", "echo", "large")
    assert ex.profiles.profiles[base_key].sample_count == 1
    assert ex.profiles.profiles[keyed_key].sample_count == 1
    assert ex.profiles.profiles[keyed_key].observations[0].memory_delta_gb == 2.5


def test_submit_feasibility_uses_keyed_estimate():
    # Base profile is small (feasible); the "large" bucket is huge (infeasible).
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=32.0), memory_headroom_gb=2.0)
    small = LearnedProfile()
    large = LearnedProfile()
    for _ in range(10):
        small.add(_obs(mem=1.0))
        large.add(_obs(mem=40.0))
    ex.profiles.profiles[derive_store_key("task_fns", "echo")] = small
    ex.profiles.profiles[derive_store_key("task_fns", "echo", "large")] = large

    # Keyed estimate (~40 GB) exceeds capacity (30 GB) -> raised at submit.
    with pytest.raises(InfeasibleTaskError) as excinfo:
        ex.submit(task_fns.echo, 1, profile_key="large")
    assert excinfo.value.kind == "memory"
    assert len(ex.pending) == 0

    # Same function without a key uses the small base estimate -> enqueued.
    ex.submit(task_fns.echo, 1)
    assert len(ex.pending) == 1
