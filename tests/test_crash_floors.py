"""Crash floors: a persistent memory floor learned from an OOM/SIGKILL crash.

The crashing run reports no observation (the worker died), so without this the
profile learns nothing and the next batch of fresh submits repeats the crash. A
floor lower-bounds ``estimate()``'s memory AFTER the safety margin, ratchets
upward while active, persists through the JSON store (older floor-less files
still load), and is cleared after a documented recovery rule so a shrunk
workload is not penalized forever.
"""

import json

from adaptive_executor.dtypes import ResourceObservation
from adaptive_executor.profiles import (
    FLOOR_RECOVERY_FRACTION,
    FLOOR_RECOVERY_OBSERVATIONS,
    LearnedProfile,
    ProfileStore,
    derive_store_key,
)


def _obs(mem=1.0):
    return ResourceObservation(
        memory_delta_gb=mem, vram_delta_gb=0.0, cpu_percent=50.0, duration_seconds=0.1
    )


# --- estimate lower-bounding -------------------------------------------------


def test_floor_lower_bounds_estimate_after_safety_margin():
    # Empty profile: default estimate is 1.0 * 1.5 safety = 1.5 GB. A 5 GB floor
    # raises it to 5 GB.
    prof = LearnedProfile()
    assert prof.estimate().memory_gb < 5.0
    prof.set_memory_floor(5.0)
    assert prof.estimate().memory_gb == 5.0


def test_floor_wins_over_optimistic_hint():
    # An explicit hint below the floor is still floored: OOM knowledge is fact.
    prof = LearnedProfile()
    prof.set_memory_floor(8.0)
    assert prof.estimate(memory_hint=1.0).memory_gb == 8.0


def test_floor_below_computed_estimate_is_noop():
    # A floor below the already-computed estimate does not lower it.
    prof = LearnedProfile()
    for _ in range(10):
        prof.add(_obs(mem=20.0))  # confident, ~20 GB estimate
    computed = prof.estimate().memory_gb
    prof.set_memory_floor(2.0)
    assert prof.estimate().memory_gb == computed


# --- ratcheting --------------------------------------------------------------


def test_floor_ratchets_up_only():
    prof = LearnedProfile()
    prof.set_memory_floor(4.0)
    assert prof.memory_floor_gb == 4.0
    prof.set_memory_floor(6.0)  # higher -> ratchets up
    assert prof.memory_floor_gb == 6.0
    prof.set_memory_floor(3.0)  # lower -> ignored
    assert prof.memory_floor_gb == 6.0


# --- recovery ----------------------------------------------------------------


def test_floor_cleared_after_consecutive_low_observations():
    prof = LearnedProfile()
    prof.set_memory_floor(10.0)
    threshold = 10.0 * FLOOR_RECOVERY_FRACTION  # 5.0
    # One short of the recovery count: floor stays.
    for _ in range(FLOOR_RECOVERY_OBSERVATIONS - 1):
        prof.add(_obs(mem=threshold - 1.0))
    assert prof.memory_floor_gb == 10.0
    # The final low observation clears it.
    prof.add(_obs(mem=threshold - 1.0))
    assert prof.memory_floor_gb is None


def test_high_observation_resets_the_recovery_streak():
    prof = LearnedProfile()
    prof.set_memory_floor(10.0)
    low = 10.0 * FLOOR_RECOVERY_FRACTION - 1.0
    for _ in range(FLOOR_RECOVERY_OBSERVATIONS - 1):
        prof.add(_obs(mem=low))
    # A high peak (above half the floor) breaks the streak.
    prof.add(_obs(mem=9.0))
    assert prof.memory_floor_gb == 10.0
    # Need another full run of lows to recover.
    for _ in range(FLOOR_RECOVERY_OBSERVATIONS - 1):
        prof.add(_obs(mem=low))
    assert prof.memory_floor_gb == 10.0
    prof.add(_obs(mem=low))
    assert prof.memory_floor_gb is None


# --- persistence -------------------------------------------------------------


def test_floor_persists_and_round_trips(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)
    store.record("m", "f", _obs(mem=1.0))
    store.record_memory_floor("m", "f", 7.0)
    store.flush()

    data = json.loads(path.read_text())
    entry = data[derive_store_key("m", "f")]
    assert entry["memory_floor_gb"] == 7.0

    reloaded = ProfileStore(persist_path=path)
    prof = reloaded.get("m", "f")
    assert prof.memory_floor_gb == 7.0
    assert prof.estimate().memory_gb >= 7.0


def test_floorless_profile_persists_as_plain_list(tmp_path):
    # Backward/forward compatible: a profile without a floor is still stored as a
    # plain observation list (so older readers and existing tests are unaffected).
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=1)
    store.record("m", "f", _obs(mem=1.0))
    store.flush()
    data = json.loads(path.read_text())
    assert isinstance(data[derive_store_key("m", "f")], list)


def test_old_format_file_without_floor_field_loads(tmp_path):
    # A file written before floors existed (plain lists) must still load, with
    # no floor.
    path = tmp_path / "p.json"
    old = {derive_store_key("m", "f"): [
        {"memory_delta_gb": 2.0, "vram_delta_gb": 0.0, "cpu_percent": 50.0, "duration_seconds": 0.1}
    ]}
    path.write_text(json.dumps(old))
    store = ProfileStore(persist_path=path)
    prof = store.get("m", "f")
    assert prof.sample_count == 1
    assert prof.memory_floor_gb is None


def test_loaded_floor_not_spuriously_recovered(tmp_path):
    # Loading observations that are all below half the floor must NOT clear the
    # stored floor during load (the floor is set after the observations).
    path = tmp_path / "p.json"
    store = ProfileStore(persist_path=path, save_every_n=100)
    for _ in range(FLOOR_RECOVERY_OBSERVATIONS + 2):
        store.record("m", "f", _obs(mem=1.0))
    store.record_memory_floor("m", "f", 10.0)  # 1.0 << 5.0 threshold
    store.flush()

    reloaded = ProfileStore(persist_path=path)
    assert reloaded.get("m", "f").memory_floor_gb == 10.0


# --- store dual-write --------------------------------------------------------


def test_record_memory_floor_writes_base_and_keyed():
    store = ProfileStore()
    store.record_memory_floor("m", "f", 5.0, profile_key="large")
    assert store.profiles[derive_store_key("m", "f")].memory_floor_gb == 5.0
    assert store.profiles[derive_store_key("m", "f", "large")].memory_floor_gb == 5.0


def test_get_copy_carries_floor():
    store = ProfileStore()
    store.record_memory_floor("m", "f", 5.0)
    assert store.get("m", "f").memory_floor_gb == 5.0
