import json
import logging
import os
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .dtypes import ResourceEstimate, ResourceObservation

logger = logging.getLogger("adaptive_executor.profiles")

# Separator between the base function key (``module:qualname``) and an optional
# caller-supplied ``profile_key``. User keys are opaque strings; escaping is not
# required. The convention is that the base key never contains this separator, so
# ``module:qualname#profile_key`` cannot collide with a plain ``module:qualname``
# key. Callers who want disjoint buckets should keep their keys free of ``#``
# themselves (two keys that differ only around a ``#`` are their responsibility).
STORE_KEY_SEPARATOR = "#"

# Crash-floor recovery: a persistent memory floor is cleared once this many
# consecutive successful observations all peak below ``FLOOR_RECOVERY_FRACTION``
# of the floor, so a workload that has genuinely shrunk is not penalized forever.
FLOOR_RECOVERY_OBSERVATIONS = 5
FLOOR_RECOVERY_FRACTION = 0.5


def derive_store_key(
    fn_module: str, fn_name: str, profile_key: str | None = None
) -> str:
    """Return the ProfileStore key for a function, optionally input-bucketed.

    This is the single place store keys are derived. Without a ``profile_key`` the
    key is the base ``module:qualname``; with one it is
    ``module:qualname#profile_key``. The base profile always lives at the
    separator-free key and remains the aggregate fallback.
    """
    base = f"{fn_module}:{fn_name}"
    if profile_key is None:
        return base
    return f"{base}{STORE_KEY_SEPARATOR}{profile_key}"


@dataclass
class LearnedProfile:
    observations: list[ResourceObservation] = field(default_factory=list)
    max_observations: int = 100
    # Persistent memory lower bound (GB) learned from an OOM/SIGKILL crash: the
    # estimate the task was admitted under was proven too small, so future
    # estimates for this profile are floored at this value even though the
    # crashing run reported no observation. ``None`` means no active floor.
    memory_floor_gb: float | None = None

    def add(self, obs: ResourceObservation):
        self.observations.append(obs)
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]
        self._maybe_recover_floor()

    def set_memory_floor(self, floor_gb: float) -> None:
        """Ratchet the persistent memory floor up to at least ``floor_gb``.

        Floors only ever ratchet upward while active (``max(old, new)``); they
        are lowered only by the recovery rule in :meth:`_maybe_recover_floor`.
        """
        previous = self.memory_floor_gb
        self.memory_floor_gb = floor_gb if previous is None else max(previous, floor_gb)
        logger.info(
            "memory floor set previous=%s new=%.3f",
            f"{previous:.3f}" if previous is not None else None,
            self.memory_floor_gb,
        )

    def _maybe_recover_floor(self) -> None:
        """Clear an active floor once ``FLOOR_RECOVERY_OBSERVATIONS`` consecutive
        successful observations all peak below ``FLOOR_RECOVERY_FRACTION`` of it,
        so a workload that has genuinely shrunk is not floored forever.
        """
        if self.memory_floor_gb is None:
            return
        recent = self.observations[-FLOOR_RECOVERY_OBSERVATIONS:]
        if len(recent) < FLOOR_RECOVERY_OBSERVATIONS:
            return
        threshold = self.memory_floor_gb * FLOOR_RECOVERY_FRACTION
        if all(o.memory_delta_gb < threshold for o in recent):
            logger.info(
                "memory floor cleared floor=%.3f after %d observations below %.3f",
                self.memory_floor_gb,
                FLOOR_RECOVERY_OBSERVATIONS,
                threshold,
            )
            self.memory_floor_gb = None

    @property
    def sample_count(self) -> int:
        return len(self.observations)

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int((len(sorted_vals) - 1) * p)
        return sorted_vals[idx]

    def estimate(self, memory_hint: float | None = None, vram_hint: float | None = None) -> ResourceEstimate:
        if memory_hint is not None:
            memory = memory_hint
        elif self.observations:
            memory = self.percentile([o.memory_delta_gb for o in self.observations], 0.9)
        else:
            memory = 1.0

        if vram_hint is not None:
            vram = vram_hint
        elif self.observations:
            vram = self.percentile([o.vram_delta_gb for o in self.observations], 0.9)
        else:
            vram = 0.0

        if self.observations:
            cpu_cores = sum(o.cpu_percent for o in self.observations) / len(self.observations) / 100.0
        else:
            cpu_cores = 1.0

        # p90 run duration, or None when there is no observed history. A function
        # with no history has an unknown duration, which backfill scheduling
        # treats as infinite (the task is assumed never to release resources).
        if self.observations:
            duration_p90 = self.percentile([o.duration_seconds for o in self.observations], 0.9)
        else:
            duration_p90 = None

        confidence = min(1.0, self.sample_count / 10.0)

        # Apply safety margin when confidence is low.
        # At confidence=0, add 50% margin; at confidence=1, no margin.
        safety_multiplier = 1.0 + 0.5 * (1.0 - confidence)

        # Persistent crash floor lower-bounds memory AFTER the safety margin (and
        # after any hint): OOM knowledge is empirical fact that a smaller estimate
        # was proven insufficient, so it wins over an optimistic hint/percentile.
        memory_out = max(0.1, memory * safety_multiplier)
        if self.memory_floor_gb is not None:
            memory_out = max(memory_out, self.memory_floor_gb)

        return ResourceEstimate(
            memory_gb=memory_out,
            vram_gb=max(0.0, vram * safety_multiplier),
            cpu_cores=max(0.1, cpu_cores),  # CPU doesn't need safety margin
            confidence=confidence,
            duration_p90_seconds=duration_p90,
        )


class ProfileStore:
    """Thread-safe store for learned profiles with debounced persistence.

    Persistence is debounced to avoid rewriting the entire JSON file on every
    observation while holding the store lock. A save is triggered when either
    ``save_every_n`` observations have accumulated since the last save, or
    ``save_interval_seconds`` have elapsed, whichever comes first. Data is
    serialized under the store lock but file I/O happens outside it, guarded by a
    separate save lock. Call :meth:`flush` to force an immediate save.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        save_every_n: int = 20,
        save_interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.profiles: dict[str, LearnedProfile] = defaultdict(LearnedProfile)
        self.lock = threading.Lock()
        self._save_lock = threading.Lock()
        self.persist_path = Path(persist_path) if persist_path else None
        self.save_every_n = save_every_n
        self.save_interval_seconds = save_interval_seconds
        self._clock = clock
        self._pending_since_save = 0
        self._last_save_time = self._clock()

        if self.persist_path and self.persist_path.exists():
            self._load()

    def fn_key(self, fn_module: str, fn_name: str) -> str:
        return derive_store_key(fn_module, fn_name)

    def _select_profile_locked(
        self, fn_module: str, fn_name: str, profile_key: str | None
    ) -> LearnedProfile:
        """Pick the profile to estimate from. Must hold ``self.lock``.

        When a ``profile_key`` is given, the input-bucketed profile is used only
        if it already has at least one observation; otherwise estimation falls
        back to the aggregate base profile (and its confidence/safety machinery).
        Reads the keyed entry via ``dict.get`` so an empty bucket is never
        materialized on the fallback path.
        """
        if profile_key is not None:
            keyed = self.profiles.get(
                derive_store_key(fn_module, fn_name, profile_key)
            )
            if keyed is not None and keyed.observations:
                return keyed
        return self.profiles[derive_store_key(fn_module, fn_name)]

    def get(
        self, fn_module: str, fn_name: str, profile_key: str | None = None
    ) -> LearnedProfile:
        """Return a snapshot copy of the profile for estimation purposes.

        Returns a copy to avoid race conditions when reading observations. With a
        ``profile_key`` the input-bucketed profile is preferred when it has any
        observations, else the base profile is used (see
        ``_select_profile_locked``).
        """
        with self.lock:
            profile = self._select_profile_locked(fn_module, fn_name, profile_key)
            copy = LearnedProfile(max_observations=profile.max_observations)
            copy.observations = list(profile.observations)  # shallow copy of list
            copy.memory_floor_gb = profile.memory_floor_gb
            return copy

    def estimation_identity(
        self, fn_module: str, fn_name: str, profile_key: str | None = None
    ) -> tuple[str, int]:
        """Return ``(store_key, sample_count)`` for the profile that estimation
        would actually draw from — the keyed bucket when it has observations,
        else the base aggregate.

        This is the profile *identity* used by the cold-start canary: a task is
        "cold" when the identity it estimates from has zero observations. A keyed
        submit that falls back to a base profile with observations therefore
        reports the base identity with a positive count (i.e. not cold). Reads
        via ``dict.get`` so no empty entry is materialized.
        """
        with self.lock:
            if profile_key is not None:
                keyed_key = derive_store_key(fn_module, fn_name, profile_key)
                keyed = self.profiles.get(keyed_key)
                if keyed is not None and keyed.observations:
                    return keyed_key, keyed.sample_count
            base_key = derive_store_key(fn_module, fn_name)
            base = self.profiles.get(base_key)
            return base_key, (base.sample_count if base is not None else 0)

    def record(
        self,
        fn_module: str,
        fn_name: str,
        observation: ResourceObservation,
        profile_key: str | None = None,
    ):
        """Record one observation into the base profile and, when a
        ``profile_key`` is supplied, additionally into the input-bucketed
        profile. The base profile always accumulates every observation so it
        remains the aggregate fallback.
        """
        snapshot: dict[str, object] | None = None
        with self.lock:
            self.profiles[derive_store_key(fn_module, fn_name)].add(observation)
            if profile_key is not None:
                self.profiles[
                    derive_store_key(fn_module, fn_name, profile_key)
                ].add(observation)
            self._pending_since_save += 1
            if self._should_save_locked():
                snapshot = self._snapshot_locked()
                self._pending_since_save = 0
                self._last_save_time = self._clock()

        if snapshot is not None:
            self._persist_snapshot(snapshot)

    def record_memory_floor(
        self,
        fn_module: str,
        fn_name: str,
        floor_gb: float,
        profile_key: str | None = None,
    ):
        """Persist a memory floor into the base profile and, when a
        ``profile_key`` is supplied, additionally into the keyed profile.

        Mirrors :meth:`record`'s dual-write so OOM knowledge survives even when a
        crash happened under an input-bucketed submit. Floors ratchet upward
        (never weakened here) and participate in the same debounced persistence.
        """
        snapshot: dict | None = None
        with self.lock:
            self.profiles[derive_store_key(fn_module, fn_name)].set_memory_floor(floor_gb)
            if profile_key is not None:
                self.profiles[
                    derive_store_key(fn_module, fn_name, profile_key)
                ].set_memory_floor(floor_gb)
            self._pending_since_save += 1
            if self._should_save_locked():
                snapshot = self._snapshot_locked()
                self._pending_since_save = 0
                self._last_save_time = self._clock()

        if snapshot is not None:
            self._persist_snapshot(snapshot)

    def flush(self):
        """Force an immediate persist of the current profiles (if configured)."""
        snapshot: dict[str, object] | None = None
        with self.lock:
            if self.persist_path is None:
                return
            snapshot = self._snapshot_locked()
            self._pending_since_save = 0
            self._last_save_time = self._clock()
        self._persist_snapshot(snapshot)

    def _should_save_locked(self) -> bool:
        if self.persist_path is None:
            return False
        if self._pending_since_save <= 0:
            return False
        if self._pending_since_save >= self.save_every_n:
            return True
        return (self._clock() - self._last_save_time) >= self.save_interval_seconds

    def _snapshot_locked(self) -> dict[str, object]:
        """Serialize profiles for persistence.

        A profile with no active floor is stored as a plain observation list
        (the original, forward/backward-compatible format). A profile carrying a
        memory floor is stored as ``{"observations": [...], "memory_floor_gb":
        X}`` so the floor round-trips. :meth:`_load` accepts both shapes, and
        older files (all plain lists) load unchanged.
        """
        result: dict[str, object] = {}
        for key, profile in self.profiles.items():
            obs = [asdict(o) for o in profile.observations]
            if profile.memory_floor_gb is None:
                result[key] = obs
            else:
                result[key] = {
                    "observations": obs,
                    "memory_floor_gb": profile.memory_floor_gb,
                }
        return result

    def _persist_snapshot(self, data: dict[str, object]):
        """Write ``data`` to disk atomically, logging (not raising) on failure."""
        if self.persist_path is None:
            return
        with self._save_lock:
            try:
                self._write_atomic(data)
            except (OSError, TypeError, ValueError) as exc:
                logger.error(
                    "profile save failed path=%s error=%r", self.persist_path, exc
                )

    def _write_atomic(self, data: dict[str, object]):
        """Atomically persist ``data`` using write-to-temp-then-rename."""
        assert self.persist_path is not None
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=self.persist_path.parent,
            prefix=".profiles_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.persist_path)
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _load(self):
        assert self.persist_path is not None
        try:
            data = json.loads(self.persist_path.read_text())
            for key, value in data.items():
                # Backward compatible: a plain list is the original
                # observations-only format; a dict additionally carries the
                # persisted memory floor. Files predating floors have no dicts.
                if isinstance(value, dict):
                    obs_list = value.get("observations", [])
                    floor = value.get("memory_floor_gb")
                else:
                    obs_list = value
                    floor = None
                profile = LearnedProfile()
                for o in obs_list:
                    profile.add(ResourceObservation(**o))
                # Set the floor AFTER loading observations so recovery is not
                # spuriously triggered against the just-restored history.
                profile.memory_floor_gb = floor
                self.profiles[key] = profile
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "failed to load profiles, starting empty path=%s error=%r",
                self.persist_path,
                exc,
            )
            self.profiles = defaultdict(LearnedProfile)
