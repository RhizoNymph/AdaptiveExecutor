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


@dataclass
class LearnedProfile:
    observations: list[ResourceObservation] = field(default_factory=list)
    max_observations: int = 100

    def add(self, obs: ResourceObservation):
        self.observations.append(obs)
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]

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

        confidence = min(1.0, self.sample_count / 10.0)

        # Apply safety margin when confidence is low.
        # At confidence=0, add 50% margin; at confidence=1, no margin.
        safety_multiplier = 1.0 + 0.5 * (1.0 - confidence)

        return ResourceEstimate(
            memory_gb=max(0.1, memory * safety_multiplier),
            vram_gb=max(0.0, vram * safety_multiplier),
            cpu_cores=max(0.1, cpu_cores),  # CPU doesn't need safety margin
            confidence=confidence,
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
        return f"{fn_module}:{fn_name}"

    def get(self, fn_module: str, fn_name: str) -> LearnedProfile:
        """Return a snapshot copy of the profile for estimation purposes.

        Returns a copy to avoid race conditions when reading observations.
        """
        with self.lock:
            profile = self.profiles[self.fn_key(fn_module, fn_name)]
            copy = LearnedProfile(max_observations=profile.max_observations)
            copy.observations = list(profile.observations)  # shallow copy of list
            return copy

    def record(self, fn_module: str, fn_name: str, observation: ResourceObservation):
        snapshot: dict[str, list[dict]] | None = None
        with self.lock:
            self.profiles[self.fn_key(fn_module, fn_name)].add(observation)
            self._pending_since_save += 1
            if self._should_save_locked():
                snapshot = self._snapshot_locked()
                self._pending_since_save = 0
                self._last_save_time = self._clock()

        if snapshot is not None:
            self._persist_snapshot(snapshot)

    def flush(self):
        """Force an immediate persist of the current profiles (if configured)."""
        snapshot: dict[str, list[dict]] | None = None
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

    def _snapshot_locked(self) -> dict[str, list[dict]]:
        return {
            key: [asdict(o) for o in profile.observations]
            for key, profile in self.profiles.items()
        }

    def _persist_snapshot(self, data: dict[str, list[dict]]):
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

    def _write_atomic(self, data: dict[str, list[dict]]):
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
            for key, obs_list in data.items():
                profile = LearnedProfile()
                for o in obs_list:
                    profile.add(ResourceObservation(**o))
                self.profiles[key] = profile
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "failed to load profiles, starting empty path=%s error=%r",
                self.persist_path,
                exc,
            )
            self.profiles = defaultdict(LearnedProfile)
