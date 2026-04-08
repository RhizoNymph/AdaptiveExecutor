import json
import threading
import tempfile
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from .dtypes import ResourceObservation, ResourceEstimate

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
        # Correct percentile index calculation
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

        # Apply safety margin when confidence is low
        # At confidence=0, add 50% margin; at confidence=1, no margin
        safety_multiplier = 1.0 + 0.5 * (1.0 - confidence)

        return ResourceEstimate(
            memory_gb=max(0.1, memory * safety_multiplier),
            vram_gb=max(0.0, vram * safety_multiplier),
            cpu_cores=max(0.1, cpu_cores),  # CPU doesn't need safety margin
            confidence=confidence,
        )


class ProfileStore:
    """Thread-safe store for learned profiles with optional persistence"""
    
    def __init__(self, persist_path: str | Path | None = None):
        self.profiles: dict[str, LearnedProfile] = defaultdict(LearnedProfile)
        self.lock = threading.Lock()
        self.persist_path = Path(persist_path) if persist_path else None
        
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def fn_key(self, fn_module: str, fn_name: str) -> str:
        return f"{fn_module}:{fn_name}"
    
    def get(self, fn_module: str, fn_name: str) -> LearnedProfile:
        """
        Get a snapshot copy of the profile for estimation purposes.
        Returns a copy to avoid race conditions when reading observations.
        """
        with self.lock:
            profile = self.profiles[self.fn_key(fn_module, fn_name)]
            # Return a copy with a snapshot of observations
            copy = LearnedProfile(max_observations=profile.max_observations)
            copy.observations = list(profile.observations)  # Shallow copy of list
            return copy
    
    def record(self, fn_module: str, fn_name: str, observation: ResourceObservation):
        with self.lock:
            self.profiles[self.fn_key(fn_module, fn_name)].add(observation)
            self._maybe_persist()
    
    def _maybe_persist(self):
        """Persist on every observation to avoid data loss"""
        if self.persist_path is None:
            return
        self._save()
    
    def _save(self):
        """
        Atomically persist profiles to disk.
        Uses write-to-temp-then-rename pattern to prevent corruption on crash.
        """
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: [asdict(o) for o in profile.observations]
            for key, profile in self.profiles.items()
        }
        
        # Write to temp file, then atomically rename
        # This prevents data corruption if the process crashes mid-write
        fd, temp_path = tempfile.mkstemp(
            dir=self.persist_path.parent, 
            prefix=".profiles_",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self.persist_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    
    def _load(self):
        try:
            data = json.loads(self.persist_path.read_text())
            for key, obs_list in data.items():
                profile = LearnedProfile()
                for o in obs_list:
                    profile.add(ResourceObservation(**o))
                self.profiles[key] = profile
        except Exception:
            pass 