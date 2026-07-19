from dataclasses import dataclass, field
from typing import Any

@dataclass
class ResourceSnapshot:
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpus: dict[int, "GPUSnapshot"] = field(default_factory=dict)

@dataclass
class GPUSnapshot:
    device_id: int
    vram_used_gb: float
    vram_total_gb: float
    utilization_percent: float

@dataclass
class ResourceObservation:
    memory_delta_gb: float
    vram_delta_gb: float
    cpu_percent: float
    duration_seconds: float

@dataclass
class ResourceEstimate:
    memory_gb: float
    vram_gb: float
    cpu_cores: float
    confidence: float = 0.0
    # p90 of observed run durations, or None when the function has no observed
    # history. None means "unknown duration" and is treated as INFINITE for
    # backfill-scheduling purposes (the task is assumed to never release).
    duration_p90_seconds: float | None = None

@dataclass(frozen=True)
class WorkItem:
    """Serializable work unit sent to worker processes"""
    id: str
    fn_module: str
    fn_name: str
    args: tuple
    kwargs: dict
    gpu_id: int | None

@dataclass
class WorkResult:
    """Result sent back from worker process"""
    id: str
    worker_id: int
    success: bool
    result: Any | None
    exception: BaseException | None
    observation: ResourceObservation
