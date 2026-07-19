"""Pure resource-commitment accounting for admission control.

A running task's *committed* resources are the portion of its estimate that has
NOT yet been realized in the live monitor snapshot. Once a task actually
allocates memory/VRAM, that usage already appears in the snapshot's ``used``
figure; counting the full estimate on top of it would double-count and severely
under-utilize the machine (a 10GB-estimate task that has already allocated its
10GB would otherwise still block another 10GB of admission).

We therefore commit only the unrealized remainder ``max(estimate - observed, 0)``
where ``observed`` is the task's current attributable usage. A task that overruns
its estimate contributes zero committed resources: its real usage is already in
the snapshot.

This module is pure and deterministic -- no executor state, no threads, no I/O --
so the accounting can be exhaustively unit-tested in isolation. It is the single
source of truth for both admission-time committed totals and the amount a running
task is projected to *release* when it finishes (its committed remainder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "ResourceUsage",
    "committed_gb",
    "total_committed_gb",
    "committed_vram_per_gpu",
]


def committed_gb(estimate_gb: float, observed_gb: float) -> float:
    """Unrealized remainder of ``estimate_gb`` given already-realized ``observed_gb``.

    Never negative: a task whose observed usage meets or exceeds its estimate
    contributes zero committed resources, because its real usage is already
    reflected in the live snapshot's ``used`` figure.
    """
    remainder = estimate_gb - observed_gb
    return remainder if remainder > 0.0 else 0.0


@dataclass(frozen=True)
class ResourceUsage:
    """One running task's (estimate, observed) usage for a single GPU dimension.

    ``gpu_id is None`` means the task is not pinned to a GPU and contributes no
    per-GPU VRAM commitment.
    """

    gpu_id: int | None
    estimate_gb: float
    observed_gb: float


def total_committed_gb(pairs: Iterable[tuple[float, float]]) -> float:
    """Sum of committed remainders over ``(estimate, observed)`` pairs."""
    return sum(committed_gb(estimate, observed) for estimate, observed in pairs)


def committed_vram_per_gpu(usages: Iterable[ResourceUsage]) -> dict[int, float]:
    """Per-GPU committed VRAM remainders.

    Entries whose ``gpu_id is None`` are ignored (an unpinned task holds no
    device VRAM). GPUs are keyed by NVML index.
    """
    result: dict[int, float] = {}
    for usage in usages:
        if usage.gpu_id is None:
            continue
        result[usage.gpu_id] = result.get(usage.gpu_id, 0.0) + committed_gb(
            usage.estimate_gb, usage.observed_gb
        )
    return result
