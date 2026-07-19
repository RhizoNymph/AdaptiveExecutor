"""Synthetic workloads for the scheduler simulation.

All generation is deterministic: fixed scenarios are hand-built and random
mixes are driven only by an explicit ``seed`` (``random.Random(seed)``), never
by wall-clock or unseeded randomness.

Design goal: every task a workload produces must be *feasible* (its admission
estimate, and for OOM tasks its doubled retry estimate, fits within usable
capacity). Infeasible tasks would head-of-line-block the current scheduler
forever; the harness surfaces that as a ``SimStall`` rather than hanging.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal

from . import task_stubs

# A submitted-estimate carries the profile safety multiplier at zero
# confidence (see LearnedProfile.estimate): estimate == hint * 1.5. Keep this
# in sync only as a sizing guide for generators; assertions read the real
# estimate off the executor, never this constant.
_COLD_SAFETY = 1.5

Outcome = Literal["success", "exception", "oom"]


@dataclass(frozen=True)
class TaskSpec:
    """A single synthetic task.

    ``est_memory_gb`` / ``est_vram_gb`` are the *hints* passed to ``submit``
    (``None`` means "no hint" -> cold-start / learned estimate). The ``actual_*``
    fields are the synthetic usage reported back through the real result path as
    a ``ResourceObservation``. ``outcome`` selects how the task resolves;
    ``oom_crashes`` is the number of SIGKILL-style crashes an ``"oom"`` task
    suffers before it resolves (each crash triggers the executor's real
    penalize+exclusive-retry path).
    """

    name: str
    fn: Callable[..., object]
    duration: float
    actual_memory_gb: float
    actual_vram_gb: float = 0.0
    cpu_percent: float = 100.0
    outcome: Outcome = "success"
    est_memory_gb: float | None = None
    est_vram_gb: float | None = None
    oom_crashes: int = 0


@dataclass(frozen=True)
class SimGpu:
    device_id: int
    vram_total_gb: float
    vram_baseline_used_gb: float = 0.0


@dataclass(frozen=True)
class SimCapacity:
    """Static machine capacity the ``SimMonitor`` reports for the whole run."""

    memory_total_gb: float
    memory_baseline_used_gb: float
    memory_headroom_gb: float = 2.0
    vram_headroom_gb: float = 1.0
    gpus: tuple[SimGpu, ...] = ()

    def usable_memory_gb(self) -> float:
        return self.memory_total_gb - self.memory_baseline_used_gb - self.memory_headroom_gb

    def usable_vram_gb(self, device_id: int) -> float:
        for gpu in self.gpus:
            if gpu.device_id == device_id:
                return gpu.vram_total_gb - gpu.vram_baseline_used_gb - self.vram_headroom_gb
        raise KeyError(f"unknown gpu device_id={device_id}")

    def gpu_ids(self) -> list[int]:
        return [gpu.device_id for gpu in self.gpus]


@dataclass(frozen=True)
class Workload:
    capacity: SimCapacity
    tasks: tuple[TaskSpec, ...]
    max_workers: int
    label: str = ""


# --------------------------------------------------------------------------- #
# Fixed, hand-built scenarios (targeted property tests).
# --------------------------------------------------------------------------- #


def cpu_fifo_workload() -> Workload:
    """Uniform CPU tasks with backpressure (max_workers=2) so dispatch order
    must equal submission order."""
    cap = SimCapacity(memory_total_gb=32.0, memory_baseline_used_gb=8.0)
    tasks = tuple(
        TaskSpec(
            name=f"fifo-{i}",
            fn=task_stubs.work,
            duration=float(1 + (i % 3)),
            actual_memory_gb=3.0,
            est_memory_gb=3.0,
        )
        for i in range(6)
    )
    return Workload(capacity=cap, tasks=tasks, max_workers=2, label="cpu_fifo")


def head_of_line_workload() -> Workload:
    """Three mediums that fill memory, then a LARGE head that only fits once the
    mediums release, then smalls blocked behind the large head."""
    cap = SimCapacity(memory_total_gb=32.0, memory_baseline_used_gb=8.0)
    mediums = [
        TaskSpec(f"medium-{i}", task_stubs.work, duration=2.0, actual_memory_gb=4.0, est_memory_gb=4.0)
        for i in range(3)
    ]
    large = TaskSpec("large-head", task_stubs.work, duration=1.0, actual_memory_gb=10.0, est_memory_gb=10.0)
    smalls = [
        TaskSpec(f"small-{i}", task_stubs.work, duration=1.0, actual_memory_gb=1.0, est_memory_gb=1.0)
        for i in range(2)
    ]
    tasks = tuple(mediums + [large] + smalls)
    return Workload(capacity=cap, tasks=tasks, max_workers=4, label="head_of_line")


def oom_retry_workload() -> Workload:
    """A long CPU task plus a flaky task that OOM-crashes once. The exclusive
    retry must wait for the long task to finish, then dispatch alone."""
    cap = SimCapacity(memory_total_gb=32.0, memory_baseline_used_gb=8.0)
    tasks = (
        TaskSpec("long", task_stubs.work, duration=5.0, actual_memory_gb=3.0, est_memory_gb=3.0),
        TaskSpec(
            "flaky",
            task_stubs.flaky,
            duration=1.0,
            actual_memory_gb=2.0,
            est_memory_gb=2.0,
            outcome="oom",
            oom_crashes=1,
        ),
    )
    return Workload(capacity=cap, tasks=tasks, max_workers=4, label="oom_retry")


def oom_exhausted_workload() -> Workload:
    """A flaky task that crashes more times than the retry budget allows, so it
    must resolve as a crash exception (still resolved -- never lost)."""
    cap = SimCapacity(memory_total_gb=32.0, memory_baseline_used_gb=8.0)
    tasks = (
        TaskSpec(
            "doomed",
            task_stubs.flaky,
            duration=1.0,
            actual_memory_gb=2.0,
            est_memory_gb=2.0,
            outcome="oom",
            oom_crashes=5,
        ),
    )
    return Workload(capacity=cap, tasks=tasks, max_workers=2, label="oom_exhausted")


def gpu_roundrobin_workload() -> Workload:
    """VRAM tasks across two GPUs to exercise per-GPU committed accounting."""
    cap = SimCapacity(
        memory_total_gb=32.0,
        memory_baseline_used_gb=8.0,
        gpus=(SimGpu(0, vram_total_gb=16.0, vram_baseline_used_gb=2.0), SimGpu(1, vram_total_gb=16.0, vram_baseline_used_gb=2.0)),
    )
    tasks = tuple(
        TaskSpec(
            name=f"gpu-{i}",
            fn=task_stubs.gpu_work,
            duration=float(1 + (i % 2)),
            actual_memory_gb=1.0,
            actual_vram_gb=3.0,
            est_memory_gb=1.0,
            est_vram_gb=3.0,
        )
        for i in range(6)
    )
    return Workload(capacity=cap, tasks=tasks, max_workers=4, label="gpu_roundrobin")


# --------------------------------------------------------------------------- #
# Deterministic random mixes.
# --------------------------------------------------------------------------- #


def random_workload(seed: int, n_tasks: int = 24) -> Workload:
    """A deterministic mix of task sizes, durations, wrong estimates, cold
    starts, and occasional OOM-retry tasks. Sizes are bounded so every task
    (and every OOM retry) stays feasible."""
    rng = random.Random(seed)
    cap = SimCapacity(
        memory_total_gb=48.0,
        memory_baseline_used_gb=8.0,
        gpus=(SimGpu(0, vram_total_gb=16.0, vram_baseline_used_gb=2.0), SimGpu(1, vram_total_gb=16.0, vram_baseline_used_gb=2.0)),
    )

    tasks: list[TaskSpec] = []
    for i in range(n_tasks):
        kind = rng.choices(
            population=("cpu", "gpu", "cold", "oom"),
            weights=(5, 3, 2, 2),
            k=1,
        )[0]
        duration = round(rng.uniform(0.5, 4.0), 3)

        if kind == "cpu":
            hint = round(rng.uniform(1.0, 6.0), 3)
            # Deliberately wrong estimate: actual is above or below the hint.
            actual = round(hint * rng.uniform(0.4, 1.8), 3)
            tasks.append(
                TaskSpec(f"cpu-{i}", task_stubs.work, duration, actual_memory_gb=actual, est_memory_gb=hint)
            )
        elif kind == "gpu":
            mem_hint = round(rng.uniform(0.5, 3.0), 3)
            vram_hint = round(rng.uniform(1.0, 4.0), 3)
            tasks.append(
                TaskSpec(
                    f"gpu-{i}",
                    task_stubs.gpu_work,
                    duration,
                    actual_memory_gb=round(mem_hint * rng.uniform(0.5, 1.5), 3),
                    actual_vram_gb=round(vram_hint * rng.uniform(0.5, 1.4), 3),
                    est_memory_gb=mem_hint,
                    est_vram_gb=vram_hint,
                )
            )
        elif kind == "cold":
            # No hint: cold-start default estimate (1.0 * safety margin).
            tasks.append(
                TaskSpec(
                    f"cold-{i}",
                    task_stubs.cold,
                    duration,
                    actual_memory_gb=round(rng.uniform(0.5, 3.0), 3),
                )
            )
        else:  # oom
            hint = round(rng.uniform(1.0, 3.0), 3)
            tasks.append(
                TaskSpec(
                    f"oom-{i}",
                    task_stubs.flaky,
                    duration,
                    actual_memory_gb=round(hint * rng.uniform(0.6, 1.2), 3),
                    est_memory_gb=hint,
                    outcome="oom",
                    oom_crashes=1,
                )
            )

    return Workload(capacity=cap, tasks=tuple(tasks), max_workers=6, label=f"random-{seed}")
