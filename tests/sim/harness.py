"""Discrete-event simulation engine for the executor scheduler.

The harness constructs a real :class:`AdaptiveExecutor`, injects a virtual
clock and a synthetic monitor, and stubs worker spawning with in-process
:class:`FakeProcess` / :class:`FakeQueue` (from ``tests/fakes.py``). It then
drives the executor's *real* scheduling methods -- ``_check_workers``,
``_maybe_dispatch`` (and thus ``scheduling.plan_dispatch`` /
committed-resource accounting), ``_handle_dead_worker`` and ``_process_result``
-- one event at a time against a monotonically advancing virtual clock. No
background threads run, no real time passes, and results are injected through
the real result-handling path.

The core artifact is a ``list[TraceEvent]`` recording every
submit/dispatch/complete/crash/retry/fail with its virtual timestamp and the
committed-resource state at that instant. Tests assert properties over it.

The scheduling policy under test is a parameter: pass ``dispatch`` /
``check_workers`` overrides to :class:`SchedulerSim` (defaulting to the
executor's current methods) so a future backfill scheduler can be simulated
without changing the harness.
"""

from __future__ import annotations

import heapq
import itertools
import signal
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Literal

from fakes import FakeProcess, FakeQueue

from adaptive_executor.adaptive_executor import AdaptiveExecutor, PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    GPUSnapshot,
    ResourceObservation,
    ResourceSnapshot,
    WorkResult,
)

from .workloads import SimCapacity, TaskSpec, Workload


class SimError(RuntimeError):
    """Base class for simulation failures."""


class SimStall(SimError):
    """Raised when work remains but nothing can make progress (e.g. an
    infeasible head-of-line task under the current scheduler)."""


class SimStepLimit(SimError):
    """Raised when the simulation does not reach quiescence within the step
    budget (guards against an unexpected non-terminating schedule)."""


EventKind = Literal["submit", "dispatch", "complete", "crash", "retry", "fail"]


@dataclass(frozen=True)
class TraceEvent:
    """One recorded scheduler event at a virtual timestamp.

    ``committed_*`` capture the executor's committed-resource accounting at the
    instant the event is recorded (after the mutation that produced it), so the
    admission invariants can be checked directly against the trace.
    """

    time: float
    kind: EventKind
    work_id: str
    task_name: str
    worker_id: int | None
    gpu_id: int | None
    est_memory_gb: float
    est_vram_gb: float
    committed_memory_gb: float
    committed_vram_per_gpu: tuple[tuple[int, float], ...]
    in_flight_count: int
    retry_count: int
    attempt: int
    detail: str = ""


@dataclass(order=True)
class _Completion:
    """A scheduled task-completion event (heap-ordered by time then seq)."""

    time: float
    seq: int
    work_id: str = field(compare=False)
    attempt: int = field(compare=False)


class VirtualClock:
    """Monotonic virtual time source bound to the executor via its clock seam."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> None:
        if t < self._t:
            raise SimError(f"cannot move virtual clock backwards: {t} < {self._t}")
        self._t = t


class SimMonitor:
    """Synthetic monitor exposing the read-only surface the executor uses.

    Reports a single static capacity snapshot for the whole run. The executor
    tracks in-flight load via committed estimates, not the snapshot, so a
    constant baseline is the correct model of usable capacity.
    """

    def __init__(self, snapshot: ResourceSnapshot):
        self._snapshot = snapshot

    @property
    def current(self) -> ResourceSnapshot:
        return self._snapshot

    def snapshot(self) -> ResourceSnapshot:
        return self._snapshot

    def start(self) -> None:  # pragma: no cover - no-op seam
        return None

    def stop(self) -> None:  # pragma: no cover - no-op seam
        return None


def _snapshot_from_capacity(cap: SimCapacity) -> ResourceSnapshot:
    gpus = {
        gpu.device_id: GPUSnapshot(
            device_id=gpu.device_id,
            vram_used_gb=gpu.vram_baseline_used_gb,
            vram_total_gb=gpu.vram_total_gb,
            utilization_percent=0.0,
        )
        for gpu in cap.gpus
    }
    return ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=cap.memory_baseline_used_gb,
        memory_total_gb=cap.memory_total_gb,
        gpus=gpus,
    )


class SchedulerSim:
    """Drives one workload to quiescence and records a trace.

    Parameters
    ----------
    workload:
        The tasks and capacity to simulate.
    max_resource_crash_retries:
        Forwarded to the executor; bounds OOM-retry storms.
    dispatch / check_workers:
        Optional overrides for the scheduling policy under test. Default to the
        executor's own ``_maybe_dispatch`` / ``_check_workers`` so the real
        current scheduler is exercised; a future backfill scheduler can be
        supplied here without touching the harness.
    """

    def __init__(
        self,
        workload: Workload,
        *,
        max_resource_crash_retries: int = 1,
        dispatch: Callable[[], None] | None = None,
        check_workers: Callable[[], None] | None = None,
    ):
        self.workload = workload
        self.capacity = workload.capacity
        self.clock = VirtualClock()
        self.monitor = SimMonitor(_snapshot_from_capacity(workload.capacity))

        self.executor = AdaptiveExecutor(
            max_workers=workload.max_workers,
            gpu_ids=workload.capacity.gpu_ids(),
            memory_headroom_gb=workload.capacity.memory_headroom_gb,
            vram_headroom_gb=workload.capacity.vram_headroom_gb,
            # Large timeout: the sim never drives the timeout path.
            task_timeout_seconds=1e18,
            max_resource_crash_retries=max_resource_crash_retries,
            # Disable recycling so worker reuse is simple and deterministic.
            worker_recycle_after_tasks=None,
            monitor=self.monitor,
            clock=self.clock.now,
        )
        # Mark started so submit() does not launch real threads / monitor.
        self.executor._started = True
        self._install_fake_spawn()

        self._dispatch = dispatch or self.executor._maybe_dispatch
        self._check_workers = check_workers or self.executor._check_workers

        self.trace: list[TraceEvent] = []
        self.submit_order: list[str] = []
        self._specs_by_id: dict[str, TaskSpec] = {}
        self._futures_by_id: dict[str, Future] = {}
        self._name_by_id: dict[str, str] = {}

        self._events: list[_Completion] = []
        self._seq = itertools.count()
        self._attempts: dict[str, int] = {}
        self._scheduled: set[str] = set()
        self._retry_seen: dict[str, int] = {}

    # -- setup ------------------------------------------------------------- #

    def _install_fake_spawn(self) -> None:
        ex = self.executor

        def fake_spawn(pinned_gpu_id: int | None) -> WorkerSlot:
            worker_id = ex._next_worker_id
            ex._next_worker_id += 1
            slot = WorkerSlot(
                worker_id=worker_id,
                process=FakeProcess(alive=True),
                work_queue=FakeQueue(),
                pinned_gpu_id=pinned_gpu_id,
            )
            ex.workers[worker_id] = slot
            return slot

        ex._spawn_worker = fake_spawn  # type: ignore[method-assign]

    # -- submission -------------------------------------------------------- #

    def submit_all(self) -> None:
        for spec in self.workload.tasks:
            future = self.executor.submit(
                spec.fn,
                memory_gb=spec.est_memory_gb,
                vram_gb=spec.est_vram_gb,
            )
            pending = self.executor.pending[-1]
            work_id = pending.item.id
            self._specs_by_id[work_id] = spec
            self._futures_by_id[work_id] = future
            self._name_by_id[work_id] = spec.name
            self.submit_order.append(work_id)
            self._record("submit", pending)

    # -- main loop --------------------------------------------------------- #

    def run(self, max_steps: int = 200_000) -> list[TraceEvent]:
        """Submit all tasks and drive the scheduler to quiescence."""
        self.submit_all()

        for _ in range(max_steps):
            self._check_workers()
            self._detect_retries()
            self._dispatch()
            self._detect_dispatches()

            if self._quiescent():
                return self.trace

            if not self._events:
                head = self.executor.pending[0] if self.executor.pending else None
                raise SimStall(
                    "no progress possible with work remaining "
                    f"(pending={len(self.executor.pending)}, in_flight={len(self.executor.in_flight)}, "
                    f"head={self._name_by_id.get(head.item.id) if head else None})"
                )

            self._advance_and_fire()

        raise SimStepLimit(f"did not reach quiescence within {max_steps} steps")

    def _quiescent(self) -> bool:
        return not self.executor.pending and not self.executor.in_flight and not self._events

    def _advance_and_fire(self) -> None:
        comp = heapq.heappop(self._events)
        self.clock.advance_to(comp.time)
        self._fire_completion(comp)

    # -- event handling ---------------------------------------------------- #

    def _detect_dispatches(self) -> None:
        """Record a dispatch and schedule a completion for each task that has
        newly entered ``in_flight`` since the last scan."""
        for work_id, pending in list(self.executor.in_flight.items()):
            if work_id in self._scheduled:
                continue
            self._scheduled.add(work_id)
            attempt = self._attempts.get(work_id, 0) + 1
            self._attempts[work_id] = attempt
            self._record("dispatch", pending)
            spec = self._specs_by_id[work_id]
            finish_time = self.clock.now() + spec.duration
            heapq.heappush(
                self._events, _Completion(finish_time, next(self._seq), work_id, attempt)
            )

    def _detect_retries(self) -> None:
        """Emit a retry event when a crashed task has been re-queued with an
        incremented retry_count (its estimate is already penalized/doubled)."""
        for pending in list(self.executor.pending):
            work_id = pending.item.id
            if pending.retry_count > self._retry_seen.get(work_id, 0):
                self._retry_seen[work_id] = pending.retry_count
                self._record("retry", pending, detail=f"exclusive={pending.exclusive}")

    def _fire_completion(self, comp: _Completion) -> None:
        work_id = comp.work_id
        # Ignore a completion for a superseded attempt (e.g. after a re-dispatch).
        if self._attempts.get(work_id) != comp.attempt:
            return
        self._scheduled.discard(work_id)
        pending = self.executor.in_flight.get(work_id)
        if pending is None:
            return

        spec = self._specs_by_id[work_id]
        if spec.outcome == "oom" and comp.attempt <= spec.oom_crashes:
            self._crash(pending)
        else:
            self._finish(pending, spec)

    def _crash(self, pending: PendingWork) -> None:
        """Simulate a SIGKILL-style OOM crash of the worker running ``pending``.

        Marking the fake process dead lets the executor's real
        ``_check_workers`` -> ``_handle_dead_worker`` path decide retry vs. final
        failure on the next loop iteration.
        """
        self._record("crash", pending, detail="sigkill")
        worker = self.executor.workers.get(pending.worker_id) if pending.worker_id is not None else None
        if worker is not None:
            worker.process.set_dead(exitcode=-signal.SIGKILL)

    def _finish(self, pending: PendingWork, spec: TaskSpec) -> None:
        observation = ResourceObservation(
            memory_delta_gb=spec.actual_memory_gb,
            vram_delta_gb=spec.actual_vram_gb,
            cpu_percent=spec.cpu_percent,
            duration_seconds=spec.duration,
        )
        if spec.outcome == "exception":
            result = WorkResult(
                id=pending.item.id,
                worker_id=pending.worker_id if pending.worker_id is not None else -1,
                success=False,
                result=None,
                exception=RuntimeError(f"{spec.name} raised"),
                observation=observation,
            )
            kind: EventKind = "fail"
        else:
            result = WorkResult(
                id=pending.item.id,
                worker_id=pending.worker_id if pending.worker_id is not None else -1,
                success=True,
                result=spec.name,
                exception=None,
                observation=observation,
            )
            kind = "complete"

        # Resolve through the real result path, then record post-release state.
        self.executor._process_result(result)
        self._record(kind, pending)

    # -- recording --------------------------------------------------------- #

    def _record(self, kind: EventKind, pending: PendingWork, detail: str = "") -> None:
        committed = self.executor._committed_resources()
        per_gpu = self.executor._committed_vram_per_gpu()
        work_id = pending.item.id
        self.trace.append(
            TraceEvent(
                time=self.clock.now(),
                kind=kind,
                work_id=work_id,
                task_name=self._name_by_id.get(work_id, "?"),
                worker_id=pending.worker_id,
                gpu_id=pending.assigned_gpu_id,
                est_memory_gb=pending.estimate.memory_gb,
                est_vram_gb=pending.estimate.vram_gb,
                committed_memory_gb=committed.memory_gb,
                committed_vram_per_gpu=tuple(sorted(per_gpu.items())),
                in_flight_count=len(self.executor.in_flight),
                retry_count=pending.retry_count,
                attempt=self._attempts.get(work_id, 0),
                detail=detail,
            )
        )

    # -- queries used by tests -------------------------------------------- #

    def futures(self) -> dict[str, Future]:
        return dict(self._futures_by_id)

    def committed_now(self) -> tuple[float, dict[int, float]]:
        committed = self.executor._committed_resources()
        return committed.memory_gb, dict(self.executor._committed_vram_per_gpu())

    def dispatch_events(self, attempt: int | None = None) -> list[TraceEvent]:
        return [
            e
            for e in self.trace
            if e.kind == "dispatch" and (attempt is None or e.attempt == attempt)
        ]

    def events_for(self, task_name: str) -> list[TraceEvent]:
        return [e for e in self.trace if e.task_name == task_name]


def run_to_quiescence(
    workload: Workload, *, max_resource_crash_retries: int = 1
) -> SchedulerSim:
    """Convenience: build a sim, run it, and return it (trace on ``.trace``)."""
    sim = SchedulerSim(workload, max_resource_crash_retries=max_resource_crash_retries)
    sim.run()
    return sim
