import logging
import math
import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, replace
from queue import Empty
from typing import Callable, Literal

import psutil

from .accounting import (
    ResourceUsage,
    committed_gb,
    committed_vram_per_gpu,
    total_committed_gb,
)
from .dtypes import ResourceEstimate, ResourceSnapshot, WorkItem, WorkResult
from .errors import InfeasibleTaskError
from .monitor import ResourceMonitor
from .profiles import ProfileStore
from .resolve import validate_submittable
from .scheduling import (
    Capacity,
    PendingEntry,
    RunningEntry,
    plan_dispatch,
)
from .worker import worker_process_entry

logger = logging.getLogger("adaptive_executor")


@dataclass
class PendingWork:
    item: WorkItem
    future: Future
    estimate: ResourceEstimate
    submitted_at: float
    assigned_gpu_id: int | None = None
    started_at: float | None = None
    worker_id: int | None = None
    retry_count: int = 0
    exclusive: bool = False
    result_ignored: bool = False
    # True once the running handshake (``set_running_or_notify_cancel``) has put
    # this future into the RUNNING state. The handshake must run exactly once per
    # future; a crash-retry re-queues an already-RUNNING future, so re-dispatch
    # must skip it (and ``cancel()`` already returns False on a running task).
    running_notified: bool = False
    # Observed-usage accounting (parent-side polling). Baselines are captured at
    # dispatch; ``observed_*`` are the current attributable usage above baseline,
    # refreshed at most once per ``_observation_refresh_seconds``. The full
    # estimate stays committed until usage is realized (observed == 0 fallback).
    worker_pid: int | None = None
    rss_baseline_bytes: int | None = None
    vram_baseline_gb: float | None = None
    observed_memory_gb: float = 0.0
    observed_vram_gb: float = 0.0
    observed_refreshed_at: float | None = None
    # Optional caller-supplied bucket for input-aware profiling. Travels with the
    # task (parent-side only; workers never see it) so result recording writes to
    # the same keyed profile the estimate was drawn from. Survives crash-retry
    # re-queuing because the PendingWork object is reused.
    profile_key: str | None = None


@dataclass
class WorkerSlot:
    worker_id: int
    process: mp.Process
    work_queue: mp.Queue
    pinned_gpu_id: int | None
    current_work_id: str | None = None
    intentionally_stopped: bool = False
    pending_retry_work_id: str | None = None
    tasks_completed: int = 0


class AdaptiveExecutor:
    """
    Resource-aware parallel executor.

    Learns resource usage patterns and automatically adjusts parallelism.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        gpu_ids: list[int] | None = None,
        profile_path: str | None = None,
        memory_headroom_gb: float = 2.0,
        vram_headroom_gb: float = 1.0,
        task_timeout_seconds: float = 300.0,
        on_timeout: Literal["fail_future", "kill_worker"] = "fail_future",
        max_resource_crash_retries: int = 1,
        worker_recycle_after_tasks: int | None = 50,
        monitor: "ResourceMonitor | None" = None,
        clock: Callable[[], float] = time.time,
    ):
        self.max_workers = max_workers or os.cpu_count() or 4
        self.gpu_ids = gpu_ids
        self.memory_headroom_gb = memory_headroom_gb
        self.vram_headroom_gb = vram_headroom_gb
        self.task_timeout_seconds = task_timeout_seconds
        self.on_timeout = on_timeout
        self.max_resource_crash_retries = max_resource_crash_retries
        self.worker_recycle_after_tasks = worker_recycle_after_tasks
        self._next_gpu_index = 0
        self._next_worker_id = 0
        # Cache per-task observed-usage samples this long so psutil/NVML are not
        # polled on every 10ms dispatch tick.
        self._observation_refresh_seconds = 0.1

        # Time source and monitor are injectable seams so a deterministic
        # simulation can substitute a virtual clock and a synthetic monitor.
        # Defaults preserve exactly the production wall-clock/NVML behavior.
        self._clock = clock
        self.monitor = monitor if monitor is not None else ResourceMonitor()
        self.profiles = ProfileStore(persist_path=profile_path)

        self.result_queue: mp.Queue = mp.Queue()
        self.workers: dict[int, WorkerSlot] = {}
        # Workers removed from the pool (retired/recycled/evicted) awaiting reap.
        self._retiring: list[WorkerSlot] = []

        self.pending: deque[PendingWork] = deque()
        self.in_flight: dict[str, PendingWork] = {}
        self.completed_or_abandoned: set[str] = set()
        self.lock = threading.Lock()

        self._started = False
        self._shutdown = False
        self._accepting = True
        self._threads_should_stop = False

    def start(self):
        """Initialize monitoring and background threads."""
        if self._shutdown:
            raise RuntimeError("Cannot restart a shutdown AdaptiveExecutor")
        if self._started:
            return

        self._started = True
        self.monitor.start()

        if self.gpu_ids is None:
            snapshot = self.monitor.snapshot()
            self.gpu_ids = list(snapshot.gpus.keys())

        self._result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._timeout_thread = threading.Thread(target=self._check_timeouts, daemon=True)

        self._result_thread.start()
        self._dispatch_thread.start()
        self._timeout_thread.start()

    def shutdown(self, wait: bool = True):
        """Stop accepting work, fail queued tasks, and tear down background resources."""
        if self._shutdown:
            return

        if not self._started:
            self._shutdown = True
            self._accepting = False
            return

        self._accepting = False
        self._fail_pending_futures(RuntimeError("Executor shut down before task was dispatched"))

        if wait:
            self._wait_for_in_flight()

        self._shutdown = True
        self._threads_should_stop = True
        self._stop_all_workers(wait=wait)
        self._join_background_threads()
        self.profiles.flush()
        self.monitor.stop()

    def submit(
        self,
        fn: Callable,
        *args,
        memory_gb: float | None = None,
        vram_gb: float | None = None,
        profile_key: str | None = None,
        **kwargs,
    ) -> Future:
        """
        Submit work for execution.

        Args:
            fn: Function to execute (must be importable)
            *args: Positional arguments
            memory_gb: Optional hint for expected RAM usage
            vram_gb: Optional hint for expected VRAM usage
            profile_key: Optional opaque string bucketing inputs by expected
                resource usage (e.g. "small"/"large", a resolution, a file-size
                band). Observations are learned per bucket so an input-sensitive
                function does not merge a small and a huge input into one
                distribution. When the bucket has no history yet, estimation
                falls back to the function's aggregate profile.
            **kwargs: Keyword arguments

        Returns:
            Future that will contain the result.

            Cancellation follows standard ``concurrent.futures`` semantics: while
            the task is still queued, ``future.cancel()`` succeeds, frees its
            queue slot, and the task is never executed. Once the executor
            dispatches it to a worker the future transitions to RUNNING and
            ``cancel()`` returns ``False`` — a running task cannot be cancelled.
            The dispatch/running transition is decided atomically by
            ``set_running_or_notify_cancel()`` at ship time, so exactly one of
            "cancelled" or "dispatched" wins a race between the two.
        """
        if self._shutdown or not self._accepting:
            raise RuntimeError("Cannot submit to a shutdown AdaptiveExecutor")

        # Fail fast on functions that cannot be re-resolved in a worker
        # subprocess (lambdas, closures, bound methods, unimportable callables).
        validate_submittable(fn)

        if not self._started:
            self.start()

        item = WorkItem(
            id=str(uuid.uuid4()),
            fn_module=fn.__module__,
            fn_name=fn.__qualname__,
            args=args,
            kwargs=kwargs,
            gpu_id=None,
        )

        profile = self.profiles.get(item.fn_module, item.fn_name, profile_key)
        estimate = profile.estimate(memory_hint=memory_gb, vram_hint=vram_gb)

        # Reject a task that can never fit on this machine before it ever enters
        # the queue, so it fails at the call site instead of silently blocking
        # the FIFO head forever. The polled snapshot may not be populated yet
        # right after auto-start, so take a direct reading in that case.
        snapshot = self.monitor.current or self.monitor.snapshot()
        infeasible = self._infeasible_estimate(estimate, snapshot, retry_count=0)
        if infeasible is not None:
            raise infeasible

        future = Future()
        pending = PendingWork(
            item=item,
            future=future,
            estimate=estimate,
            submitted_at=self._clock(),
            profile_key=profile_key,
        )

        with self.lock:
            self.pending.append(pending)

        return future

    def _dispatch_loop(self):
        while not self._threads_should_stop:
            self._check_workers()
            self._maybe_dispatch()
            time.sleep(0.01)

    def _maybe_dispatch(self):
        with self.lock:
            if not self.pending:
                return

            # Fail any queued task whose estimate can never fit (e.g. crash-
            # retry penalization doubled it past capacity). Backfill would
            # route around an infeasible task rather than stall on it, but it
            # would then sit queued forever; fail its future so callers see
            # the error. The dispatch thread itself must keep running.
            snapshot = self.monitor.current
            feasible: deque[PendingWork] = deque()
            for pending in self.pending:
                infeasible = self._infeasible_estimate(
                    pending.estimate, snapshot, retry_count=pending.retry_count
                )
                if infeasible is None:
                    feasible.append(pending)
                    continue
                # A queued task the caller already cancelled is done: drop it as
                # abandoned rather than stuffing an exception into a cancelled
                # future (which would raise InvalidStateError).
                self._settle_future_exception(pending, infeasible)
                self.completed_or_abandoned.add(pending.item.id)
            self.pending = feasible
            if not self.pending:
                return

            plan = self._build_dispatch_plan()
            self._next_gpu_index = plan.next_gpu_index
            if not plan.decisions:
                return

            by_id = {p.item.id: p for p in self.pending}
            dispatched: set[str] = set()
            cancelled: set[str] = set()
            for decision in plan.decisions:
                pending = by_id.get(decision.pending_id)
                if pending is None:
                    continue
                worker = self._get_or_spawn_idle_worker(decision.gpu_id)
                if worker is None:
                    # No worker available right now; leave the task queued. This
                    # never delays the head below FIFO — a worker shortage would
                    # stall FIFO identically.
                    continue

                # Running handshake, immediately before shipping and only once a
                # worker is secured so the future is still PENDING here. Run it
                # exactly once per future: a crash-retry re-queues an already-
                # RUNNING future, and calling the handshake twice raises. If the
                # caller already cancelled this queued task,
                # set_running_or_notify_cancel() returns False: drop it as
                # abandoned, never dispatch, and leave worker state untouched
                # (current_work_id stays None, so the worker is reused next
                # decision). After it returns True the future is RUNNING and
                # cancel() can no longer succeed, so the in-flight accounting can
                # never be cancelled out from under us.
                if not pending.running_notified:
                    if not pending.future.set_running_or_notify_cancel():
                        logger.debug(
                            "dropping cancelled queued task work_id=%s",
                            pending.item.id,
                        )
                        self.completed_or_abandoned.add(pending.item.id)
                        cancelled.add(pending.item.id)
                        continue
                    pending.running_notified = True

                pending.assigned_gpu_id = decision.gpu_id
                pending.worker_id = worker.worker_id
                pending.started_at = self._clock()
                self._capture_baseline(pending, worker, decision.gpu_id)
                self.in_flight[pending.item.id] = pending
                worker.current_work_id = pending.item.id
                worker.work_queue.put(replace(pending.item, gpu_id=decision.gpu_id))
                dispatched.add(pending.item.id)

            removed = dispatched | cancelled
            if removed:
                self.pending = deque(
                    p for p in self.pending if p.item.id not in removed
                )

    def _infeasible_estimate(
        self,
        estimate: ResourceEstimate,
        snapshot: ResourceSnapshot | None,
        retry_count: int,
    ) -> InfeasibleTaskError | None:
        """Return an error if ``estimate`` can never fit, else ``None``.

        Infeasibility means the estimate exceeds *total* capacity minus headroom
        — a permanent condition, distinct from "doesn't fit right now" (normal
        queuing). It is only declared when capacity is actually known: memory
        needs a snapshot with a positive total; VRAM additionally needs GPU info.
        When capacity is unknown, returns ``None`` so admission behaves as before.
        """
        if snapshot is None:
            return None

        if snapshot.memory_total_gb > 0:
            memory_capacity = snapshot.memory_total_gb - self.memory_headroom_gb
            if estimate.memory_gb > memory_capacity:
                return InfeasibleTaskError(
                    kind="memory",
                    estimate_gb=estimate.memory_gb,
                    capacity_gb=memory_capacity,
                    retry_count=retry_count,
                )

        if estimate.vram_gb > 0 and snapshot.gpus:
            largest_vram_total = max(g.vram_total_gb for g in snapshot.gpus.values())
            vram_capacity = largest_vram_total - self.vram_headroom_gb
            if estimate.vram_gb > vram_capacity:
                return InfeasibleTaskError(
                    kind="vram",
                    estimate_gb=estimate.vram_gb,
                    capacity_gb=vram_capacity,
                    retry_count=retry_count,
                )

        return None

    def _build_dispatch_plan(self):
        """Gather live state and delegate the scheduling decision to the pure
        reservation-based backfill scheduler in :mod:`adaptive_executor.scheduling`.
        """
        snapshot = self.monitor.current
        now = self._clock()
        # Refresh observed usage first so committed totals credit realized
        # allocation (already reflected in ``snapshot``) against the estimates.
        self._refresh_observations(now)
        committed = self._committed_resources()
        committed_vram = self._committed_vram_per_gpu()
        gpu_ids = self.gpu_ids or []

        if snapshot is None:
            # No snapshot yet: memory/VRAM are not gating (treated as infinite),
            # matching the prior startup admission behavior.
            memory_free = math.inf
            gpu_free: dict[int, float] = {g: math.inf for g in gpu_ids}
            snapshot_present = False
        else:
            memory_free = (
                snapshot.memory_total_gb
                - snapshot.memory_used_gb
                - self.memory_headroom_gb
                - committed.memory_gb
            )
            gpu_free = {}
            for g in gpu_ids:
                gpu = snapshot.gpus.get(g)
                if gpu is None:
                    continue
                gpu_free[g] = (
                    gpu.vram_total_gb
                    - gpu.vram_used_gb
                    - self.vram_headroom_gb
                    - committed_vram.get(g, 0.0)
                )
            snapshot_present = True

        # A running task releases only its committed remainder to the live pool
        # when it finishes; see ``_running_release_gb`` for the conservative
        # rationale. Using the remainder keeps the head's reservation from ever
        # assuming more frees than the accounting guarantees.
        running = [
            RunningEntry(
                id=p.item.id,
                memory_gb=committed_gb(p.estimate.memory_gb, p.observed_memory_gb),
                vram_gb=committed_gb(p.estimate.vram_gb, p.observed_vram_gb),
                gpu_id=p.assigned_gpu_id,
                remaining_seconds=self._running_remaining_seconds(p, now),
                exclusive=p.exclusive,
            )
            for p in self.in_flight.values()
        ]

        pending_entries = [
            PendingEntry(
                id=p.item.id,
                memory_gb=p.estimate.memory_gb,
                vram_gb=p.estimate.vram_gb,
                cpu_cores=p.estimate.cpu_cores,
                duration_p90_seconds=p.estimate.duration_p90_seconds,
                exclusive=p.exclusive,
            )
            for p in self.pending
        ]

        capacity = Capacity(
            memory_free_gb=memory_free,
            gpu_free_vram_gb=gpu_free,
            gpu_round_robin=tuple(gpu_ids),
            next_gpu_index=self._next_gpu_index,
            max_workers=self.max_workers,
            running_count=len(self.in_flight),
            snapshot_present=snapshot_present,
        )

        return plan_dispatch(pending_entries, running, capacity)

    def _running_remaining_seconds(self, pending: PendingWork, now: float) -> float | None:
        """Expected seconds until ``pending`` releases its resources.

        Returns None (unknown, assumed never to release) when the function has no
        learned duration, when the task has not started, or when it has already
        overrun its estimate — the last case makes the head's reservation slip,
        degrading toward FIFO rather than below it.
        """
        expected = pending.estimate.duration_p90_seconds
        if expected is None or pending.started_at is None:
            return None
        elapsed = now - pending.started_at
        if elapsed >= expected:
            return None
        return expected - elapsed

    def _committed_resources(self) -> ResourceEstimate:
        """In-flight resources still awaiting realization.

        Memory/VRAM commit only the unrealized remainder
        ``max(estimate - observed, 0)`` because realized usage already shows up in
        the monitor snapshot's ``used`` figure (avoiding admission double-counting).
        CPU cores carry no observed credit — they gate worker-slot pressure, not
        snapshot memory — so they remain the sum of estimates.
        """
        return ResourceEstimate(
            memory_gb=total_committed_gb(
                (p.estimate.memory_gb, p.observed_memory_gb)
                for p in self.in_flight.values()
            ),
            vram_gb=total_committed_gb(
                (p.estimate.vram_gb, p.observed_vram_gb)
                for p in self.in_flight.values()
            ),
            cpu_cores=sum(p.estimate.cpu_cores for p in self.in_flight.values()),
        )

    def _committed_vram_per_gpu(self) -> dict[int, float]:
        return committed_vram_per_gpu(
            ResourceUsage(
                gpu_id=p.assigned_gpu_id,
                estimate_gb=p.estimate.vram_gb,
                observed_gb=p.observed_vram_gb,
            )
            for p in self.in_flight.values()
        )

    # -- observed-usage polling (parent-side, no worker protocol changes) -----

    def _capture_baseline(
        self, pending: PendingWork, worker: WorkerSlot, gpu_id: int | None
    ) -> None:
        """Record the assigned worker's RSS (and pinned-GPU VRAM) baseline at
        dispatch. Observed usage is measured as growth above this baseline. If the
        worker pid is unknown or unreadable, baselines stay ``None`` and observed
        usage stays 0 (the full estimate remains committed — conservative).
        """
        pid = getattr(worker.process, "pid", None)
        pending.worker_pid = pid
        pending.observed_memory_gb = 0.0
        pending.observed_vram_gb = 0.0
        pending.observed_refreshed_at = self._clock()
        if pid is None:
            pending.rss_baseline_bytes = None
            pending.vram_baseline_gb = None
            return
        pending.rss_baseline_bytes = self._sample_rss_bytes(pid)
        pending.vram_baseline_gb = (
            self._sample_process_vram_gb(pid, gpu_id) if gpu_id is not None else None
        )

    def _refresh_observations(self, now: float) -> None:
        """Refresh each in-flight task's observed usage, throttled to at most once
        per ``_observation_refresh_seconds`` so psutil/NVML are not hammered on the
        10ms dispatch tick. Must be called while holding ``self.lock``.
        """
        for pending in self.in_flight.values():
            last = pending.observed_refreshed_at
            if last is not None and (now - last) < self._observation_refresh_seconds:
                continue
            self._update_observed(pending, now)

    def _update_observed(self, pending: PendingWork, now: float) -> None:
        pending.observed_refreshed_at = now
        pid = pending.worker_pid
        if pid is None:
            return

        if pending.rss_baseline_bytes is not None:
            current = self._sample_rss_bytes(pid)
            pending.observed_memory_gb = (
                max(0.0, (current - pending.rss_baseline_bytes) / 1e9)
                if current is not None
                else 0.0
            )

        if pending.assigned_gpu_id is not None and pending.vram_baseline_gb is not None:
            current_vram = self._sample_process_vram_gb(pid, pending.assigned_gpu_id)
            pending.observed_vram_gb = (
                max(0.0, current_vram - pending.vram_baseline_gb)
                if current_vram is not None
                else 0.0
            )

        logger.debug(
            "observed usage work_id=%s est_mem_gb=%.3f obs_mem_gb=%.3f "
            "est_vram_gb=%.3f obs_vram_gb=%.3f",
            pending.item.id,
            pending.estimate.memory_gb,
            pending.observed_memory_gb,
            pending.estimate.vram_gb,
            pending.observed_vram_gb,
        )

    def _sample_rss_bytes(self, pid: int) -> int | None:
        """Current RSS (bytes) of ``pid``, or None if the process is unreadable.

        Overridable seam. Narrow, typed excepts only; polling failures never
        propagate into the dispatch thread.
        """
        try:
            return psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
            logger.debug("rss sample failed pid=%d error=%r", pid, exc)
            return None

    def _worker_process_pids(self, pid: int) -> set[int]:
        """``pid`` plus its descendant pids, for per-process VRAM attribution."""
        pids = {pid}
        try:
            for child in psutil.Process(pid).children(recursive=True):
                pids.add(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
            logger.debug("child pid enumeration failed pid=%d error=%r", pid, exc)
        return pids

    def _sample_process_vram_gb(self, pid: int, gpu_id: int | None) -> float | None:
        """VRAM (GB) attributable to ``pid``'s process tree on ``gpu_id``.

        Returns None when the monitor cannot supply per-process VRAM (no NVML,
        unavailable API), so callers keep the full estimate committed. Overridable
        seam.
        """
        if gpu_id is None:
            return None
        per_process = getattr(self.monitor, "per_process_vram_gb", None)
        if per_process is None:
            return None
        pids = self._worker_process_pids(pid)
        return per_process(gpu_id, pids)

    def _get_or_spawn_idle_worker(self, pinned_gpu_id: int | None) -> WorkerSlot | None:
        for worker in self.workers.values():
            if (
                worker.pinned_gpu_id == pinned_gpu_id
                and worker.current_work_id is None
                and not worker.intentionally_stopped
                and worker.process.is_alive()
            ):
                return worker

        if len(self.workers) >= self.max_workers:
            # At the cap with no matching idle worker. If some idle worker is
            # pinned differently, evict it so a correctly-pinned replacement can
            # be spawned; this prevents a permanent dispatch stall. If every
            # worker is genuinely busy, return None (correct backpressure).
            victim = self._find_idle_evictable_worker()
            if victim is None:
                return None
            self._retire_worker(victim, reason="pin_mismatch")

        return self._spawn_worker(pinned_gpu_id)

    def _find_idle_evictable_worker(self) -> WorkerSlot | None:
        """Return an alive, idle, not-already-retiring worker, or None."""
        for worker in self.workers.values():
            if (
                worker.current_work_id is None
                and not worker.intentionally_stopped
                and worker.process.is_alive()
            ):
                return worker
        return None

    def _retire_worker(self, worker: WorkerSlot, reason: str) -> None:
        """Remove ``worker`` from the pool and schedule it for reaping.

        Must be called while holding ``self.lock``. Sends the stop sentinel,
        marks the worker as intentionally stopped, drops it from ``self.workers``
        so the cap frees immediately, and parks it in ``self._retiring`` to be
        joined by ``_check_workers``.
        """
        worker.intentionally_stopped = True
        worker.pending_retry_work_id = None
        try:
            worker.work_queue.put(None)
        except (ValueError, OSError) as exc:
            logger.warning(
                "failed to send stop sentinel worker_id=%d error=%r",
                worker.worker_id,
                exc,
            )
        self.workers.pop(worker.worker_id, None)
        self._retiring.append(worker)
        logger.debug(
            "retired worker worker_id=%d pinned_gpu_id=%s reason=%s tasks_completed=%d",
            worker.worker_id,
            worker.pinned_gpu_id,
            reason,
            worker.tasks_completed,
        )

    def _should_recycle(self, worker: WorkerSlot) -> bool:
        if self.worker_recycle_after_tasks is None:
            return False
        if worker.intentionally_stopped:
            return False
        return worker.tasks_completed >= self.worker_recycle_after_tasks

    def _spawn_worker(self, pinned_gpu_id: int | None) -> WorkerSlot:
        worker_id = self._next_worker_id
        self._next_worker_id += 1

        work_queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=worker_process_entry,
            args=(work_queue, self.result_queue, worker_id, pinned_gpu_id),
        )
        process.start()

        worker = WorkerSlot(
            worker_id=worker_id,
            process=process,
            work_queue=work_queue,
            pinned_gpu_id=pinned_gpu_id,
        )
        self.workers[worker_id] = worker
        return worker

    def _collect_results(self):
        while not self._threads_should_stop or self.in_flight:
            try:
                result: WorkResult = self.result_queue.get(timeout=0.1)
            except Empty:
                continue

            self._process_result(result)

    def _process_result(self, result: WorkResult) -> None:
        """Handle a single worker result: clear/recycle the worker, pop the
        in-flight entry, record the observation, and resolve the future.

        Extracted from the ``_collect_results`` loop body so the same logic can
        be driven one result at a time (e.g. by a deterministic simulation).
        """
        with self.lock:
            worker = self.workers.get(result.worker_id)
            if worker is not None and worker.current_work_id == result.id:
                worker.current_work_id = None
                worker.tasks_completed += 1
                # Recycle a worker that has processed enough tasks so its RSS
                # baseline (which CPython rarely returns to the OS) cannot
                # ratchet up and cause future memory observations to shrink.
                if self._should_recycle(worker):
                    self._retire_worker(worker, reason="recycle")

            pending = self.in_flight.pop(result.id, None)
            if pending is None:
                return

            if pending.result_ignored:
                self.completed_or_abandoned.add(result.id)
                return

        self.profiles.record(
            pending.item.fn_module,
            pending.item.fn_name,
            result.observation,
            profile_key=pending.profile_key,
        )

        # Guard both settle paths: a late result for a future already resolved
        # by timeout/crash handling (or cancelled) must never raise
        # InvalidStateError and kill this collector thread.
        if result.success:
            self._settle_future_result(pending, result.result)
        else:
            self._settle_future_exception(pending, result.exception)

    def _check_timeouts(self):
        while not self._threads_should_stop:
            time.sleep(0.1)
            now = self._clock()
            timed_out: list[PendingWork] = []

            with self.lock:
                for pending in self.in_flight.values():
                    if pending.started_at is None:
                        continue
                    if now - pending.started_at > self.task_timeout_seconds:
                        timed_out.append(pending)

            for pending in timed_out:
                self._handle_timeout(pending)

    def _handle_timeout(self, pending: PendingWork):
        with self.lock:
            active = self.in_flight.get(pending.item.id)
            if active is None or active.future.done():
                return

            if self.on_timeout == "kill_worker" and active.worker_id is not None:
                worker = self.workers.get(active.worker_id)
                if worker is not None:
                    worker.intentionally_stopped = True
                    worker.pending_retry_work_id = None
                    worker.process.terminate()
                    worker.current_work_id = None
                self.in_flight.pop(active.item.id, None)
            else:
                active.result_ignored = True
                self.in_flight.pop(active.item.id, None)
                worker = self.workers.get(active.worker_id) if active.worker_id is not None else None
                if worker is not None and worker.current_work_id == active.item.id:
                    worker.current_work_id = None

        self._settle_future_exception(
            active,
            TimeoutError(f"Task {active.item.id} timed out after {self.task_timeout_seconds}s"),
        )
        self.completed_or_abandoned.add(active.item.id)

    def _check_workers(self):
        dead_workers: list[WorkerSlot] = []
        reaped: list[WorkerSlot] = []
        with self.lock:
            for worker in list(self.workers.values()):
                if worker.process.is_alive():
                    continue
                dead_workers.append(worker)
                self.workers.pop(worker.worker_id, None)

            # Reap retired/recycled/evicted workers that have exited so no
            # zombies accumulate; keep the ones still shutting down.
            still_retiring: list[WorkerSlot] = []
            for worker in self._retiring:
                if worker.process.is_alive():
                    still_retiring.append(worker)
                else:
                    reaped.append(worker)
            self._retiring = still_retiring

        for worker in reaped:
            worker.process.join(timeout=0.1)
            logger.debug("reaped retired worker worker_id=%d", worker.worker_id)

        for worker in dead_workers:
            self._handle_dead_worker(worker)

    def _handle_dead_worker(self, worker: WorkerSlot):
        worker.process.join(timeout=0.1)

        lost_pending: PendingWork | None = None
        if worker.current_work_id is not None:
            with self.lock:
                lost_pending = self.in_flight.pop(worker.current_work_id, None)

        if worker.intentionally_stopped:
            return

        if lost_pending is None:
            return

        if self._should_retry_resource_crash(worker.process.exitcode, lost_pending):
            self._penalize_estimate(lost_pending)
            with self.lock:
                lost_pending.retry_count += 1
                lost_pending.assigned_gpu_id = None
                lost_pending.worker_id = None
                lost_pending.started_at = None
                lost_pending.result_ignored = False
                # Re-baseline observed usage on the next dispatch of this retry.
                lost_pending.worker_pid = None
                lost_pending.rss_baseline_bytes = None
                lost_pending.vram_baseline_gb = None
                lost_pending.observed_memory_gb = 0.0
                lost_pending.observed_vram_gb = 0.0
                lost_pending.observed_refreshed_at = None
                self.pending.appendleft(lost_pending)
            return

        self._settle_future_exception(
            lost_pending,
            RuntimeError(f"Worker {worker.worker_id} crashed with exit code {worker.process.exitcode}"),
        )
        self.completed_or_abandoned.add(lost_pending.item.id)

    def _should_retry_resource_crash(self, exitcode: int | None, pending: PendingWork) -> bool:
        if pending.retry_count >= self.max_resource_crash_retries:
            return False
        if exitcode is None:
            return False
        return exitcode in (-signal.SIGKILL, 137)

    def _penalize_estimate(self, pending: PendingWork):
        pending.estimate = ResourceEstimate(
            memory_gb=max(0.1, pending.estimate.memory_gb * 2.0),
            vram_gb=max(0.0, pending.estimate.vram_gb * 2.0),
            cpu_cores=max(1.0, pending.estimate.cpu_cores),
            confidence=pending.estimate.confidence,
        )
        pending.exclusive = True

    def _fail_pending_futures(self, exc: BaseException):
        with self.lock:
            while self.pending:
                pending = self.pending.popleft()
                # A queued task the caller cancelled is already done: drop it as
                # abandoned instead of raising InvalidStateError on shutdown.
                self._settle_future_exception(pending, exc)
                self.completed_or_abandoned.add(pending.item.id)

    # -- future settling (cancellation-safe) ----------------------------------

    def _settle_future_result(self, pending: PendingWork, value: object) -> None:
        """Resolve ``pending.future`` with ``value`` unless it is already done.

        A future can already be done because the caller cancelled it, a timeout
        fired, or the worker was declared dead — in which case a late result must
        be dropped rather than raise ``InvalidStateError`` into the background
        thread. ``done()`` is the primary guard; the narrow ``InvalidStateError``
        except closes the tiny race where a caller cancels between the check and
        the set (``cancel()`` only wins while the future is still pending).
        """
        if pending.future.done():
            logger.warning(
                "dropping result for already-done future work_id=%s",
                pending.item.id,
            )
            return
        try:
            pending.future.set_result(value)
        except InvalidStateError as exc:
            logger.warning(
                "future resolved concurrently work_id=%s error=%r",
                pending.item.id,
                exc,
            )

    def _settle_future_exception(self, pending: PendingWork, exc: BaseException) -> None:
        """Set ``exc`` on ``pending.future`` unless it is already done.

        Same guard rationale as :meth:`_settle_future_result`: a cancelled or
        already-resolved future must never take an exception (that would raise
        ``InvalidStateError`` and kill a background thread). Callers still record
        the task in ``completed_or_abandoned`` themselves.
        """
        if pending.future.done():
            logger.debug(
                "dropping exception for already-done future work_id=%s exc=%s",
                pending.item.id,
                type(exc).__name__,
            )
            return
        try:
            pending.future.set_exception(exc)
        except InvalidStateError as err:
            logger.warning(
                "future resolved concurrently work_id=%s error=%r",
                pending.item.id,
                err,
            )

    def _wait_for_in_flight(self):
        while True:
            self._check_workers()
            with self.lock:
                if not self.in_flight:
                    return
            time.sleep(0.05)

    def _stop_all_workers(self, wait: bool):
        for worker in list(self.workers.values()):
            worker.intentionally_stopped = True
            worker.work_queue.put(None)

        to_join = list(self.workers.values()) + list(self._retiring)

        if wait:
            for worker in to_join:
                worker.process.join(timeout=5.0)

        for worker in to_join:
            if not worker.process.is_alive():
                worker.process.join(timeout=0.1)

        self._retiring = [w for w in self._retiring if w.process.is_alive()]

    def _join_background_threads(self):
        for name in ("_dispatch_thread", "_timeout_thread", "_result_thread"):
            thread = getattr(self, name, None)
            if thread is not None:
                thread.join(timeout=2.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
