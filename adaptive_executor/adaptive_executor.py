import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from queue import Empty
from typing import Callable, Literal

from .dtypes import ResourceEstimate, WorkItem, WorkResult
from .monitor import ResourceMonitor
from .profiles import ProfileStore
from .worker import worker_process_entry


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


@dataclass
class WorkerSlot:
    worker_id: int
    process: mp.Process
    work_queue: mp.Queue
    pinned_gpu_id: int | None
    current_work_id: str | None = None
    intentionally_stopped: bool = False
    pending_retry_work_id: str | None = None


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
    ):
        self.max_workers = max_workers or os.cpu_count() or 4
        self.gpu_ids = gpu_ids
        self.memory_headroom_gb = memory_headroom_gb
        self.vram_headroom_gb = vram_headroom_gb
        self.task_timeout_seconds = task_timeout_seconds
        self.on_timeout = on_timeout
        self.max_resource_crash_retries = max_resource_crash_retries
        self._next_gpu_index = 0
        self._next_worker_id = 0

        self.monitor = ResourceMonitor()
        self.profiles = ProfileStore(persist_path=profile_path)

        self.result_queue: mp.Queue = mp.Queue()
        self.workers: dict[int, WorkerSlot] = {}

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
        self.monitor.stop()

    def submit(
        self,
        fn: Callable,
        *args,
        memory_gb: float | None = None,
        vram_gb: float | None = None,
        **kwargs,
    ) -> Future:
        """
        Submit work for execution.

        Args:
            fn: Function to execute (must be importable)
            *args: Positional arguments
            memory_gb: Optional hint for expected RAM usage
            vram_gb: Optional hint for expected VRAM usage
            **kwargs: Keyword arguments

        Returns:
            Future that will contain the result
        """
        if self._shutdown or not self._accepting:
            raise RuntimeError("Cannot submit to a shutdown AdaptiveExecutor")
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

        profile = self.profiles.get(item.fn_module, item.fn_name)
        estimate = profile.estimate(memory_hint=memory_gb, vram_hint=vram_gb)

        future = Future()
        pending = PendingWork(
            item=item,
            future=future,
            estimate=estimate,
            submitted_at=time.time(),
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
            while self.pending:
                pending = self.pending[0]
                can_admit, gpu_id = self._can_admit(pending)
                if not can_admit:
                    break

                worker = self._get_or_spawn_idle_worker(gpu_id)
                if worker is None:
                    break

                self.pending.popleft()
                pending.assigned_gpu_id = gpu_id
                pending.worker_id = worker.worker_id
                pending.started_at = time.time()
                self.in_flight[pending.item.id] = pending
                worker.current_work_id = pending.item.id
                worker.work_queue.put(replace(pending.item, gpu_id=gpu_id))

    def _can_admit(self, pending: PendingWork) -> tuple[bool, int | None]:
        estimate = pending.estimate
        snapshot = self.monitor.current

        workers_ok = len(self.in_flight) < self.max_workers
        if pending.exclusive and self.in_flight:
            return False, None
        if not workers_ok:
            return False, None

        if snapshot is None:
            if estimate.vram_gb > 0 and self.gpu_ids:
                gpu_id = self._pick_round_robin_gpu(None, estimate)
                return gpu_id is not None, gpu_id
            return True, None

        committed = self._committed_resources()
        available_memory = snapshot.memory_total_gb - snapshot.memory_used_gb - self.memory_headroom_gb
        memory_ok = committed.memory_gb + estimate.memory_gb < available_memory
        if not memory_ok:
            return False, None

        assigned_gpu = None
        if estimate.vram_gb > 0:
            assigned_gpu = self._pick_round_robin_gpu(snapshot, estimate)
            if assigned_gpu is None:
                return False, None

        effective_max = self.max_workers
        if estimate.cpu_cores > 1:
            effective_max = max(1, int(self.max_workers / estimate.cpu_cores))

        return len(self.in_flight) < effective_max, assigned_gpu

    def _pick_round_robin_gpu(self, snapshot, estimate: ResourceEstimate) -> int | None:
        if not self.gpu_ids:
            return None

        committed_per_gpu = self._committed_vram_per_gpu()
        num_gpus = len(self.gpu_ids)

        for offset in range(num_gpus):
            gpu_idx = (self._next_gpu_index + offset) % num_gpus
            gpu_id = self.gpu_ids[gpu_idx]

            if snapshot is not None:
                if gpu_id not in snapshot.gpus:
                    continue
                gpu = snapshot.gpus[gpu_id]
                available = gpu.vram_total_gb - gpu.vram_used_gb - self.vram_headroom_gb
                if committed_per_gpu.get(gpu_id, 0.0) + estimate.vram_gb >= available:
                    continue

            self._next_gpu_index = (gpu_idx + 1) % num_gpus
            return gpu_id

        return None

    def _committed_resources(self) -> ResourceEstimate:
        return ResourceEstimate(
            memory_gb=sum(p.estimate.memory_gb for p in self.in_flight.values()),
            vram_gb=sum(p.estimate.vram_gb for p in self.in_flight.values()),
            cpu_cores=sum(p.estimate.cpu_cores for p in self.in_flight.values()),
        )

    def _committed_vram_per_gpu(self) -> dict[int, float]:
        committed: dict[int, float] = {}
        for pending in self.in_flight.values():
            if pending.assigned_gpu_id is not None:
                committed[pending.assigned_gpu_id] = committed.get(pending.assigned_gpu_id, 0.0) + pending.estimate.vram_gb
        return committed

    def _get_or_spawn_idle_worker(self, pinned_gpu_id: int | None) -> WorkerSlot | None:
        for worker in self.workers.values():
            if worker.pinned_gpu_id == pinned_gpu_id and worker.current_work_id is None and worker.process.is_alive():
                return worker

        if len(self.workers) >= self.max_workers:
            return None

        return self._spawn_worker(pinned_gpu_id)

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

            with self.lock:
                worker = self.workers.get(result.worker_id)
                if worker is not None and worker.current_work_id == result.id:
                    worker.current_work_id = None

                pending = self.in_flight.pop(result.id, None)
                if pending is None:
                    continue

                if pending.result_ignored:
                    self.completed_or_abandoned.add(result.id)
                    continue

            self.profiles.record(
                pending.item.fn_module,
                pending.item.fn_name,
                result.observation,
            )

            if result.success:
                pending.future.set_result(result.result)
            else:
                pending.future.set_exception(result.exception)

    def _check_timeouts(self):
        while not self._threads_should_stop:
            time.sleep(0.1)
            now = time.time()
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

        active.future.set_exception(
            TimeoutError(f"Task {active.item.id} timed out after {self.task_timeout_seconds}s")
        )
        self.completed_or_abandoned.add(active.item.id)

    def _check_workers(self):
        dead_workers: list[WorkerSlot] = []
        with self.lock:
            for worker in list(self.workers.values()):
                if worker.process.is_alive():
                    continue
                dead_workers.append(worker)
                self.workers.pop(worker.worker_id, None)

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
                self.pending.appendleft(lost_pending)
            return

        lost_pending.future.set_exception(
            RuntimeError(f"Worker {worker.worker_id} crashed with exit code {worker.process.exitcode}")
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
                pending.future.set_exception(exc)
                self.completed_or_abandoned.add(pending.item.id)

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

        if wait:
            for worker in list(self.workers.values()):
                worker.process.join(timeout=5.0)

        for worker in list(self.workers.values()):
            if not worker.process.is_alive():
                worker.process.join(timeout=0.1)

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
