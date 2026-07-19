import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback
from queue import Empty

import psutil

from .dtypes import ResourceObservation, WorkItem, WorkResult
from .monitor import ResourceMonitor
from .resolve import resolve_function

logger = logging.getLogger("adaptive_executor.worker")


def _make_picklable_exception(exc: BaseException) -> Exception:
    """Convert an exception to a form that can be safely pickled.

    Some exceptions hold references to unpicklable objects (lambdas, closures,
    etc). When pickling fails, fall back to a RuntimeError carrying the message
    and original traceback.
    """
    try:
        pickle.dumps(exc)
        return exc  # type: ignore[return-value]
    except (pickle.PicklingError, TypeError, AttributeError):
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return RuntimeError(f"{type(exc).__name__}: {exc}\n\nOriginal traceback:\n{tb_str}")


class Worker:
    """Runs in a subprocess. Receives work, executes it, observes resource
    usage, and sends results back.

    The worker is pinned to a single GPU (NVML index) for its entire lifetime
    via ``CUDA_VISIBLE_DEVICES``. Because NVML ignores that variable, VRAM is
    measured against the pinned NVML device directly, preferring per-process
    attribution for this worker's process tree.
    """

    def __init__(
        self,
        work_queue: mp.Queue,
        result_queue: mp.Queue,
        worker_id: int,
        pinned_gpu_id: int | None,
    ):
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.pinned_gpu_id = pinned_gpu_id
        self.monitor = ResourceMonitor(poll_interval=0.05)
        self._monitor_started = False

    def run(self):
        """Main loop - runs in subprocess."""
        # Pin the process to its assigned GPU for its entire lifetime.
        if self.pinned_gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.pinned_gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.monitor.start()
        self._monitor_started = True

        while True:
            try:
                item: WorkItem | None = self.work_queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            result = self._execute(item)
            self.result_queue.put(result)

        if self._monitor_started:
            self.monitor.stop()

    def _execute(self, item: WorkItem) -> WorkResult:
        try:
            fn = resolve_function(item.fn_module, item.fn_name)
        except ValueError as exc:
            logger.error(
                "function resolution failed worker_id=%d fn=%s:%s error=%s",
                self.worker_id,
                item.fn_module,
                item.fn_name,
                exc,
            )
            return WorkResult(
                id=item.id,
                worker_id=self.worker_id,
                success=False,
                result=None,
                exception=_make_picklable_exception(exc),
                observation=ResourceObservation(0, 0, 0, 0),
            )

        return self._execute_with_observation(fn, item.args, item.kwargs, item.id)

    def _process_tree_pids(self) -> set[int]:
        """PIDs of this worker process plus its descendants."""
        pids = {os.getpid()}
        try:
            for child in psutil.Process().children(recursive=True):
                pids.add(child.pid)
        except psutil.Error as exc:
            logger.debug("failed to enumerate child pids error=%r", exc)
        return pids

    def _gpu_vram_gb(self) -> float:
        """Current VRAM (GB) attributable to this worker on its pinned device.

        Returns 0.0 when the worker is not pinned to a GPU. Prefers per-process
        attribution (this process tree); falls back to the pinned device's total
        used memory; returns 0.0 if no GPU measurement is available. The device
        queried is the pinned NVML index.
        """
        if self.pinned_gpu_id is None:
            return 0.0

        pids = self._process_tree_pids()
        per_process = self.monitor.per_process_vram_gb(self.pinned_gpu_id, pids)
        if per_process is not None:
            return per_process

        device = self.monitor.device_vram_used_gb(self.pinned_gpu_id)
        if device is not None:
            return device

        return 0.0

    def _execute_with_observation(
        self,
        fn,
        args: tuple,
        kwargs: dict,
        work_id: str,
    ) -> WorkResult:
        process = psutil.Process()
        poll_gpu = self.pinned_gpu_id is not None

        before_memory_gb = process.memory_info().rss / 1e9
        before_vram = self._gpu_vram_gb()

        peak_memory_gb = before_memory_gb
        peak_vram = before_vram
        cpu_samples: list[float] = []

        process.cpu_percent()
        stop_tracking = threading.Event()

        def track():
            nonlocal peak_memory_gb, peak_vram
            while not stop_tracking.is_set():
                try:
                    current_mem = process.memory_info().rss / 1e9
                    peak_memory_gb = max(peak_memory_gb, current_mem)

                    if poll_gpu:
                        peak_vram = max(peak_vram, self._gpu_vram_gb())

                    cpu_samples.append(process.cpu_percent())
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)

        tracker = threading.Thread(target=track, daemon=True)
        tracker.start()

        start = time.time()
        success = True
        result = None
        exception = None

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            success = False
            exception = _make_picklable_exception(exc)
        finally:
            stop_tracking.set()
            tracker.join(timeout=0.2)

        duration = time.time() - start
        observation = ResourceObservation(
            memory_delta_gb=peak_memory_gb - before_memory_gb,
            vram_delta_gb=max(0.0, peak_vram - before_vram),
            cpu_percent=sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            duration_seconds=duration,
        )

        return WorkResult(
            id=work_id,
            worker_id=self.worker_id,
            success=success,
            result=result,
            exception=exception,
            observation=observation,
        )


def worker_process_entry(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
    pinned_gpu_id: int | None,
):
    """Entry point for worker subprocess."""
    worker = Worker(work_queue, result_queue, worker_id, pinned_gpu_id)
    worker.run()
