import logging
import threading
import time

import psutil

from .dtypes import GPUSnapshot, ResourceSnapshot

logger = logging.getLogger("adaptive_executor.monitor")


def _import_nvml():
    """Import NVML bindings from the supported nvidia-ml-py distribution.

    The maintained PyPI package is ``nvidia-ml-py``, but it still exposes the
    ``pynvml`` module namespace for the actual bindings.
    """
    import pynvml

    return pynvml


def _nvml_error_type(pynvml) -> type[BaseException]:
    """Return the NVML error class, falling back to ``Exception``."""
    err = getattr(pynvml, "NVMLError", None)
    if isinstance(err, type) and issubclass(err, BaseException):
        return err
    return Exception


class ResourceMonitor:
    """Monitors system resources from any process.

    GPU device ids are NVML indices and are independent of
    ``CUDA_VISIBLE_DEVICES`` (which NVML ignores).
    """

    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self._current: ResourceSnapshot | None = None
        self._lock = threading.Lock()
        self._running = False
        self._gpu_handles: list[tuple[int, object]] = []
        self._warned_devices: set[int] = set()

    def start(self):
        if self._running:
            return
        self._running = True
        self._init_nvml()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=self.poll_interval * 2 + 0.1)
        self._shutdown_nvml()

    def _init_nvml(self):
        try:
            pynvml = _import_nvml()
        except ImportError:
            logger.debug("nvml unavailable, gpu monitoring disabled")
            self._gpu_handles = []
            return

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                (i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(count)
            ]
            logger.info("nvml initialized gpu_count=%d", count)
        except Exception as exc:
            logger.warning("nvml init failed, gpu monitoring disabled error=%r", exc)
            self._gpu_handles = []

    def _shutdown_nvml(self):
        if not self._gpu_handles:
            return
        try:
            pynvml = _import_nvml()
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("nvml shutdown failed error=%r", exc)
        self._gpu_handles = []

    def _poll_loop(self):
        while self._running:
            snapshot = self.snapshot()
            with self._lock:
                self._current = snapshot
            time.sleep(self.poll_interval)

    def snapshot(self) -> ResourceSnapshot:
        """Take a snapshot right now (can be called directly)."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)

        gpus: dict[int, GPUSnapshot] = {}
        if self._gpu_handles:
            pynvml = _import_nvml()
            nvml_error = _nvml_error_type(pynvml)
            for i, handle in self._gpu_handles:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpus[i] = GPUSnapshot(
                        device_id=i,
                        vram_used_gb=info.used / 1e9,
                        vram_total_gb=info.total / 1e9,
                        utilization_percent=util.gpu,
                    )
                except nvml_error as exc:
                    if i not in self._warned_devices:
                        self._warned_devices.add(i)
                        logger.debug("gpu snapshot failed device_id=%d error=%r", i, exc)

        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_used_gb=mem.used / 1e9,
            memory_total_gb=mem.total / 1e9,
            gpus=gpus,
        )

    def _handle_for(self, device_id: int) -> object | None:
        for i, handle in self._gpu_handles:
            if i == device_id:
                return handle
        return None

    def device_vram_used_gb(self, device_id: int) -> float | None:
        """Total used VRAM (GB) on NVML device ``device_id``, or None if unknown."""
        snapshot = self.current or self.snapshot()
        gpu = snapshot.gpus.get(device_id)
        if gpu is None:
            return None
        return gpu.vram_used_gb

    def per_process_vram_gb(self, device_id: int, pids: set[int]) -> float | None:
        """Sum VRAM (GB) used by ``pids`` on NVML device ``device_id``.

        Uses ``nvmlDeviceGetComputeRunningProcesses_v3`` (with ``_v2`` and the
        unversioned symbol as fallbacks). Returns None when NVML or the
        per-process API is unavailable, so callers can fall back to the
        device-wide used-memory delta.
        """
        handle = self._handle_for(device_id)
        if handle is None:
            return None
        try:
            pynvml = _import_nvml()
        except ImportError:
            return None

        procs = self._compute_running_processes(pynvml, handle)
        if procs is None:
            return None

        total = 0.0
        for proc in procs:
            if proc.pid not in pids:
                continue
            used = getattr(proc, "usedGpuMemory", None)
            if used:
                total += used / 1e9
        return total

    def _compute_running_processes(self, pynvml, handle):
        nvml_error = _nvml_error_type(pynvml)
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(pynvml, name, None)
            if fn is None:
                continue
            try:
                return fn(handle)
            except nvml_error as exc:
                logger.debug("per-process nvml query failed api=%s error=%r", name, exc)
                return None
        return None

    @property
    def current(self) -> ResourceSnapshot | None:
        with self._lock:
            return self._current
