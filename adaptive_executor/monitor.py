import threading
import time
import psutil

from .dtypes import ResourceSnapshot, GPUSnapshot


def _import_nvml():
    """Import NVML bindings from the supported nvidia-ml-py distribution."""
    # The maintained PyPI package is `nvidia-ml-py`, but it still exposes the
    # `pynvml` module namespace for the actual bindings.
    import pynvml

    return pynvml


class ResourceMonitor:
    """Monitors system resources from any process"""
    
    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self._current: ResourceSnapshot | None = None
        self._lock = threading.Lock()
        self._running = False
        self._gpu_handles = []
    
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
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                (i, pynvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(count)
            ]
        except Exception:
            self._gpu_handles = []

    def _shutdown_nvml(self):
        if self._gpu_handles:
            try:
                pynvml = _import_nvml()
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._gpu_handles = []

    def _poll_loop(self):
        while self._running:
            snapshot = self.snapshot()
            with self._lock:
                self._current = snapshot
            time.sleep(self.poll_interval)
    
    def snapshot(self) -> ResourceSnapshot:
        """Take a snapshot right now (can be called directly)"""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        
        gpus = {}
        if self._gpu_handles:
            pynvml = _import_nvml()
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
                except Exception:
                    pass
        
        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_used_gb=mem.used / 1e9,
            memory_total_gb=mem.total / 1e9,
            gpus=gpus,
        )
    
    @property
    def current(self) -> ResourceSnapshot | None:
        with self._lock:
            return self._current
