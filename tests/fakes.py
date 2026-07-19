"""Fakes/stubs for non-flaky, GPU-free testing.

No test touches real NVML or launches real GPU work; the machine has no GPU.
"""

import types


class FakeProcess:
    """Stand-in for mp.Process with controllable liveness."""

    def __init__(self, alive: bool = True):
        self._alive = alive
        self.exitcode = None
        self.join_timeouts: list[float | None] = []
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def set_dead(self, exitcode: int = 0):
        self._alive = False
        self.exitcode = exitcode


class FakeQueue:
    """Stand-in for mp.Queue that records puts."""

    def __init__(self):
        self.items: list = []

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        raise NotImplementedError


class FakeMonitor:
    """Minimal monitor exposing the VRAM measurement API used by Worker."""

    def __init__(self, per_process=None, device=None):
        # per_process/device may be a dict {device_id: value} or a scalar/None.
        self.per_process = per_process
        self.device = device
        self.per_process_calls: list[tuple[int, set[int]]] = []
        self.device_calls: list[int] = []

    def _lookup(self, table, device_id):
        if isinstance(table, dict):
            return table.get(device_id)
        return table

    def per_process_vram_gb(self, device_id, pids):
        self.per_process_calls.append((device_id, set(pids)))
        return self._lookup(self.per_process, device_id)

    def device_vram_used_gb(self, device_id):
        self.device_calls.append(device_id)
        return self._lookup(self.device, device_id)


class NVMLError(Exception):
    pass


class _MemInfo:
    def __init__(self, used, total):
        self.used = used
        self.total = total


class _Util:
    def __init__(self, gpu):
        self.gpu = gpu


class FakeComputeProc:
    def __init__(self, pid, used):
        self.pid = pid
        self.usedGpuMemory = used


def make_fake_pynvml(
    *,
    procs=None,
    available=("v3", "v2", "plain"),
    raise_procs=False,
    mem=None,
    raise_mem=False,
    raise_init=False,
):
    """Build a fake ``pynvml`` module namespace."""
    ns = types.SimpleNamespace()
    ns.NVMLError = NVMLError

    def get_procs(handle):
        if raise_procs:
            raise NVMLError("compute process query failed")
        return list(procs or [])

    if "v3" in available:
        ns.nvmlDeviceGetComputeRunningProcesses_v3 = get_procs
    if "v2" in available:
        ns.nvmlDeviceGetComputeRunningProcesses_v2 = get_procs
    if "plain" in available:
        ns.nvmlDeviceGetComputeRunningProcesses = get_procs

    def get_mem(handle):
        if raise_mem:
            raise NVMLError("memory info failed")
        return _MemInfo(*(mem or (0, 0)))

    ns.nvmlDeviceGetMemoryInfo = get_mem
    ns.nvmlDeviceGetUtilizationRates = lambda h: _Util(0)

    def nvml_init():
        if raise_init:
            raise NVMLError("init failed")

    ns.nvmlInit = nvml_init
    ns.nvmlDeviceGetCount = lambda: 1
    ns.nvmlDeviceGetHandleByIndex = lambda i: object()
    ns.nvmlShutdown = lambda: None
    return ns
