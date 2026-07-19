# Adaptive Executor

A resource-aware parallel executor for Python that learns from resource usage patterns to autoscale workers and avoid out-of-memory errors.

## Features

- **Automatic Resource Learning**: Tracks memory, VRAM, and CPU usage for each function and builds profiles over time
- **Adaptive Parallelism**: Adjusts the number of concurrent workers based on learned resource requirements
- **GPU Support**: Round-robin GPU assignment with VRAM tracking (requires `nvidia-ml-py`)
- **Profile Persistence**: Save and load learned profiles across runs
- **OOM Prevention**: Maintains configurable memory headroom to prevent out-of-memory crashes

## Installation

```bash
pip install -e .

# For GPU support:
pip install -e ".[gpu]"
```

## Testing

```bash
uv run --group dev pytest
```

## Quick Start

```python
from adaptive_executor import AdaptiveExecutor

def process_data(data):
    # Your memory/compute intensive work
    result = expensive_computation(data)
    return result

# Use as context manager
with AdaptiveExecutor(max_workers=8, memory_headroom_gb=2.0) as executor:
    futures = [executor.submit(process_data, item) for item in dataset]
    results = [f.result() for f in futures]
```

## Configuration

```python
executor = AdaptiveExecutor(
    max_workers=8,               # Maximum concurrent workers
    gpu_ids=[0, 1],              # GPUs to use (None = auto-detect)
    profile_path="profiles.json",   # Persist learned profiles
    memory_headroom_gb=2.0,      # RAM to keep free
    vram_headroom_gb=1.0,        # VRAM to keep free per GPU
    task_timeout_seconds=300.0,  # Task timeout
    worker_recycle_after_tasks=50,  # Retire+respawn a worker after N tasks (None disables)
)
```

### Worker recycling

`worker_recycle_after_tasks` (default `50`) retires a worker once it has
completed that many tasks and spawns a fresh one on demand. CPython rarely
returns freed memory to the OS, so a long-lived worker's RSS baseline ratchets
upward; recycling keeps memory observations accurate. Pass `None` to disable.

### Profile persistence

When `profile_path` is set, learned profiles are persisted with **debounced**
writes: the store saves when either `save_every_n` observations (default `20`)
have accumulated or `save_interval_seconds` (default `5.0`) have elapsed since
the last save — whichever comes first. Writes are atomic (temp file +
`os.replace`) and happen outside the store lock. `shutdown()` flushes any
pending observations so nothing is lost on a clean exit.

### Submitting functions

Submitted callables must be importable in a worker subprocess by their module
and qualified name. Lambdas, closures (functions defined inside other
functions), and bound methods are rejected at submit time with a clear
`ValueError`. For an instance method, submit the underlying function and pass
the instance as the first argument, e.g. `executor.submit(Cls.method, instance,
...)`.

### Infeasible tasks

A task whose resource estimate can never fit on this machine — for example a
64 GB RAM estimate on a 32 GB box, or a VRAM estimate larger than every GPU —
would otherwise sit at the head of the FIFO queue forever, silently blocking
every task behind it. The executor detects this and fails the task instead:

- **At submit time**, `submit()` raises `InfeasibleTaskError` synchronously when
  the estimate already exceeds total capacity minus headroom, so you can catch
  it right at the call site.
- **At dispatch time**, if an estimate becomes infeasible later (a task killed
  under memory pressure has its estimate doubled by crash-retry penalization),
  the affected task's future fails with `InfeasibleTaskError` while the executor
  keeps running and moves on to the next task.

Infeasibility means "exceeds total capacity" (a permanent condition), not
"doesn't fit right now" (normal queuing). It is only declared when capacity is
actually known from a monitor snapshot; if capacity is unknown, admission is
unchanged.

```python
from adaptive_executor import AdaptiveExecutor, InfeasibleTaskError

with AdaptiveExecutor() as executor:
    try:
        future = executor.submit(train, dataset, memory_gb=64.0)
    except InfeasibleTaskError as err:
        # Structured fields, not just a message:
        print(err.kind)         # "memory" or "vram"
        print(err.estimate_gb)  # what the task needs
        print(err.capacity_gb)  # usable capacity it exceeded
        print(err.retry_count)  # >0 if a crash-retry penalty caused it
```

## Resource Hints

You can provide hints if you know the resource requirements upfront:

```python
# Hint expected resource usage
future = executor.submit(
    heavy_function, 
    arg1, arg2,
    memory_gb=4.0,  # Expected RAM usage
    vram_gb=2.0,    # Expected VRAM usage
)
```

For first-run GPU workloads, a nonzero `vram_gb` hint is important if the function has no learned profile yet. Otherwise the executor may initially treat the task as CPU-only until it has observed a GPU-backed run.

## How It Works

1. **Submission**: When you submit work, the executor looks up the function's resource profile
2. **Estimation**: If no profile exists, uses conservative defaults. Otherwise, uses the 90th percentile of observed usage plus a safety margin
3. **Admission Control**: Only admits work if projected memory + committed resources < available - headroom
4. **Execution**: Workers execute tasks and measure actual resource usage
5. **Learning**: Observations are recorded and used to improve future estimates

## Requirements

- Python 3.12+
- psutil
- nvidia-ml-py (optional, for GPU support; imported as `pynvml` at runtime)
