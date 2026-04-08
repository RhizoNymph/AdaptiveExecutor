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
python3 test_executor.py
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
    max_workers=8,              # Maximum concurrent workers
    gpu_ids=[0, 1],             # GPUs to use (None = auto-detect)
    profile_path="profiles.json",  # Persist learned profiles
    memory_headroom_gb=2.0,     # RAM to keep free
    vram_headroom_gb=1.0,       # VRAM to keep free per GPU
    task_timeout_seconds=300.0, # Task timeout
)
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
