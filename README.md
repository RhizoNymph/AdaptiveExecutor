# Adaptive Executor

A resource-aware parallel executor for Python that learns from resource usage patterns to autoscale workers and avoid out-of-memory errors.

## Features

- **Automatic Resource Learning**: Tracks memory, VRAM, and CPU usage for each function and builds profiles over time
- **Adaptive Parallelism**: Adjusts the number of concurrent workers based on learned resource requirements
- **GPU Support**: Round-robin GPU assignment with VRAM tracking (requires `nvidia-ml-py`)
- **Profile Persistence**: Save and load learned profiles across runs
- **OOM Prevention**: Maintains configurable memory headroom to prevent out-of-memory crashes
- **Closed Learning Loop**: Dispatch-time re-estimation of queued tasks, a
  cold-start canary for first-contact workloads, and persistent crash (OOM)
  memory floors — so what the executor learns feeds straight back into admission

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

### Backfill scheduling

Dispatch is **not** strict head-of-line FIFO. When the task at the front of the
queue cannot be admitted right now (not enough free RAM/VRAM/worker slots), the
executor computes a *reservation* for it — the earliest time the currently
running tasks are expected to release enough resources for it to start — and then
lets **later** tasks jump ahead (backfill) only when doing so cannot delay that
reservation. A later task may run ahead if either it fits in capacity that is
free even after setting aside the head's reservation, or its expected duration
means it finishes before the reservation. The head never starts later than it
would under strict FIFO. Reservations respect per-GPU VRAM, and an exclusive
task (used by OOM-crash retries) at the front blocks all backfill. This is the
default and only behavior; there is no flag. See
`docs/features/backfill-scheduling.md`.

### Cold-start canary

The first time you submit a function the executor has **no profile** for it, so
every one of N simultaneous first submits gets the *same* default estimate and,
without protection, they would all be admitted together — a mass first-contact
OOM, exactly the failure this library exists to prevent.

The cold-start canary prevents that: while a function's estimation profile is
**cold** (zero observations), at most **one** task of that profile identity runs
at a time. The first task is the canary; its siblings wait. When it completes it
records an observation, the profile is no longer cold, and (thanks to
dispatch-time re-estimation) the queued siblings automatically unclamp with a
*learned* estimate. A cold-blocked task never stalls unrelated feasible work —
other functions and disjoint tasks continue to backfill past it.

The identity is per profile bucket: with `profile_key`, each keyed bucket gets
its own canary; a keyed submit that falls back to a base profile that already
has observations is not cold. Passing an explicit `memory_gb` / `vram_gb` hint
**bypasses** the canary entirely — the hint is you asserting you already know the
resource cost, so no first-contact protection is needed.

### Dispatch-time re-estimation

A task's estimate is not frozen at submit. On every dispatch cycle the executor
refreshes each queued task's estimate from the current profile, so a task
waiting behind a long backlog benefits from the profile sharpening as its
siblings complete. User hints keep overriding the learned value, and a
crash-penalized retry (whose estimate was deliberately doubled) is never
weakened.

### Crash floors (persistent OOM knowledge)

When a worker is SIGKILLed under memory pressure (an OOM), the crashing run
reports **no** observation — the worker died before it could measure anything —
so naively the profile learns nothing and the *next* batch of submits repeats
the same crash. To stop that, a crash records a persistent **memory floor** into
the profile: the estimate the task was admitted under, now proven too small.

- The floor **lower-bounds** future memory estimates for that function
  (`max(computed_estimate, floor)`), applied after the safety margin.
- It is written to both the base profile and, if the crash happened under a
  `profile_key`, that keyed bucket.
- It only **ratchets up** while active, and it is **cleared** once the workload
  demonstrably shrinks — after five consecutive successful runs whose peak
  memory is below half the floor — so a fixed workload is not penalized forever.
- Floors persist across restarts. Only host RAM is floored: an OOM/SIGKILL is a
  host-memory event; VRAM exhaustion surfaces as an in-process CUDA error rather
  than a SIGKILL, so it is not treated as evidence a VRAM estimate was too low.

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

## Input-aware profiles

Profiles are learned per function. But some functions use wildly different
amounts of memory depending on their **input** — `process(small_file)` versus
`process(huge_file)`. Merged into one profile, the p90 estimate either OOMs on
the big inputs or needlessly throttles the small ones.

Pass an optional `profile_key` — an opaque string you choose — to bucket inputs
that behave alike. Each bucket learns its own distribution, so estimation and
admission control adapt to the input at hand:

```python
def bucket_for(path: str) -> str:
    size_gb = os.path.getsize(path) / 1e9
    if size_gb < 1:
        return "small"
    if size_gb < 10:
        return "medium"
    return "large"

with AdaptiveExecutor(profile_path="profiles.json") as executor:
    futures = [
        executor.submit(process, path, profile_key=bucket_for(path))
        for path in paths
    ]
    results = [f.result() for f in futures]
```

Semantics:

- **Storage.** A keyed profile is stored under a derived key
  `module:qualname#profile_key`; the base profile stays at `module:qualname`.
  Keys are opaque strings and are not escaped — the only convention is that a
  base key never contains `#`, so a keyed entry can never collide with a base
  entry.
- **Recording.** Every observation is recorded into **both** the keyed profile
  (when a key was given) and the base profile, so the base remains an aggregate
  fallback across all inputs.
- **Estimation.** With a `profile_key`, the keyed profile is used once it has at
  least one observation; until then the estimate falls back to the base profile
  (with its usual confidence and safety-margin behavior). No `profile_key`
  behaves exactly as before. Explicit `memory_gb` / `vram_gb` hints still
  override.
- **Feasibility.** The submit-time infeasibility check uses whichever estimate
  applies — so a bucket known to be too big for the machine fails fast, while
  smaller buckets of the same function keep flowing.
- **Persistence.** Keyed profiles round-trip through the JSON store exactly like
  base profiles (the store keys are just strings).

## How It Works

1. **Submission**: When you submit work, the executor looks up the function's resource profile
2. **Estimation**: If no profile exists, uses conservative defaults. Otherwise, uses the 90th percentile of observed usage plus a safety margin
3. **Admission Control**: Only admits work if projected memory + committed resources < available - headroom
4. **Backfill Scheduling**: When the head of the queue is blocked, later tasks may run ahead if they cannot delay the head's resource reservation (per-GPU aware)
5. **Execution**: Workers execute tasks and measure actual resource usage
6. **Learning**: Observations (including run duration) are recorded and used to improve future estimates
7. **Closing the loop**: Queued tasks are re-estimated as the profile sharpens, a cold-start canary gates first-contact bursts, and OOM crashes record a persistent memory floor so the next batch does not repeat the crash

## Requirements

- Python 3.12+
- psutil
- nvidia-ml-py (optional, for GPU support; imported as `pynvml` at runtime)
