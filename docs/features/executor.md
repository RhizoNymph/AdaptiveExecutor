# Feature: Adaptive Executor

## Scope
- Public API: `AdaptiveExecutor.submit`, `start`, `shutdown`, context-manager use.
- Submit-time validation of callable importability.
- Admission control from learned profiles + live resource snapshots.
- Worker-pool lifecycle: spawn, reuse (by pin), evict-and-replace on pin
  mismatch at the cap, recycle after N tasks, reap retired workers.
- GPU round-robin assignment across NVML devices; per-worker GPU pinning.
- Per-run resource measurement (RSS/VRAM/CPU) and learning.
- Debounced, atomic profile persistence with flush-on-shutdown.
- Timeout handling and resource-crash retry.

## Non-scope
- Real GPU/NVML behavior is not exercised by tests (no GPU on CI); GPU paths are
  covered with fakes.
- Distributed / multi-host execution.
- Backwards-compatibility shims for prior persistence formats.

## Data / control flow
1. **submit(fn, *args, memory_gb=, vram_gb=, **kwargs)**
   - `validate_submittable(fn)` rejects lambdas/closures (`<locals>` in
     qualname), unimportable callables, and objects that re-resolve to something
     other than `fn` (e.g. bound methods) with a clear `ValueError`.
   - Builds a `WorkItem(module, qualname, args, kwargs)`.
   - `ProfileStore.get` returns a snapshot `LearnedProfile`; `estimate()` yields a
     `ResourceEstimate` (p90 + confidence-scaled safety margin, or hints).
   - Appends `PendingWork` to `self.pending`.
2. **_dispatch_loop** (thread): `_check_workers()` then `_maybe_dispatch()`.
   - `_maybe_dispatch()` gathers live state via `_build_dispatch_plan()` and
     calls the pure `scheduling.plan_dispatch()` (reservation-based backfill).
     `_build_dispatch_plan()` computes admittable memory (snapshot memory minus
     used, headroom, and committed in-flight estimates), per-GPU admittable VRAM
     (via `_committed_vram_per_gpu`), and each running task's remaining time
     (via `_running_remaining_seconds`, from `duration_p90_seconds` vs elapsed).
     When no monitor snapshot exists yet, memory/VRAM are treated as infinite
     (non-gating). The scheduler returns an ordered list of `DispatchDecision`
     (pending id + GPU); the executor executes them, updating `_next_gpu_index`.
     Full semantics are in `docs/features/backfill-scheduling.md`. Admission
     arithmetic (memory headroom against committed estimates, cpu-cores derived
     effective max, per-GPU round-robin VRAM fit) now lives inside the pure
     scheduler.
   - `_get_or_spawn_idle_worker(gpu_id)`:
     - reuse an alive idle worker with matching pin; else
     - if below cap, spawn a worker pinned to `gpu_id`; else
     - at the cap, evict one idle mismatched worker (`_retire_worker`) and spawn
       a replacement; else (all busy) return `None` (backpressure).
   - On success: move to `in_flight`, mark worker busy, send the `WorkItem`.
     Dispatched ids are removed from `self.pending` (which may be non-contiguous
     when tasks backfilled past a blocked head).
3. **worker** (subprocess): resolves fn via shared `resolve_function`, runs it,
   samples RSS and (only when pinned) pinned-GPU VRAM, returns a `WorkResult`.
4. **_collect_results** (thread): clears the worker's `current_work_id`,
   increments `tasks_completed`, recycles the worker if it hit
   `worker_recycle_after_tasks`, pops `in_flight`, records the observation
   (`ProfileStore.record`, debounced persist), and resolves the future.
5. **_check_timeouts** (thread): fails or kills workers for tasks exceeding
   `task_timeout_seconds` per `on_timeout`.
6. **shutdown**: stops accepting, fails queued futures, drains in-flight, stops
   all workers (including `_retiring`), joins threads, `profiles.flush()`,
   stops the monitor.

## Files and roles
- `adaptive_executor/adaptive_executor.py` — `AdaptiveExecutor`, `WorkerSlot`
  (adds `tasks_completed`), `PendingWork`. Key methods:
  `_get_or_spawn_idle_worker`, `_find_idle_evictable_worker`, `_retire_worker`,
  `_should_recycle`, `_check_workers` (reaps `_retiring`), `_maybe_dispatch`,
  `_build_dispatch_plan`, `_running_remaining_seconds`, `_committed_resources`,
  `_committed_vram_per_gpu`, `_collect_results`.
- `adaptive_executor/scheduling.py` — pure reservation-based backfill scheduler
  (`plan_dispatch`); see `docs/features/backfill-scheduling.md`.
- `adaptive_executor/worker.py` — `Worker`, `worker_process_entry`. Key:
  `_gpu_vram_gb` (pinned-only, per-process preferred), `_process_tree_pids`,
  `_execute_with_observation`.
- `adaptive_executor/monitor.py` — `ResourceMonitor`. Key exports:
  `snapshot`, `current`, `device_vram_used_gb`, `per_process_vram_gb`,
  `_compute_running_processes` (v3/v2/unversioned fallback).
- `adaptive_executor/profiles.py` — `LearnedProfile`, `ProfileStore`. Key:
  `record`, `flush`, `_should_save_locked`, `_snapshot_locked`,
  `_persist_snapshot`, `_write_atomic`, `_load`.
- `adaptive_executor/resolve.py` — `resolve_function`, `validate_submittable`,
  `FunctionResolutionError`.
- `adaptive_executor/dtypes.py` — dataclasses.

## Invariants and constraints
- A `WorkerSlot` in `self.workers` is alive or about to be reaped by
  `_check_workers`; retired workers live only in `self._retiring` until joined.
- Workers are pinned to one GPU (NVML index) for their lifetime; a pin change
  requires evicting the worker and spawning a replacement.
- Committed estimates always cover in-flight tasks; admission uses
  committed + new estimate against available resources minus headroom.
- Only an idle worker (`current_work_id is None`, alive, not
  `intentionally_stopped`) may be reused, evicted, or recycled.
- VRAM is attributed to the worker's pinned device only; an unpinned worker
  records `vram_delta_gb == 0.0` and performs no GPU polling.
- `_retire_worker`/recycle/evict mutate `self.workers` and `self._retiring` while
  holding `self.lock`.
- Persistence is debounced (`save_every_n` observations or
  `save_interval_seconds`); serialization happens under the store lock, file I/O
  outside it under a separate save lock; `flush()` (and `shutdown`) force a save.
- Profile save/load failures are logged (warning/error), never raised into the
  record path; a corrupt file yields an empty store, not a crash.
