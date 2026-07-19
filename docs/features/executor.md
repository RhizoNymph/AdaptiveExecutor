# Feature: Adaptive Executor

## Scope
- Public API: `AdaptiveExecutor.submit` (incl. optional `profile_key` input
  bucketing), `start`, `shutdown`, context-manager use.
- Submit-time validation of callable importability.
- Admission control from learned profiles + live resource snapshots.
- Worker-pool lifecycle: spawn, reuse (by pin), evict-and-replace on pin
  mismatch at the cap, recycle after N tasks, reap retired workers.
- GPU round-robin assignment across NVML devices; per-worker GPU pinning.
- Per-run resource measurement (RSS/VRAM/CPU) and learning.
- Debounced, atomic profile persistence with flush-on-shutdown.
- Timeout handling and resource-crash retry.
- The learning loop (three interlocking mechanisms):
  - **Dispatch-time re-estimation**: a queued task's estimate is refreshed from
    the current profile each dispatch cycle, so a task waiting behind a backlog
    benefits from its siblings' completed observations.
  - **Cold-start canary**: while a profile identity has zero observations, at
    most one task of that identity runs at a time (enforced in the pure
    scheduler; see `docs/features/backfill-scheduling.md`).
  - **Crash floors**: an OOM/SIGKILL crash records a persistent `memory_floor_gb`
    into the profile so the next batch of fresh submits does not repeat the crash.

## Non-scope
- Real GPU/NVML behavior is not exercised by tests (no GPU on CI); GPU paths are
  covered with fakes.
- Distributed / multi-host execution.
- Backwards-compatibility shims for prior persistence formats.

## Data / control flow
1. **submit(fn, *args, memory_gb=, vram_gb=, profile_key=, **kwargs)**
   - `validate_submittable(fn)` rejects lambdas/closures (`<locals>` in
     qualname), unimportable callables, and objects that re-resolve to something
     other than `fn` (e.g. bound methods) with a clear `ValueError`.
   - Builds a `WorkItem(module, qualname, args, kwargs)`.
   - `ProfileStore.get(module, qualname, profile_key)` returns a snapshot
     `LearnedProfile`: with a `profile_key` it returns the input-bucketed
     profile once that bucket has at least one observation, else the base
     profile (fallback). `estimate()` yields a `ResourceEstimate` (p90 +
     confidence-scaled safety margin, or hints). The `profile_key` is an opaque
     caller-chosen string used to bucket inputs whose resource usage differs
     (e.g. "small"/"large", a file-size band); it is stored on the `PendingWork`
     so result-side recording writes back to the same bucket.
   - `_infeasible_estimate(estimate, snapshot, retry_count=0)` against a snapshot
     (`monitor.current` or a direct `monitor.snapshot()` if not yet populated):
     if the estimate exceeds total capacity minus headroom (`memory_gb >
     memory_total_gb - memory_headroom_gb`, or `vram_gb > largest GPU
     vram_total_gb - vram_headroom_gb`), raise `InfeasibleTaskError` synchronously.
     Only declared when capacity is known (snapshot present with positive memory
     total; for VRAM, snapshot GPUs present). Unknown capacity -> not declared.
   - Appends `PendingWork` to `self.pending`.
2. **_dispatch_loop** (thread): `_check_workers()` then `_maybe_dispatch()`.
   - `_maybe_dispatch()` first calls `_reestimate_pending(now)` (see
     **Dispatch-time re-estimation** below), BEFORE the infeasibility sweep and
     the scheduler, so both see fresh estimates.
   - **Dispatch-time re-estimation.** For each pending task, `_reestimate_pending`
     re-runs `profile.estimate()` from the current profile so a task sharpens as
     its siblings complete. Guarantees: (1) the original `memory_hint`/`vram_hint`
     captured at submit are re-passed every refresh, so a user hint always keeps
     overriding; (2) an entry with `retry_count > 0` is skipped entirely — a
     crash penalty's doubled estimate + exclusive flag must never be weakened;
     (3) throttled — an entry refreshed within `_reestimate_interval_seconds`
     (0.1s) is skipped (the loop ticks at 10ms and `profiles.get` takes a lock +
     copies), and past the interval the estimate is only recomputed when the
     profile's `sample_count` actually changed. Each `PendingWork` tracks
     `memory_hint`/`vram_hint`, `estimate_sample_count`, and `estimate_refreshed_at`.
   - `_maybe_dispatch()` then sweeps the pending queue with
     `_infeasible_estimate(estimate, monitor.current, retry_count)`: any task
     whose estimate can never fit (e.g. crash-retry penalization doubled it
     past capacity) has `InfeasibleTaskError` set on its future (with
     `retry_count` context) and is recorded abandoned; the dispatch thread
     never raises/dies from this. Backfill would route around an infeasible
     task rather than stall on it, but it would then sit queued forever — so
     it must fail instead.
   - `_maybe_dispatch()` then gathers live state via `_build_dispatch_plan()`
     and calls the pure `scheduling.plan_dispatch()` (reservation-based
     backfill). `_build_dispatch_plan()` first calls `_refresh_observations()`
     (see **Observed-usage accounting** below), then computes admittable memory
     (snapshot memory minus used, headroom, and *committed* in-flight
     resources), per-GPU admittable VRAM (via `_committed_vram_per_gpu`), and
     each running task's remaining time (via `_running_remaining_seconds`, from
     `duration_p90_seconds` vs elapsed, using the injected clock). When no
     monitor snapshot exists yet, memory/VRAM are treated as infinite
     (non-gating). The scheduler returns an ordered list of `DispatchDecision`
     (pending id + GPU); the executor executes them, updating `_next_gpu_index`.
     Full semantics are in `docs/features/backfill-scheduling.md`. Admission
     arithmetic (memory headroom against committed estimates, cpu-cores derived
     effective max, per-GPU round-robin VRAM fit) now lives inside the pure
     scheduler.
   - **Cold-start canary identity.** `_build_dispatch_plan` builds each
     `PendingEntry` via `_pending_entry`, which asks the store for
     `estimation_identity(module, qualname, profile_key)` — the store key
     estimation actually draws from (the honored keyed bucket, else the base)
     plus its `sample_count`. The identity is stamped onto the `PendingWork`
     (`profile_identity`) so once dispatched the running task reports the same
     identity to the scheduler. A task is marked `cold` when that identity has
     zero observations AND the caller supplied no `memory_gb`/`vram_gb` hint (an
     explicit hint bypasses the canary — the caller asserted knowledge). A keyed
     submit that falls back to a base profile *with* observations is therefore
     not cold. The pure scheduler enforces "one cold task per identity at a
     time"; see `docs/features/backfill-scheduling.md`.
   - **Observed-usage accounting (avoids admission double-counting).** A running
     task's realized RAM/VRAM already appears in `snapshot.used`; counting its
     full estimate on top of that double-counts and under-utilizes the machine.
     So committed resources credit observed usage: `committed_i =
     max(estimate_i - observed_i, 0)` (pure helpers in
     `adaptive_executor/accounting.py`). `observed_i` is measured **parent-side**
     with no worker-protocol changes: `_capture_baseline` samples the assigned
     worker process's RSS (and pinned-GPU per-process VRAM) at dispatch;
     `_refresh_observations` re-samples at most once per
     `_observation_refresh_seconds` (0.1s) and sets `observed = max(0, current -
     baseline)`. If the process is unreadable (exited/permission) or NVML is
     unavailable, observed falls back to 0 so the full estimate stays committed
     (conservative). Sampling uses narrow, typed excepts
     (`psutil.NoSuchProcess/AccessDenied/ZombieProcess`, NVML errors handled in
     the monitor) and never propagates into the dispatch thread. The sampling
     seams `_sample_rss_bytes` / `_sample_process_vram_gb` are overridable for
     tests. **Reservation release projection:** each `RunningEntry` is fed its
     committed remainder `max(estimate - observed, 0)` as the amount it releases
     when it finishes — *not* the full estimate. The realized portion already
     sits in `snapshot.used` and is deliberately NOT assumed to return, so the
     head's reservation never assumes more frees than the accounting guarantees
     (under-promise = safe, degrades toward FIFO).
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
   (`ProfileStore.record(module, qualname, obs, profile_key=pending.profile_key)`,
   debounced persist — writes both the base and, when a key was carried, the
   keyed profile), and resolves the future.
5. **_check_timeouts** (thread): fails or kills workers for tasks exceeding
   `task_timeout_seconds` per `on_timeout`.
6. **_handle_dead_worker / crash-retry** (thread): when a worker dies with a
   SIGKILL/OOM exit code and the task is retryable (`_should_retry_resource_crash`),
   `_penalize_estimate` is called before re-queuing. It (a) records a persistent
   **memory floor** into the ProfileStore via `record_memory_floor(module,
   qualname, admitted_memory_gb, profile_key=…)` — the estimate the task was
   admitted under, now proven too small, written to the base AND (when present)
   keyed profile — then (b) doubles the retry instance's estimate and marks it
   exclusive. The floor persists so the *next batch of fresh submits* is
   lower-bounded (the crashing run itself produced no observation). Only memory
   is floored: a SIGKILL is a host-RAM OOM event; VRAM exhaustion surfaces as an
   in-process CUDA error, not a SIGKILL.
7. **shutdown**: stops accepting, fails queued futures, drains in-flight, stops
   all workers (including `_retiring`), joins threads, `profiles.flush()`,
   stops the monitor.

## Files and roles
- `adaptive_executor/adaptive_executor.py` — `AdaptiveExecutor`, `WorkerSlot`
  (adds `tasks_completed`), `PendingWork` (carries the optional `profile_key`
  parent-side from submit through `_process_result`; workers never see it). Key
  methods:
  `_get_or_spawn_idle_worker`, `_find_idle_evictable_worker`, `_retire_worker`,
  `_should_recycle`, `_check_workers` (reaps `_retiring`), `_maybe_dispatch`,
  `_reestimate_pending`, `_infeasible_estimate`, `_build_dispatch_plan`,
  `_pending_entry` (canary identity/cold stamping), `_running_remaining_seconds`,
  `_committed_resources`, `_committed_vram_per_gpu`, `_capture_baseline`,
  `_refresh_observations`, `_update_observed`, `_sample_rss_bytes`,
  `_sample_process_vram_gb`, `_worker_process_pids`, `_collect_results`,
  `_penalize_estimate` (records the crash floor).
  `PendingWork` also carries observed-usage fields (`worker_pid`,
  `rss_baseline_bytes`, `vram_baseline_gb`, `observed_memory_gb`,
  `observed_vram_gb`, `observed_refreshed_at`) and learning-loop fields
  (`memory_hint`, `vram_hint`, `estimate_sample_count`, `estimate_refreshed_at`,
  `profile_identity`).
- `adaptive_executor/accounting.py` — pure observed-usage commitment accounting:
  `committed_gb(estimate, observed) = max(estimate - observed, 0)`,
  `total_committed_gb`, `committed_vram_per_gpu`, `ResourceUsage`. Single source
  of truth for both admission committed totals and the reservation release
  projection. Unit-tested in `tests/test_accounting.py`.
- `adaptive_executor/scheduling.py` — pure reservation-based backfill scheduler
  (`plan_dispatch`); see `docs/features/backfill-scheduling.md`.
- `adaptive_executor/errors.py` — `InfeasibleTaskError` (structured fields:
  `kind` in {"memory","vram"}, `estimate_gb`, `capacity_gb`, `retry_count`).
- `adaptive_executor/worker.py` — `Worker`, `worker_process_entry`. Key:
  `_gpu_vram_gb` (pinned-only, per-process preferred), `_process_tree_pids`,
  `_execute_with_observation`.
- `adaptive_executor/monitor.py` — `ResourceMonitor`. Key exports:
  `snapshot`, `current`, `device_vram_used_gb`, `per_process_vram_gb`,
  `_compute_running_processes` (v3/v2/unversioned fallback).
- `adaptive_executor/profiles.py` — `LearnedProfile`, `ProfileStore`, and the
  pure `derive_store_key(module, qualname, profile_key=None)` (the single place
  store keys are built: base `module:qualname`, keyed
  `module:qualname#profile_key`, `STORE_KEY_SEPARATOR = "#"`). Key methods:
  `get` / `_select_profile_locked` (keyed-with-fallback selection),
  `estimation_identity` (the identity + sample_count the canary keys off),
  `record` (dual base+keyed write), `record_memory_floor` (dual base+keyed floor
  write), `flush`, `_should_save_locked`, `_snapshot_locked`,
  `_persist_snapshot`, `_write_atomic`, `_load`. `LearnedProfile` carries
  `memory_floor_gb` (lower-bounds `estimate()` after the safety margin, ratchets
  up via `set_memory_floor`, cleared by `_maybe_recover_floor` after
  `FLOOR_RECOVERY_OBSERVATIONS` peaks below `FLOOR_RECOVERY_FRACTION` of it).
  Floor-less profiles persist as a plain observation list (backward compatible);
  a floored profile persists as `{"observations": [...], "memory_floor_gb": X}`,
  and `_load` accepts both.
- `adaptive_executor/resolve.py` — `resolve_function`, `validate_submittable`,
  `FunctionResolutionError`.
- `adaptive_executor/dtypes.py` — dataclasses.

## Invariants and constraints
- Infeasibility is a permanent condition (estimate > total capacity minus
  headroom), distinct from "doesn't fit right now" (normal queuing/backpressure).
  It is only ever declared when capacity is known; unknown capacity leaves
  admission behavior unchanged. It is raised synchronously from `submit()` and
  attached to the future (never raised) in the dispatch thread, which keeps
  running and moves on to the next pending task.
- A `WorkerSlot` in `self.workers` is alive or about to be reaped by
  `_check_workers`; retired workers live only in `self._retiring` until joined.
- Workers are pinned to one GPU (NVML index) for their lifetime; a pin change
  requires evicting the worker and spawning a replacement.
- Committed resources always cover in-flight tasks; admission uses
  committed + new estimate against available resources minus headroom. Committed
  credits observed (realized) usage — `committed = max(estimate - observed, 0)`
  — so an allocation that already shows in `snapshot.used` is not counted twice.
  Committed is monotone in observed only downward: it can never exceed the
  estimate, so the headroom invariant is preserved (observed credit only ever
  frees admission capacity, never grants more than the estimate reserved).
- Observed usage is measured parent-side (RSS + per-process VRAM) relative to a
  dispatch-time baseline, cached per `_observation_refresh_seconds`, and falls
  back to 0 (full estimate committed) whenever the worker process or NVML is
  unreadable. A finishing task is projected to release only its committed
  remainder to the reservation (never the already-realized portion), keeping the
  head's reservation conservative.
- Only an idle worker (`current_work_id is None`, alive, not
  `intentionally_stopped`) may be reused, evicted, or recycled.
- VRAM is attributed to the worker's pinned device only; an unpinned worker
  records `vram_delta_gb == 0.0` and performs no GPU polling.
- `_retire_worker`/recycle/evict mutate `self.workers` and `self._retiring` while
  holding `self.lock`.
- Persistence is debounced (`save_every_n` observations or
  `save_interval_seconds`); serialization happens under the store lock, file I/O
  outside it under a separate save lock; `flush()` (and `shutdown`) force a save.
- Input-aware profiles: every observation is recorded into the base profile
  (`module:qualname`), so the base is always the full aggregate and remains a
  valid fallback; the keyed profile (`module:qualname#profile_key`) is written
  only when a `profile_key` was supplied. Estimation uses the keyed profile only
  when it has ≥1 observation, else the base — and the fallback read never
  materializes an empty keyed entry. Store keys are strings, so keyed profiles
  persist and reload with no special handling. The submit/dispatch feasibility
  checks operate on whatever `profile.estimate()` returns, so choosing the keyed
  profile before estimating is all that makes them input-aware.
- Profile save/load failures are logged (warning/error), never raised into the
  record path; a corrupt file yields an empty store, not a crash.
- Learning loop:
  - Re-estimation never weakens a crash penalty (`retry_count > 0` skipped) and a
    user hint always overrides the learned estimate (hints re-passed each
    refresh). Re-estimation is throttled per entry and only recomputes when the
    profile's `sample_count` changed.
  - The cold-start canary is a scheduler invariant (one cold task per identity in
    flight; hints and warm profiles bypass it) — enforced purely, so it is
    deterministic and unit-testable. The identity a running task holds is the one
    it was estimated under (stamped on `PendingWork` before dispatch).
  - A crash floor only ratchets up while active and is written to both the base
    and keyed profile. It lower-bounds memory *after* the safety margin (and even
    over an optimistic hint), and is cleared only by the recovery rule so a
    genuinely shrunk workload is not penalized forever. Floors persist across
    restarts; files predating floors still load (no floor).
