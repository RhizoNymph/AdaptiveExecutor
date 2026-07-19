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
- Future cancellation: queued tasks are cancellable via the standard
  `concurrent.futures.Future.cancel()`; running tasks are not.

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
   - `_maybe_dispatch()` first sweeps the pending queue with
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
   - **Running handshake / cancellation.** Once a worker is secured and
     immediately before shipping, the executor calls the future's
     `set_running_or_notify_cancel()` (exactly once per future — tracked by
     `PendingWork.running_notified`, since a crash-retry re-queues an already-
     RUNNING future). If it returns `False` the caller had cancelled the queued
     task: it is dropped from `self.pending` and recorded in
     `completed_or_abandoned`, never dispatched, and no worker state is touched
     (the secured worker stays idle for the next decision). If it returns `True`
     the future is now `RUNNING`, so `cancel()` returns `False` from here on and
     the in-flight accounting can never be cancelled out from under the
     executor. This is the atomic queued/running boundary: a `cancel()` racing
     dispatch has exactly one winner.
   - On success: move to `in_flight`, mark worker busy, send the `WorkItem`.
     Dispatched ids (and cancelled ones) are removed from `self.pending` (which
     may be non-contiguous when tasks backfilled past a blocked head).
   - **Cancellation-safe future settling.** Every `set_result`/`set_exception`
     site routes through `_settle_future_result` / `_settle_future_exception`,
     which skip a future that is already `done()` (cancelled or resolved by
     timeout/crash) and, for the narrow race where a caller cancels between the
     `done()` check and the set, catch `InvalidStateError` with a structured
     log. A done/cancelled future can therefore never raise `InvalidStateError`
     into a background thread. The infeasibility sweep and `_fail_pending_futures`
     skip cancelled entries the same way (dropping them as abandoned rather than
     stuffing an exception into a cancelled future).
3. **worker** (subprocess): resolves fn via shared `resolve_function`, runs it,
   samples RSS and (only when pinned) pinned-GPU VRAM, returns a `WorkResult`.
4. **_collect_results** (thread): **per-worker result queues.** Each worker owns
   its own result queue (`WorkerSlot.result_queue`, created in `_spawn_worker`
   and passed to `worker_process_entry`); there is deliberately no shared result
   queue. The collector loops `_sweep_result_queues()`, which snapshots
   `self.workers` under the lock and non-blocking-drains each live worker's queue
   via `_read_queue` (`get_nowait()` until `Empty`), sleeping 10ms only when a
   sweep found nothing. A *poisoned* queue (its worker was intentionally killed
   and may have corrupted its own feeder state) is never read. Each collected
   `WorkResult` goes through the unchanged `_process_result`: clears the worker's
   `current_work_id`, increments `tasks_completed`, recycles the worker if it hit
   `worker_recycle_after_tasks`, pops `in_flight`, records the observation
   (`ProfileStore.record(module, qualname, obs, profile_key=pending.profile_key)`,
   debounced persist — writes both the base and, when a key was carried, the
   keyed profile), and resolves the future.
   - **Why per-worker queues.** With one shared `mp.Queue`, a SIGTERM/SIGKILL
     landing while any worker was mid-`put` was a documented hazard: the queue's
     shared feeder state could be left wedged, silently breaking result delivery
     for *every* worker thereafter. Per-worker queues bound the blast radius of
     killing a worker to that worker's own queue.
   - **Queue lifecycle (drain vs. discard).** `_drain_queue(slot)` processes any
     final results a *gracefully exited* worker left behind (only safe after the
     process has exited, so `get_nowait()` reaches `Empty` and cannot block);
     `_discard_queue(slot)` closes the queue and cancels its join thread
     (idempotent via `queue_discarded`) so a discard can never block the parent
     on GC/exit. A poisoned queue is discarded but **never** drained.
5. **_check_timeouts** / **_handle_timeout** (thread): fails or kills workers for
   tasks exceeding `task_timeout_seconds` per `on_timeout`.
   - `on_timeout="kill_worker"`: under the lock the worker is marked
     `intentionally_stopped` + `queue_poisoned`, its `current_work_id` cleared,
     removed from `self.workers` and parked in `self._retiring`, and the task
     popped from `in_flight`. Poisoning under the lock guarantees the collector
     never reads that queue again. Outside the lock, `_escalate_kill` sends
     `terminate()` (SIGTERM), waits `_kill_grace_seconds` (0.5s), and escalates to
     `kill()` (SIGKILL) if the worker is still alive (the worker installs no
     SIGTERM handler, so default SIGTERM normally kills it immediately). The
     task's future is failed with `TimeoutError`. `_check_workers` later reaps
     the retired worker: join + `_discard_queue` (poisoned ⇒ no drain).
   - `on_timeout="fail_future"`: unchanged — mark `result_ignored`, pop
     `in_flight`, clear the worker's `current_work_id`, fail the future.
6. **shutdown**: stops accepting, fails queued futures, drains in-flight, stops
   all workers (including `_retiring`) — `_stop_all_workers` sends the stop
   sentinel, joins, then **drains** each non-poisoned worker's queue (so a final
   in-flight result is still delivered) and **discards** every queue — joins
   threads, `profiles.flush()`, stops the monitor.

## Files and roles
- `adaptive_executor/adaptive_executor.py` — `AdaptiveExecutor`, `WorkerSlot`
  (adds `tasks_completed`, and for kill isolation: `result_queue` (per-worker),
  `queue_poisoned`, `queue_discarded`), `PendingWork` (carries the optional
  `profile_key` parent-side from submit through `_process_result`; workers never
  see it). Key methods:
  `_get_or_spawn_idle_worker`, `_find_idle_evictable_worker`, `_retire_worker`,
  `_should_recycle`, `_check_workers` (reaps `_retiring`, drain-then-discard each
  reaped queue), `_handle_dead_worker` (drain-then-discard a crashed worker's
  queue before crash handling), `_maybe_dispatch`,
  `_infeasible_estimate`, `_build_dispatch_plan`, `_running_remaining_seconds`,
  `_committed_resources`, `_committed_vram_per_gpu`, `_capture_baseline`,
  `_refresh_observations`, `_update_observed`, `_sample_rss_bytes`,
  `_sample_process_vram_gb`, `_worker_process_pids`, `_collect_results`,
  `_sweep_result_queues`, `_read_queue`, `_drain_queue`, `_discard_queue`,
  `_handle_timeout`, `_escalate_kill`, `_stop_all_workers`,
  `_settle_future_result`, `_settle_future_exception` (cancellation-safe future
  settling). `PendingWork` also carries `running_notified` (the once-only
  running-handshake flag).
  `PendingWork` also carries observed-usage fields (`worker_pid`,
  `rss_baseline_bytes`, `vram_baseline_gb`, `observed_memory_gb`,
  `observed_vram_gb`, `observed_refreshed_at`).
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
  `get` / `_select_profile_locked` (keyed-with-fallback selection), `record`
  (dual base+keyed write), `flush`, `_should_save_locked`, `_snapshot_locked`,
  `_persist_snapshot`, `_write_atomic`, `_load`.
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
- **Result-queue isolation.** Each `WorkerSlot` owns its `result_queue`
  (production always populates it in `_spawn_worker`; the `None` default exists
  only so test fakes may omit it). The collector reads a worker's queue only
  while the worker is live and non-poisoned. A queue is drained (final results
  processed) exactly once, at reap, and only for a gracefully-exited or crashed
  worker whose process has already exited; a poisoned (intentionally-killed)
  queue is discarded without ever being read. `_discard_queue` is idempotent
  (`queue_discarded`) and always closes + cancels the join thread, so a discard
  never blocks the parent. Consequence: killing one worker can corrupt at most
  its own queue, never result delivery for any other worker.
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
- Cancellation is a queued-only operation. A future in `self.pending` is always
  `PENDING` or `CANCELLED` (never yet handed to a worker), so the dispatch-time
  `set_running_or_notify_cancel()` handshake can safely decide between them. Once
  a task is dispatched its future is `RUNNING` and `cancel()` returns `False`;
  crash-retry re-queues an already-`RUNNING` future, so the handshake runs at
  most once per future (`running_notified`) and a retrying task is likewise not
  cancellable. A cancelled task frees its queue slot, is recorded in
  `completed_or_abandoned`, and is never executed nor given a result/exception.
- No `set_result`/`set_exception` on a done/cancelled future may reach a bare
  call: all sites go through `_settle_future_*`, which guard with `done()` plus a
  narrow `InvalidStateError` catch, so a background thread can never die from
  resolving a future the caller already cancelled or that was already resolved.
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
