# Feature: Scheduler Simulation Harness (test-only)

## Scope
- A deterministic **discrete-event** simulation of the executor's scheduling and
  admission behavior, runnable under plain `pytest`.
- Drives the executor's **real** dispatch/admission code against a **virtual
  clock** and **synthetic workloads**: `_maybe_dispatch`, `_can_admit`,
  `_pick_round_robin_gpu`, `_committed_resources` / `_committed_vram_per_gpu`,
  `_get_or_spawn_idle_worker`, `_check_workers` / `_handle_dead_worker`,
  `_process_result`.
- Synthetic workloads: mixed memory/VRAM sizes and durations, deliberately wrong
  estimates (actual above/below the hint), no-hint cold starts, and OOM-kill
  retry storms. Deterministic seeds only.
- A trace/event log (dispatch/complete/crash/retry/fail with virtual timestamps
  and committed-resource state) is the core artifact; property tests assert over
  it.
- The scheduling policy under test is a parameter (`dispatch` / `check_workers`
  overrides on `SchedulerSim`) so a future backfill scheduler can reuse the
  harness.

## Non-scope
- No real worker subprocesses, NVML, threads, or wall-clock sleeps.
- Does not simulate the timeout path (`_check_timeouts`); `task_timeout_seconds`
  is set effectively infinite.
- Does not exercise profile persistence to disk (no `profile_path`).
- Not a reimplementation of the scheduler: it only *drives* the real methods.

## Executor seams relied upon (behavior-preserving)
Added to `adaptive_executor/adaptive_executor.py`; all defaults reproduce
today's production behavior exactly (verified by the unchanged 48-test suite):
- `clock: Callable[[], float] = time.time` -> stored as `self._clock`. The three
  timestamp reads (`submit` `submitted_at`, `_maybe_dispatch` `started_at`,
  `_check_timeouts` `now`) now use `self._clock()`.
- `monitor: ResourceMonitor | None = None` -> used if provided, else
  `ResourceMonitor()` as before.
- `_process_result(result)` extracted verbatim from the `_collect_results` loop
  body so a single `WorkResult` can be handled through the real path.
- Worker spawning is overridden through the already-existing `_spawn_worker`
  seam (same pattern as `tests/test_executor_pool.py`).

## Data / control flow
1. `SchedulerSim.__init__` builds a real `AdaptiveExecutor` with the injected
   `SimMonitor` (a static capacity snapshot) and `VirtualClock.now` as the clock,
   sets `_started = True` (so `submit` never launches threads), and installs a
   fake `_spawn_worker` producing `WorkerSlot`s backed by `FakeProcess` /
   `FakeQueue` from `tests/fakes.py`.
2. `submit_all()` calls the real `submit(...)` for each `TaskSpec` (passing its
   estimate hints), captures the generated work id / future, and records a
   `submit` trace event.
3. `run()` loops: `_check_workers()` -> `_detect_retries()` -> `_dispatch()`
   (the real `_maybe_dispatch`) -> `_detect_dispatches()`. If quiescent it
   returns; otherwise it pops the earliest scheduled completion, advances the
   virtual clock to it, and fires it.
   - `_detect_dispatches` finds tasks newly in `in_flight`, records a `dispatch`
     event, and schedules a `_Completion` at `now + duration`.
   - `_detect_retries` records a `retry` event when a crashed task reappears in
     `pending` with an incremented `retry_count` (estimate already doubled).
   - Firing a completion either **finishes** the task (builds a synthetic
     `ResourceObservation` + `WorkResult` and calls `_process_result`, resolving
     the future through the real path) or **crashes** it (marks the `FakeProcess`
     dead with `-SIGKILL`, letting `_check_workers` -> `_handle_dead_worker`
     apply the real penalize+exclusive-retry or final-failure logic).
4. Quiescence = no `pending`, no `in_flight`, no scheduled events. A step budget
   guards non-termination; remaining-but-stuck work raises `SimStall`.

## Files and roles
- `tests/sim/__init__.py` — package marker / overview.
- `tests/sim/task_stubs.py` — importable module-level no-op callables used as
  `submit` targets so `validate_submittable` succeeds and task classes get
  distinct profile keys.
- `tests/sim/workloads.py` — `TaskSpec`, `SimGpu`, `SimCapacity`, `Workload`
  dataclasses; fixed scenario builders (`cpu_fifo_workload`,
  `head_of_line_workload`, `oom_retry_workload`, `oom_exhausted_workload`,
  `gpu_roundrobin_workload`) and the seeded `random_workload(seed, n)`.
- `tests/sim/harness.py` — `VirtualClock`, `SimMonitor`, `TraceEvent`,
  `SchedulerSim`, `run_to_quiescence`, and `SimError`/`SimStall`/`SimStepLimit`.
  Reuses `FakeProcess` / `FakeQueue` from `tests/fakes.py`.
- `tests/sim/test_scheduler_properties.py` — the property tests.

## Properties asserted
- **Headroom**: at every trace event, committed memory <= usable memory and
  per-GPU committed VRAM <= usable VRAM (usable = total - baseline - headroom).
- **Committed -> zero**: after quiescence, `in_flight`/`pending` empty and all
  committed totals are zero.
- **FIFO**: first-attempt dispatch order equals submission order.
- **Head-of-line**: a large feasible head dispatches only after the running
  mediums release, and tasks behind it do not jump ahead.
- **OOM-retry**: a SIGKILL crash yields a doubled estimate + exclusive retry
  that dispatches alone and then succeeds; an over-budget crash resolves the
  future as a crash `RuntimeError` (never lost).
- **No lost tasks**: every submitted future resolves.
- **GPU round-robin** spreads VRAM tasks across all devices.
- **Determinism**: equal seeds produce identical traces.

## Invariants and constraints
- The harness never mutates executor scheduling logic; it only calls real
  methods and injects results/crashes at the boundaries.
- Capacity is a static snapshot; in-flight load is modeled solely via the
  executor's committed estimates (matching production admission).
- All randomness is seeded; generated tasks (and OOM retries) are kept feasible
  so the current head-of-line scheduler cannot deadlock — an infeasible task
  would surface as `SimStall`.
- Virtual time is monotonic; completion events are ordered by `(time, seq)` for
  a fully reproducible schedule; superseded completions (after a re-dispatch)
  are ignored via an attempt counter.
