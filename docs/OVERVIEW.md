Overview:
  description: >
    AdaptiveExecutor is a resource-aware parallel executor for Python. It runs
    submitted functions in worker subprocesses, measures each run's RAM/VRAM/CPU
    usage, learns per-function resource profiles, and uses those profiles for
    admission control so concurrency is throttled to avoid out-of-memory
    conditions. GPU workloads are round-robin assigned across NVML devices and
    each worker is pinned to one GPU for its lifetime.
  subsystems:
    executor:
      module: adaptive_executor/adaptive_executor.py
      role: >
        Public API (submit/shutdown), background dispatch/result/timeout
        threads, worker pool lifecycle (spawn/reuse/evict/recycle/reap),
        admission control, GPU round-robin, crash retry. Gathers live state each
        dispatch cycle and delegates the ordering decision to the scheduling
        subsystem.
    scheduling:
      module: adaptive_executor/scheduling.py
      role: >
        Pure, deterministic EASY (reservation-based) backfill scheduler. Given
        FIFO pending entries (with resource estimates + p90 durations), running
        entries (with remaining time), and per-GPU capacity/headroom, it returns
        the set of tasks to dispatch this cycle so that later tasks may backfill
        past a blocked head only when they cannot delay the head's reservation.
        No threads, no I/O, no executor state.
    worker:
      module: adaptive_executor/worker.py
      role: >
        Subprocess entry point. Resolves the function by (module, qualname),
        executes it, and measures RSS/VRAM/CPU for the run.
    monitor:
      module: adaptive_executor/monitor.py
      role: >
        psutil + optional NVML polling. Provides system snapshots and
        pinned-device / per-process VRAM measurement helpers.
    profiles:
      module: adaptive_executor/profiles.py
      role: >
        LearnedProfile (p90 estimates + safety margin) and ProfileStore
        (thread-safe, debounced JSON persistence). Profiles are keyed by a
        derived store key from the pure ``derive_store_key(module, qualname,
        profile_key=None)`` — the base profile at ``module:qualname`` and,
        when a caller supplies an opaque ``profile_key``, an input-bucketed
        profile at ``module:qualname#profile_key``. Every observation is
        recorded into both the base and (when given) the keyed profile;
        estimation prefers the keyed profile once it has any observation and
        otherwise falls back to the base aggregate.
    resolve:
      module: adaptive_executor/resolve.py
      role: >
        Shared (module, qualname) -> callable resolution used by both the worker
        and submit-time validation.
    errors:
      module: adaptive_executor/errors.py
      role: >
        Typed, structured executor exceptions. Currently InfeasibleTaskError,
        raised/attached when a task's estimate can never fit on this machine
        (carries kind, estimate_gb, capacity_gb, retry_count).
    dtypes:
      module: adaptive_executor/dtypes.py
      role: Dataclasses for snapshots, observations, estimates, work items/results.
    scheduler_sim:
      module: tests/sim/
      role: >
        Test-only deterministic discrete-event harness. Drives the executor's
        real dispatch/admission methods (_maybe_dispatch, _can_admit,
        _pick_round_robin_gpu, committed-resource accounting, _handle_dead_worker,
        _process_result) against a virtual clock and synthetic workloads -- no
        subprocesses, no NVML, no wall-clock sleeps -- and asserts scheduling
        invariants over a recorded virtual-time trace. See
        docs/features/scheduler-sim.md.
  data_flow: >
    submit() validates the callable is importable, looks up its LearnedProfile
    (the input-bucketed one when a profile_key is passed, else the base), and
    computes a ResourceEstimate (p90 memory/VRAM/CPU plus a p90 run duration). It then checks feasibility against known capacity (total minus
    headroom) and raises InfeasibleTaskError synchronously if the estimate can
    never fit, so an impossible task fails at the call site instead of silently
    queuing forever. A background dispatch thread first re-checks feasibility
    across the pending queue (an estimate can become infeasible after
    crash-retry penalization doubles it) and sets InfeasibleTaskError on any
    such task's future without killing the thread. It then gathers live state
    each cycle (monitor snapshot, committed in-flight estimates, per-GPU VRAM,
    running tasks' elapsed-vs-expected times) into the scheduling subsystem's
    input dataclasses and calls plan_dispatch(). The scheduler returns an
    ordered set of dispatch decisions (which pending tasks to start now and on
    which GPU): front tasks are admitted in strict FIFO order while they fit,
    and when the head is blocked, later tasks may backfill only if they cannot
    delay the head's reservation. For each decision the executor obtains an idle
    worker pinned to the required GPU (spawning or evicting+replacing as
    necessary) and sends the WorkItem over that worker's queue. The worker
    executes, measures usage, and returns a WorkResult on the shared result
    queue. A result thread resolves the future and records the
    ResourceObservation (including duration_seconds) into the ProfileStore under
    both the base and (when the task carried a profile_key) the keyed profile,
    which persists to disk in a debounced fashion. Workers are recycled after a
    configurable number of tasks and reaped without leaving zombies.

Features Index:
  executor:
    description: >
      Resource-aware submission, admission control, worker-pool management,
      resource learning, and persistence.
    entry_points:
      - adaptive_executor.AdaptiveExecutor.submit
      - adaptive_executor.AdaptiveExecutor.shutdown
      - adaptive_executor.AdaptiveExecutor.start
    depends_on:
      - worker
      - monitor
      - profiles
      - resolve
      - backfill_scheduling
      - errors
    doc: docs/features/executor.md
  backfill_scheduling:
    description: >
      Reservation-based EASY backfill: when the head of the queue is blocked,
      later tasks may be dispatched ahead of it only when they cannot delay the
      head's resource reservation. Preserves the "never later than FIFO"
      invariant and per-GPU VRAM accounting; an exclusive head blocks all
      backfill.
    entry_points:
      - adaptive_executor.scheduling.plan_dispatch
      - adaptive_executor.AdaptiveExecutor._build_dispatch_plan
      - adaptive_executor.AdaptiveExecutor._maybe_dispatch
    depends_on:
      - executor
      - profiles
    doc: docs/features/backfill-scheduling.md
  scheduler_sim:
    description: >
      Deterministic discrete-event simulation of the scheduler for
      property-style tests (admission/headroom, committed accounting,
      head-not-delayed-vs-FIFO, head-of-line, OOM-retry storms, no-lost-tasks).
      Policy under test is a parameter.
    entry_points:
      - tests/sim/harness.py::SchedulerSim.run
      - tests/sim/harness.py::run_to_quiescence
    depends_on:
      - executor
    doc: docs/features/scheduler-sim.md

Testing:
  note: >
    Beyond the wall-clock concurrency tests, tests/sim/ hosts a virtual-clock
    discrete-event harness that drives the real scheduling logic deterministically.
    To make fakes substitutable without changing production behavior, the
    executor exposes three dependency-injection seams (all defaulting to today's
    behavior): a ``clock`` time source (defaults to time.time), an injectable
    ``monitor`` (defaults to ResourceMonitor()), and worker spawning via the
    overridable ``_spawn_worker``. The per-result handling was extracted into
    ``_process_result`` so a single result can be driven through the real path.
