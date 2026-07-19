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
        admission control, GPU round-robin, crash retry.
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
        (thread-safe, debounced JSON persistence).
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
    submit() validates the callable is importable, looks up its LearnedProfile,
    and computes a ResourceEstimate. It then checks feasibility against known
    capacity (total minus headroom) and raises InfeasibleTaskError synchronously
    if the estimate can never fit, so an impossible task fails at the call site
    instead of blocking the FIFO head forever. A background dispatch thread
    re-checks feasibility on the head task (an estimate can become infeasible
    after crash-retry penalization doubles it) and, if infeasible, pops it and
    sets InfeasibleTaskError on its future without killing the thread. Otherwise
    it checks admission (_can_admit) against the latest monitor snapshot plus
    committed
    in-flight estimates, picks a GPU when needed, obtains an idle worker pinned
    to the required GPU (spawning or evicting+replacing as necessary), and sends
    the WorkItem over that worker's queue. The worker executes, measures usage,
    and returns a WorkResult on the shared result queue. A result thread resolves
    the future and records the ResourceObservation into the ProfileStore, which
    persists to disk in a debounced fashion. Workers are recycled after a
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
      - errors
    doc: docs/features/executor.md
  scheduler_sim:
    description: >
      Deterministic discrete-event simulation of the scheduler for
      property-style tests (admission/headroom, committed accounting, FIFO,
      head-of-line, OOM-retry storms, no-lost-tasks). Policy under test is a
      parameter so a future backfill scheduler can reuse the harness.
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
