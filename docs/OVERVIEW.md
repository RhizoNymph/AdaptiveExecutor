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
        (thread-safe, debounced JSON persistence).
    resolve:
      module: adaptive_executor/resolve.py
      role: >
        Shared (module, qualname) -> callable resolution used by both the worker
        and submit-time validation.
    dtypes:
      module: adaptive_executor/dtypes.py
      role: Dataclasses for snapshots, observations, estimates, work items/results.
  data_flow: >
    submit() validates the callable is importable, looks up its LearnedProfile,
    and computes a ResourceEstimate (p90 memory/VRAM/CPU plus a p90 run
    duration). A background dispatch thread gathers live state each cycle
    (monitor snapshot, committed in-flight estimates, per-GPU VRAM, running
    tasks' elapsed-vs-expected times) into the scheduling subsystem's input
    dataclasses and calls plan_dispatch(). The scheduler returns an ordered set
    of dispatch decisions (which pending tasks to start now and on which GPU):
    front tasks are admitted in strict FIFO order while they fit, and when the
    head is blocked, later tasks may backfill only if they cannot delay the
    head's reservation. For each decision the executor obtains an idle worker
    pinned to the required GPU (spawning or evicting+replacing as necessary) and
    sends the WorkItem over that worker's queue. The worker executes, measures
    usage, and returns a WorkResult on the shared result queue. A result thread
    resolves the future and records the ResourceObservation (including
    duration_seconds) into the ProfileStore, which persists to disk in a
    debounced fashion. Workers are recycled after a configurable number of tasks
    and reaped without leaving zombies.

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
