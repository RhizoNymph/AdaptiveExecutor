"""Deterministic discrete-event simulation harness for the executor scheduler.

This package drives the *real* admission/dispatch logic of
``adaptive_executor.AdaptiveExecutor`` against a virtual clock and synthetic
workloads, with no worker subprocesses, no NVML, and no wall-clock sleeps. It
exists so the race-prone scheduling paths (admission control, FIFO
head-of-line behavior, committed-resource accounting, OOM-retry storms) can be
asserted deterministically under plain pytest.

Public surface:
- ``harness``    -- the simulation engine (``SchedulerSim``, ``VirtualClock``,
                    ``SimMonitor``, ``TraceEvent``, ``run_to_quiescence``).
- ``workloads``  -- synthetic workload data types and deterministic generators.
- ``task_stubs`` -- importable no-op callables used as submit targets.
"""
