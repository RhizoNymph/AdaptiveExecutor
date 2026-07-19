"""Property-style tests over the deterministic scheduler simulation.

Each test builds a synthetic workload, runs the simulation to quiescence, and
asserts an invariant over the recorded virtual-time trace. No wall clock, no
subprocesses, no unseeded randomness.
"""

import pytest

from sim.harness import SchedulerSim, run_to_quiescence
from sim.workloads import (
    cpu_fifo_workload,
    gpu_roundrobin_workload,
    head_of_line_workload,
    oom_exhausted_workload,
    oom_retry_workload,
    random_workload,
)

_EPS = 1e-9

_RANDOM_SEEDS = [0, 1, 2, 7, 42, 123, 2024]

_ALL_FIXED = [
    cpu_fifo_workload,
    head_of_line_workload,
    oom_retry_workload,
    gpu_roundrobin_workload,
]


def _all_workloads():
    for factory in _ALL_FIXED:
        yield factory()
    for seed in _RANDOM_SEEDS:
        yield random_workload(seed)


# --------------------------------------------------------------------------- #
# Admission invariant: committed estimates never exceed usable capacity.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "workload",
    list(_all_workloads()),
    ids=lambda w: w.label,
)
def test_committed_never_exceeds_capacity(workload):
    sim = SchedulerSim(workload)
    trace = sim.run()
    cap = workload.capacity
    usable_mem = cap.usable_memory_gb()

    assert trace, "expected a non-empty trace"
    for event in trace:
        assert event.committed_memory_gb <= usable_mem + _EPS, (
            f"memory over capacity at {event.kind} t={event.time}: "
            f"{event.committed_memory_gb} > {usable_mem}"
        )
        for gpu_id, committed_vram in event.committed_vram_per_gpu:
            usable_vram = cap.usable_vram_gb(gpu_id)
            assert committed_vram <= usable_vram + _EPS, (
                f"vram over capacity on gpu {gpu_id} at {event.kind} t={event.time}: "
                f"{committed_vram} > {usable_vram}"
            )


# --------------------------------------------------------------------------- #
# Committed accounting returns to zero once everything completes.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "workload",
    list(_all_workloads()),
    ids=lambda w: w.label,
)
def test_committed_returns_to_zero(workload):
    sim = run_to_quiescence(workload)
    mem, per_gpu = sim.committed_now()

    assert not sim.executor.pending
    assert not sim.executor.in_flight
    assert mem == pytest.approx(0.0, abs=_EPS)
    assert all(v == pytest.approx(0.0, abs=_EPS) for v in per_gpu.values())


# --------------------------------------------------------------------------- #
# Every submitted future resolves (result or exception) -- no lost tasks.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "workload",
    list(_all_workloads()),
    ids=lambda w: w.label,
)
def test_all_futures_resolve(workload):
    sim = run_to_quiescence(workload)
    futures = sim.futures()

    assert len(futures) == len(workload.tasks)
    for work_id, future in futures.items():
        assert future.done(), f"future for {work_id} never resolved"
        # .exception() would raise if still pending; done() above guarantees it.
        exc = future.exception()
        if exc is None:
            future.result()  # must not raise for a successful task


# --------------------------------------------------------------------------- #
# FIFO: under the current scheduler, dispatch order == submission order.
# --------------------------------------------------------------------------- #


def test_fifo_dispatch_order():
    workload = cpu_fifo_workload()
    sim = SchedulerSim(workload)
    sim.run()

    first_dispatches = [e.work_id for e in sim.dispatch_events(attempt=1)]
    assert first_dispatches == sim.submit_order


# --------------------------------------------------------------------------- #
# Head-of-line: a large feasible head dispatches once running tasks release.
# --------------------------------------------------------------------------- #


def test_head_of_line_large_task_eventually_dispatches():
    workload = head_of_line_workload()
    sim = SchedulerSim(workload)
    sim.run()

    large_dispatch = next(
        e for e in sim.events_for("large-head") if e.kind == "dispatch"
    )
    # It could not dispatch at t=0 (blocked by the mediums); it dispatches only
    # after they release their committed memory.
    assert large_dispatch.time > 0.0

    medium_completions = [
        e for e in sim.trace if e.task_name.startswith("medium-") and e.kind == "complete"
    ]
    assert len(medium_completions) == 3
    assert all(c.time <= large_dispatch.time for c in medium_completions)

    # Head-of-line blocking: nothing submitted after the large head dispatches
    # before it does.
    small_dispatches = [
        e for e in sim.trace if e.task_name.startswith("small-") and e.kind == "dispatch"
    ]
    assert small_dispatches
    assert all(s.time >= large_dispatch.time for s in small_dispatches)

    # And the head actually completed.
    large_complete = next(e for e in sim.events_for("large-head") if e.kind == "complete")
    assert large_complete.time >= large_dispatch.time


# --------------------------------------------------------------------------- #
# OOM-retry: a SIGKILL crash -> doubled estimate + exclusive retry, dispatched
# alone once the concurrent task releases resources.
# --------------------------------------------------------------------------- #


def test_oom_crash_doubles_estimate_and_retries_exclusively():
    workload = oom_retry_workload()
    sim = SchedulerSim(workload, max_resource_crash_retries=1)
    sim.run()

    flaky = sim.events_for("flaky")
    kinds = [e.kind for e in flaky]
    assert "crash" in kinds
    assert "retry" in kinds

    first_dispatch = next(e for e in flaky if e.kind == "dispatch" and e.attempt == 1)
    retry_event = next(e for e in flaky if e.kind == "retry")
    # Estimate doubled by the penalize path.
    assert retry_event.est_memory_gb == pytest.approx(first_dispatch.est_memory_gb * 2.0)
    assert "exclusive=True" in retry_event.detail

    # The exclusive retry dispatches alone (nothing else in flight).
    second_dispatch = next(e for e in flaky if e.kind == "dispatch" and e.attempt == 2)
    assert second_dispatch.in_flight_count == 1
    assert second_dispatch.time > first_dispatch.time

    # It ultimately succeeds.
    future = sim.futures()[second_dispatch.work_id]
    assert future.done()
    assert future.result() == "flaky"


def test_oom_exhausted_retries_fails_future_without_loss():
    workload = oom_exhausted_workload()
    sim = SchedulerSim(workload, max_resource_crash_retries=1)
    sim.run()

    doomed = sim.events_for("doomed")
    crash_events = [e for e in doomed if e.kind == "crash"]
    # One initial crash plus one crash on the (single) allowed retry.
    assert len(crash_events) == 2

    (work_id,) = sim.futures().keys()
    future = sim.futures()[work_id]
    assert future.done()
    exc = future.exception()
    assert isinstance(exc, RuntimeError)
    assert "crashed" in str(exc)


# --------------------------------------------------------------------------- #
# GPU round-robin spreads VRAM tasks across all devices.
# --------------------------------------------------------------------------- #


def test_gpu_tasks_spread_across_devices():
    workload = gpu_roundrobin_workload()
    sim = SchedulerSim(workload)
    sim.run()

    used_gpus = {
        e.gpu_id for e in sim.trace if e.kind == "dispatch" and e.gpu_id is not None
    }
    assert used_gpus == {0, 1}


# --------------------------------------------------------------------------- #
# Determinism: identical seeds produce identical traces.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", _RANDOM_SEEDS)
def test_simulation_is_deterministic(seed):
    def trace_signature(sim):
        return [
            (e.time, e.kind, e.task_name, e.worker_id, e.gpu_id, e.attempt)
            for e in sim.trace
        ]

    first = SchedulerSim(random_workload(seed))
    first.run()
    second = SchedulerSim(random_workload(seed))
    second.run()

    assert trace_signature(first) == trace_signature(second)
