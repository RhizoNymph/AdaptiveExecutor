"""Property test: an exclusive task runs alone for its *whole* run.

Drives the real scheduler through an OOM-retry storm and asserts that no other
task's dispatch falls strictly inside any exclusive task's run window
(dispatch -> completion). Exclusive tasks arise from the executor's real
penalize+exclusive-retry path: a crash retry has ``retry_count >= 1``, set
together with ``PendingWork.exclusive``, so a dispatch event with
``retry_count >= 1`` identifies an exclusive run.

(Co-admission at the *exact* dispatch instant is a separate, pre-existing
pure-scheduler behavior for the idle case and is not what this test targets; the
run-isolation guarantee is that nothing new starts once an exclusive task is
already in flight -- i.e. strictly after its start and before it finishes.)
"""

from sim.harness import SchedulerSim
from sim.workloads import oom_storm_workload


def _run_end_time(trace, dispatch):
    """End of a dispatched attempt's run: the matching complete/crash/fail."""
    for e in trace:
        if (
            e.work_id == dispatch.work_id
            and e.attempt == dispatch.attempt
            and e.kind in ("complete", "crash", "fail")
            and e.time >= dispatch.time
        ):
            return e.time
    return None


def test_no_dispatch_overlaps_an_exclusive_run_window():
    workload = oom_storm_workload()
    sim = SchedulerSim(workload, max_resource_crash_retries=1)
    trace = sim.run()

    exclusive_dispatches = [
        e for e in trace if e.kind == "dispatch" and e.retry_count >= 1
    ]
    assert exclusive_dispatches, "expected at least one exclusive (retry) dispatch"

    all_dispatches = [e for e in trace if e.kind == "dispatch"]
    for excl in exclusive_dispatches:
        end = _run_end_time(trace, excl)
        assert end is not None and end > excl.time, (
            f"could not find a run window end for {excl.task_name} "
            f"(attempt {excl.attempt}) dispatched at t={excl.time}"
        )
        for d in all_dispatches:
            if d.work_id == excl.work_id:
                continue
            assert not (excl.time < d.time < end), (
                f"{d.task_name} dispatched at t={d.time} inside exclusive "
                f"{excl.task_name}'s run window ({excl.time}, {end}) -- an "
                f"exclusive task must run alone start to finish"
            )
