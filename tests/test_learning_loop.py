"""Executor-level tests for the learning loop, driven through the real
``AdaptiveExecutor`` with fake worker spawning and an injected snapshot.

Covers:
- dispatch-time re-estimation refreshes a queued sibling after a completion,
- user hints survive re-estimation (hint keeps overriding the learned value),
- a crash-penalized entry (retry_count > 0) is never re-estimated (its doubled
  estimate must stand),
- the crash-retry path records a memory floor into the base AND keyed profiles,
- the cold-start canary admits one sibling, then unclamps the rest once the
  first completion warms the profile.
"""

import signal
import time
import uuid

import adaptive_executor.adaptive_executor as ae
from adaptive_executor.adaptive_executor import AdaptiveExecutor, PendingWork, WorkerSlot
from adaptive_executor.dtypes import (
    ResourceEstimate,
    ResourceObservation,
    ResourceSnapshot,
    WorkItem,
)
from adaptive_executor.profiles import derive_store_key
from fakes import FakeProcess, FakeQueue, StubSystemMonitor

import task_fns


def _obs(mem=1.0, dur=0.1):
    return ResourceObservation(
        memory_delta_gb=mem, vram_delta_gb=0.0, cpu_percent=50.0, duration_seconds=dur
    )


def _snapshot(*, memory_total_gb=100.0, memory_used_gb=0.0, gpus=None):
    return ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        gpus=gpus or {},
    )


def _make_executor(max_workers=4, snapshot=None, gpu_ids=None, **kw):
    ex = AdaptiveExecutor(max_workers=max_workers, gpu_ids=gpu_ids, **kw)
    ex.monitor = StubSystemMonitor(snapshot=snapshot)
    ex._started = True  # bypass real start(): no threads, no real monitor

    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True, pid=1000 + wid),
            work_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]
    return ex


def _pending_work(*, fn_module="task_fns", fn_name="echo", estimate, profile_key=None,
                  retry_count=0, memory_hint=None, vram_hint=None):
    item = WorkItem(
        id=str(uuid.uuid4()),
        fn_module=fn_module,
        fn_name=fn_name,
        args=(),
        kwargs={},
        gpu_id=None,
    )
    return PendingWork(
        item=item,
        future=ae.Future(),
        estimate=estimate,
        submitted_at=time.time(),
        profile_key=profile_key,
        retry_count=retry_count,
        memory_hint=memory_hint,
        vram_hint=vram_hint,
        estimate_sample_count=0,
        estimate_refreshed_at=time.time(),
    )


# --- dispatch-time re-estimation --------------------------------------------


def test_reestimation_refreshes_queued_sibling_after_completion():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    fut = ex.submit(task_fns.echo, 1)  # no hint -> learned estimate
    pending = ex.pending[0]
    original_mem = pending.estimate.memory_gb  # default ~1.5 GB, 0 observations

    # A sibling completes: record a large observation into the profile.
    for _ in range(5):
        ex.profiles.record("task_fns", "echo", _obs(mem=20.0))

    # Past the throttle interval -> the queued task re-estimates upward.
    ex._reestimate_pending(pending.estimate_refreshed_at + 1.0)

    assert pending.estimate.memory_gb > original_mem
    assert pending.estimate_sample_count == 5
    assert not fut.done()


def test_reestimation_throttled_within_interval():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.submit(task_fns.echo, 1)
    pending = ex.pending[0]
    ex.profiles.record("task_fns", "echo", _obs(mem=20.0))

    # Within the interval since submit -> skipped, estimate unchanged.
    ex._reestimate_pending(pending.estimate_refreshed_at + 0.01)
    assert pending.estimate_sample_count == 0


def test_hint_survives_reestimation():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.submit(task_fns.echo, 1, memory_gb=3.0)
    pending = ex.pending[0]

    # Many large observations would push a learned estimate to ~20 GB.
    for _ in range(10):
        ex.profiles.record("task_fns", "echo", _obs(mem=20.0))

    ex._reestimate_pending(pending.estimate_refreshed_at + 1.0)

    # The hint keeps overriding: memory stays anchored to the 3 GB hint, not 20.
    assert pending.estimate.memory_gb <= 4.5  # 3.0 * max safety (1.5)
    assert pending.memory_hint == 3.0


def test_retry_count_entry_is_not_reestimated():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    penalized_est = ResourceEstimate(memory_gb=8.0, vram_gb=0.0, cpu_cores=1.0)
    pending = _pending_work(estimate=penalized_est, retry_count=1)
    pending.exclusive = True
    ex.pending.append(pending)

    # Observations that would otherwise lower the estimate.
    for _ in range(10):
        ex.profiles.record("task_fns", "echo", _obs(mem=1.0))

    ex._reestimate_pending(pending.estimate_refreshed_at + 1.0)

    # Crash penalty stands: doubled estimate + exclusive untouched.
    assert pending.estimate.memory_gb == 8.0
    assert pending.exclusive is True


# --- crash records a memory floor into base + keyed profiles ----------------


def test_penalize_records_floor_into_base_and_keyed():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    est = ResourceEstimate(memory_gb=5.0, vram_gb=0.0, cpu_cores=1.0)
    pending = _pending_work(estimate=est, profile_key="large")

    ex._penalize_estimate(pending)

    base = ex.profiles.profiles[derive_store_key("task_fns", "echo")]
    keyed = ex.profiles.profiles[derive_store_key("task_fns", "echo", "large")]
    # Floor is the admitted (pre-double) estimate, proven too small.
    assert base.memory_floor_gb == 5.0
    assert keyed.memory_floor_gb == 5.0
    # Estimate doubled and marked exclusive for the retry itself.
    assert pending.estimate.memory_gb == 10.0
    assert pending.exclusive is True


def test_crash_retry_path_records_floor_and_requeues():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0), max_resource_crash_retries=1)
    est = ResourceEstimate(memory_gb=6.0, vram_gb=0.0, cpu_cores=1.0)
    pending = _pending_work(estimate=est, profile_key="big")

    worker = WorkerSlot(
        worker_id=7,
        process=FakeProcess(alive=True, pid=4321),
        work_queue=FakeQueue(),
        pinned_gpu_id=None,
    )
    worker.current_work_id = pending.item.id
    ex.in_flight[pending.item.id] = pending
    worker.process.set_dead(exitcode=-signal.SIGKILL)

    ex._handle_dead_worker(worker)

    # Floor persisted to base + keyed; the task re-queued for its penalized retry.
    assert ex.profiles.profiles[derive_store_key("task_fns", "echo")].memory_floor_gb == 6.0
    assert ex.profiles.profiles[derive_store_key("task_fns", "echo", "big")].memory_floor_gb == 6.0
    assert pending in ex.pending
    assert pending.retry_count == 1
    assert pending.exclusive is True


def test_recorded_floor_lower_bounds_a_fresh_submit():
    # The whole point: after a crash floor, the next fresh submit of the same
    # function is estimated at (at least) the floor even with no observations.
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.profiles.record_memory_floor("task_fns", "echo", 9.0)
    ex.submit(task_fns.echo, 1)
    assert ex.pending[0].estimate.memory_gb >= 9.0


# --- cold-start canary at the executor level --------------------------------


def test_canary_admits_one_cold_sibling_then_unclamps_after_completion():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    fut_a = ex.submit(task_fns.echo, 1)
    fut_b = ex.submit(task_fns.echo, 2)
    a_id, b_id = ex.pending[0].item.id, ex.pending[1].item.id
    assert len(ex.pending) == 2

    # First cycle: only the canary (A, the FIFO head) is admitted; B is
    # canary-blocked and remains queued.
    ex._maybe_dispatch()
    assert list(ex.in_flight) == [a_id]
    assert [p.item.id for p in ex.pending] == [b_id]

    # The canary completes and warms the profile.
    ex.profiles.record("task_fns", "echo", _obs(mem=1.0))

    # Next cycle: the profile is warm, so B is no longer cold and unclamps.
    ex._maybe_dispatch()
    assert len(ex.in_flight) == 2
    assert len(ex.pending) == 0
    assert not fut_a.done() and not fut_b.done()


def test_hint_bypasses_canary_at_executor_level():
    # Two same-function cold submits, but both carry a hint -> neither is cold,
    # so both admit together (the canary is bypassed).
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.submit(task_fns.echo, 1, memory_gb=2.0)
    ex.submit(task_fns.echo, 2, memory_gb=2.0)

    ex._maybe_dispatch()
    assert len(ex.in_flight) == 2
    assert len(ex.pending) == 0


def test_different_functions_each_get_their_own_canary():
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.submit(task_fns.echo, 1)
    ex.submit(task_fns.add, 1, 2)

    ex._maybe_dispatch()
    # Distinct identities -> both canaries admit.
    assert len(ex.in_flight) == 2


def test_floor_change_triggers_reestimation_without_new_samples():
    # A crash floor is recorded WITHOUT adding an observation (the worker died
    # before reporting). Queued siblings must still pick the floor up on the
    # next re-estimation pass — comparing sample_count alone would leave them
    # at the too-small estimate and repeat the very crash the floor prevents.
    ex = _make_executor(snapshot=_snapshot(memory_total_gb=100.0))
    ex.submit(task_fns.echo, 1)
    pending = ex.pending[0]
    original_mem = pending.estimate.memory_gb

    # Sibling crashed: floor recorded, sample_count unchanged (no observation).
    ex.profiles.record_memory_floor("task_fns", "echo", 30.0)

    ex._reestimate_pending(pending.estimate_refreshed_at + 1.0)

    assert pending.estimate.memory_gb == 30.0
    assert pending.estimate.memory_gb > original_mem
    assert pending.estimate_floor_gb == 30.0
    assert pending.estimate_sample_count == 0
