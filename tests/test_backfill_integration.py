"""Integration tests: reservation-based backfill driven through the real
AdaptiveExecutor dispatch path (``_maybe_dispatch``), using fake worker spawning
and an injected monitor snapshot. Deterministic; no real subprocesses or GPU.
"""

import time
import uuid

import adaptive_executor.adaptive_executor as ae
from adaptive_executor.adaptive_executor import AdaptiveExecutor, PendingWork, WorkerSlot
from adaptive_executor.dtypes import ResourceEstimate, ResourceSnapshot, WorkItem
from fakes import FakeProcess, FakeQueue


def _make_executor(max_workers=4, **kw):
    ex = AdaptiveExecutor(max_workers=max_workers, **kw)

    def fake_spawn(pinned_gpu_id):
        wid = ex._next_worker_id
        ex._next_worker_id += 1
        slot = WorkerSlot(
            worker_id=wid,
            process=FakeProcess(alive=True),
            work_queue=FakeQueue(),
            pinned_gpu_id=pinned_gpu_id,
        )
        ex.workers[wid] = slot
        return slot

    ex._spawn_worker = fake_spawn  # type: ignore[method-assign]
    return ex


def _set_snapshot(ex, *, memory_total=100.0, memory_used=0.0):
    ex.monitor._current = ResourceSnapshot(
        cpu_percent=0.0,
        memory_used_gb=memory_used,
        memory_total_gb=memory_total,
        gpus={},
    )


def _pending_work(*, mem, vram=0.0, cpu=1.0, duration=None, exclusive=False):
    item = WorkItem(
        id=str(uuid.uuid4()),
        fn_module="m",
        fn_name="f",
        args=(),
        kwargs={},
        gpu_id=None,
    )
    est = ResourceEstimate(
        memory_gb=mem,
        vram_gb=vram,
        cpu_cores=cpu,
        confidence=1.0,
        duration_p90_seconds=duration,
    )
    return PendingWork(item=item, future=ae.Future(), estimate=est, submitted_at=time.time(), exclusive=exclusive)


def _place_in_flight(ex, work, *, elapsed=0.0):
    work.started_at = time.time() - elapsed
    work.worker_id = -1
    ex.in_flight[work.item.id] = work


def test_backfill_dispatches_small_task_past_blocked_head():
    ex = _make_executor(max_workers=4)
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)

    # Running task holds 40GB, expected to finish in ~10s (reservation source).
    running = _pending_work(mem=40.0, duration=10.0)
    _place_in_flight(ex, running, elapsed=0.0)

    # Available now = 100 - 0 - headroom(2) - committed(40) = 58.
    head = _pending_work(mem=70.0, duration=5.0)   # 70 >= 58 -> blocked now
    small = _pending_work(mem=5.0, duration=1000.0)  # disjoint, fits in slack
    ex.pending.append(head)
    ex.pending.append(small)

    ex._maybe_dispatch()

    # Head still blocked (head-of-line), small backfilled past it.
    assert head.item.id not in ex.in_flight
    assert small.item.id in ex.in_flight
    pending_ids = [p.item.id for p in ex.pending]
    assert pending_ids == [head.item.id]


def test_backfill_rejects_task_that_would_delay_head():
    ex = _make_executor(max_workers=4)
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)

    running = _pending_work(mem=40.0, duration=10.0)
    _place_in_flight(ex, running, elapsed=0.0)

    # Reservation at t=10: mem 58 + 40 = 98; head needs 70 -> 28 slack. A 25GB
    # long task held through the reservation leaves 98-25=73... that still fits.
    # Use 35GB so 98-35=63 < 70 -> would delay the head -> rejected.
    head = _pending_work(mem=70.0, duration=5.0)
    big = _pending_work(mem=35.0, duration=1000.0)
    ex.pending.append(head)
    ex.pending.append(big)

    ex._maybe_dispatch()

    assert head.item.id not in ex.in_flight
    assert big.item.id not in ex.in_flight
    assert [p.item.id for p in ex.pending] == [head.item.id, big.item.id]


def test_exclusive_head_blocks_all_backfill_integration():
    ex = _make_executor(max_workers=4)
    _set_snapshot(ex, memory_total=100.0, memory_used=0.0)

    running = _pending_work(mem=1.0, duration=10.0)
    _place_in_flight(ex, running, elapsed=0.0)

    head = _pending_work(mem=1.0, exclusive=True)  # needs in_flight empty
    tiny = _pending_work(mem=0.1, duration=0.001)
    ex.pending.append(head)
    ex.pending.append(tiny)

    ex._maybe_dispatch()

    # Nothing dispatched: exclusive head is blocked and nothing may pass it.
    assert head.item.id not in ex.in_flight
    assert tiny.item.id not in ex.in_flight
    assert [p.item.id for p in ex.pending] == [head.item.id, tiny.item.id]


def test_overrun_running_task_slips_reservation_integration():
    # With the running task within its estimate, a short task backfills via rule
    # (b). Once the running task overruns (elapsed >= estimate), remaining becomes
    # unknown, the reservation slips to infinite, and the same short task is no
    # longer allowed to backfill (it is not disjoint).
    def build(elapsed):
        ex = _make_executor(max_workers=4)
        _set_snapshot(ex, memory_total=100.0, memory_used=0.0)
        running = _pending_work(mem=40.0, duration=10.0)
        _place_in_flight(ex, running, elapsed=elapsed)
        head = _pending_work(mem=70.0, duration=5.0)
        short = _pending_work(mem=25.0, duration=0.5)
        ex.pending.append(head)
        ex.pending.append(short)
        return ex, head, short

    # Within estimate: reservation ~ t=10; short (0.5s) finishes first -> admit.
    ex1, head1, short1 = build(elapsed=1.0)
    ex1._maybe_dispatch()
    assert short1.item.id in ex1.in_flight

    # Overran: remaining unknown -> reservation infinite -> rule (b) disabled.
    # short holds 25GB: at inf, 58 (no release) - 25 = 33 < 70 -> not disjoint.
    ex2, head2, short2 = build(elapsed=12.0)
    ex2._maybe_dispatch()
    assert short2.item.id not in ex2.in_flight
    assert head2.item.id not in ex2.in_flight
