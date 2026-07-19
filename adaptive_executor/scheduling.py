"""Pure, deterministic EASY (reservation-based) backfill scheduler.

This module contains no executor state, no threads, and no I/O. It is a pure
function of its inputs so it can be exhaustively unit tested. The executor
gathers a live snapshot of its state into the input dataclasses, calls
:func:`plan_dispatch`, and then executes the returned decisions.

Scheduling model
----------------
Tasks are considered in FIFO order. As long as the front task can be admitted
now, it is admitted (identical to strict FIFO). When the front task ("the head")
cannot be admitted now, we do NOT simply stall the whole queue. Instead:

1. We compute a RESERVATION for the head: the earliest future time at which the
   currently-running tasks are expected to release enough resources (memory,
   per-GPU VRAM, worker slots) for the head to start.
2. We scan the rest of the queue in FIFO order and admit a later task ONLY IF
   doing so cannot delay the head's reservation, via one of two rules:
     (a) resource-disjoint: the task fits in capacity that remains free even
         after setting aside everything the head's reservation requires (so the
         task may run arbitrarily long without delaying the head); or
     (b) short-enough: the task's expected duration means it finishes before the
         reservation time (and it fits now), so it releases its resources before
         the head needs them.

Invariant: the head never starts later than it would under strict FIFO.

Conservative fallbacks
----------------------
* Unknown duration (``duration_p90_seconds is None``) is treated as INFINITE. A
  backfill task with unknown duration can never satisfy rule (b); it may only
  backfill via rule (a). A running task with unknown duration is assumed to
  never release, so the head cannot count on it freeing resources.
* If the head's reservation time is infinite (it depends on an unknown-duration
  running task, or capacity is fundamentally insufficient), rule (b) is disabled
  entirely and only rule (a) backfilling is allowed.
* An overrunning running task (observed elapsed >= its estimate) is converted to
  unknown remaining by the caller, so the reservation naturally slips later,
  degrading toward FIFO — never below it.
* An exclusive head (needs an empty in-flight set) reserves the entire drain of
  every running task; admitting anything would delay that drain, so an exclusive
  head blocks all backfill.
"""

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "PendingEntry",
    "RunningEntry",
    "Capacity",
    "DispatchDecision",
    "DispatchPlan",
    "plan_dispatch",
]


@dataclass(frozen=True)
class PendingEntry:
    """A queued task awaiting dispatch, in FIFO order."""

    id: str
    memory_gb: float
    vram_gb: float
    cpu_cores: float
    # None => unknown duration => treated as infinite for backfill.
    duration_p90_seconds: float | None
    exclusive: bool = False


@dataclass(frozen=True)
class RunningEntry:
    """A currently in-flight task holding resources."""

    id: str
    memory_gb: float
    vram_gb: float
    gpu_id: int | None
    # Expected seconds until this task releases its resources. None => unknown
    # (assumed to never release: overran its estimate or has no history).
    remaining_seconds: float | None


@dataclass(frozen=True)
class Capacity:
    """Admittable capacity for this dispatch cycle.

    ``memory_free_gb`` and ``gpu_free_vram_gb`` are what is admittable *right
    now* — i.e. already net of headroom, live usage, and the committed estimates
    of running tasks. When no monitor snapshot is available these are
    ``math.inf`` and ``snapshot_present`` is False (memory/VRAM are not gating).
    """

    memory_free_gb: float
    gpu_free_vram_gb: Mapping[int, float]
    gpu_round_robin: tuple[int, ...]
    next_gpu_index: int
    max_workers: int
    running_count: int
    snapshot_present: bool


@dataclass(frozen=True)
class DispatchDecision:
    pending_id: str
    gpu_id: int | None


@dataclass(frozen=True)
class DispatchPlan:
    decisions: tuple[DispatchDecision, ...]
    next_gpu_index: int


@dataclass
class _Pools:
    """Mutable live resource pools consumed as tasks are admitted this cycle."""

    memory_free: float
    gpu_free: dict[int, float]
    slots: int
    rr: int


def plan_dispatch(
    pending: list[PendingEntry],
    running: list[RunningEntry],
    capacity: Capacity,
) -> DispatchPlan:
    """Compute the set of tasks to dispatch this cycle. Pure and deterministic."""
    planner = _Planner(pending, running, capacity)
    return planner.run()


class _Planner:
    def __init__(
        self,
        pending: list[PendingEntry],
        running: list[RunningEntry],
        capacity: Capacity,
    ):
        self.cap = capacity
        self.queue = list(pending)
        # Occupants hold resources through the cycle. Front-admitted tasks are
        # appended (they start now with remaining == their duration).
        self.occupants: list[RunningEntry] = list(running)
        self.pools = _Pools(
            memory_free=capacity.memory_free_gb,
            gpu_free=dict(capacity.gpu_free_vram_gb),
            slots=capacity.running_count,
            rr=capacity.next_gpu_index,
        )
        self.decisions: list[DispatchDecision] = []

    def run(self) -> DispatchPlan:
        # 1. Greedily admit front tasks that fit now (strict FIFO behavior).
        while self.queue:
            head = self.queue[0]
            fits, gpu_id, new_rr = self._fits(
                head, self.pools.memory_free, self.pools.gpu_free, self.pools.slots, self.pools.rr
            )
            if not fits:
                break
            self.pools.rr = new_rr
            self._admit(head, gpu_id, releases_before_reservation=None)
            self.queue.pop(0)
        else:
            # Whole queue drained; nothing is blocked.
            return self._plan()

        # 2. The head is blocked. Backfill the rest without delaying it.
        head = self.queue[0]
        if head.exclusive:
            # An exclusive head needs the in-flight set to drain to empty;
            # admitting anything delays that drain. No backfill.
            return self._plan()

        reservation = self._compute_reservation(head)
        self._backfill(head, reservation)
        return self._plan()

    def _plan(self) -> DispatchPlan:
        return DispatchPlan(
            decisions=tuple(self.decisions), next_gpu_index=self.pools.rr
        )

    # -- admission primitives ------------------------------------------------

    def _effective_max(self, task: PendingEntry) -> int:
        # Mirrors the executor's cpu-cores derived cap; only applied when a
        # monitor snapshot is present (matching prior admission behavior).
        if not self.cap.snapshot_present:
            return self.cap.max_workers
        if task.cpu_cores > 1:
            return max(1, int(self.cap.max_workers / task.cpu_cores))
        return self.cap.max_workers

    def _pick_gpu(
        self, vram_gb: float, gpu_free: Mapping[int, float], rr: int
    ) -> tuple[int | None, int]:
        """Round-robin first-fit GPU selection; returns (gpu_id, next_rr)."""
        order = self.cap.gpu_round_robin
        n = len(order)
        if n == 0:
            return None, rr
        for offset in range(n):
            idx = (rr + offset) % n
            gpu_id = order[idx]
            if gpu_id not in gpu_free:
                continue
            if vram_gb < gpu_free[gpu_id]:
                return gpu_id, (idx + 1) % n
        return None, rr

    def _fits(
        self,
        task: PendingEntry,
        memory_free: float,
        gpu_free: Mapping[int, float],
        slots: int,
        rr: int,
    ) -> tuple[bool, int | None, int]:
        """Whether ``task`` is admissible against the given pools.

        Returns (fits, gpu_id, next_rr). Does not mutate anything.
        """
        if task.exclusive and slots > 0:
            return False, None, rr
        if slots >= self.cap.max_workers:
            return False, None, rr
        if slots >= self._effective_max(task):
            return False, None, rr
        if task.memory_gb >= memory_free:
            return False, None, rr
        gpu_id: int | None = None
        if task.vram_gb > 0:
            gpu_id, rr = self._pick_gpu(task.vram_gb, gpu_free, rr)
            if gpu_id is None:
                return False, None, rr
        return True, gpu_id, rr

    def _admit(
        self,
        task: PendingEntry,
        gpu_id: int | None,
        *,
        releases_before_reservation: bool | None,
    ) -> None:
        """Record a dispatch decision and consume its resources from the pools.

        ``releases_before_reservation`` controls whether this task is added as a
        held occupant of the head's reservation:
          * None  -> front admit (a new occupant; may free at its own duration).
          * True  -> backfill via rule (b); frees before the reservation, so it
                     is NOT added as an occupant that holds through the reservation.
          * False -> backfill via rule (a); holds its resources indefinitely
                     relative to the reservation. Its consumption of the live
                     pools already reflects the hold, so no occupant entry needed.
        """
        self.decisions.append(DispatchDecision(pending_id=task.id, gpu_id=gpu_id))
        self.pools.memory_free -= task.memory_gb
        if gpu_id is not None and gpu_id in self.pools.gpu_free:
            self.pools.gpu_free[gpu_id] -= task.vram_gb
        self.pools.slots += 1
        if releases_before_reservation is None:
            # Front admit: becomes an occupant that starts now.
            self.occupants.append(
                RunningEntry(
                    id=task.id,
                    memory_gb=task.memory_gb,
                    vram_gb=task.vram_gb,
                    gpu_id=gpu_id,
                    remaining_seconds=task.duration_p90_seconds,
                )
            )

    # -- reservation ---------------------------------------------------------

    def _release_times(self) -> list[float]:
        return sorted({o.remaining_seconds for o in self.occupants if o.remaining_seconds is not None})

    def _state_at(self, t: float) -> tuple[float, dict[int, float], int]:
        """Live pools projected to time ``t``, assuming occupants with
        ``remaining_seconds <= t`` have freed their resources and worker slots.
        """
        memory = self.pools.memory_free
        gpu = dict(self.pools.gpu_free)
        slots = self.pools.slots
        for o in self.occupants:
            if o.remaining_seconds is not None and o.remaining_seconds <= t:
                memory += o.memory_gb
                slots -= 1
                if o.gpu_id is not None and o.gpu_id in gpu:
                    gpu[o.gpu_id] += o.vram_gb
        return memory, gpu, slots

    def _compute_reservation(self, head: PendingEntry) -> float:
        """Earliest time the head is expected to be admissible. May be inf."""
        for t in self._release_times():
            memory, gpu, slots = self._state_at(t)
            fits, _, _ = self._fits(head, memory, gpu, slots, self.pools.rr)
            if fits:
                return t
        return math.inf

    # -- backfill ------------------------------------------------------------

    def _fits_nongpu(self, task: PendingEntry) -> bool:
        """The GPU-independent part of admissibility against the live pools."""
        if task.exclusive and self.pools.slots > 0:
            return False
        if self.pools.slots >= self.cap.max_workers:
            return False
        if self.pools.slots >= self._effective_max(task):
            return False
        if task.memory_gb >= self.pools.memory_free:
            return False
        return True

    def _gpu_options(self, vram_gb: float) -> list[tuple[int, int]]:
        """GPUs (in round-robin order) with room for ``vram_gb`` now, each paired
        with the resulting next round-robin index if chosen.
        """
        order = self.cap.gpu_round_robin
        n = len(order)
        options: list[tuple[int, int]] = []
        for offset in range(n):
            idx = (self.pools.rr + offset) % n
            gpu_id = order[idx]
            if gpu_id not in self.pools.gpu_free:
                continue
            if vram_gb < self.pools.gpu_free[gpu_id]:
                options.append((gpu_id, (idx + 1) % n))
        return options

    def _backfill(self, head: PendingEntry, reservation: float) -> None:
        for candidate in self.queue[1:]:
            if not self._fits_nongpu(candidate):
                continue

            duration = candidate.duration_p90_seconds
            finishes_before_reservation = (
                math.isfinite(reservation)
                and duration is not None
                and duration <= reservation
            )

            placement = self._place_backfill(
                head, candidate, reservation, finishes_before_reservation
            )
            if placement is None:
                continue
            gpu_id, new_rr, releases = placement
            self.pools.rr = new_rr
            self._admit(candidate, gpu_id, releases_before_reservation=releases)

    def _place_backfill(
        self,
        head: PendingEntry,
        candidate: PendingEntry,
        reservation: float,
        finishes_before_reservation: bool,
    ) -> tuple[int | None, int, bool] | None:
        """Choose a GPU (if needed) on which the candidate can backfill without
        delaying the head. Returns (gpu_id, next_rr, releases_before_reservation)
        or None. GPU candidates are tried in round-robin order so a task is
        steered onto a GPU disjoint from the head's reservation when possible.
        """
        if candidate.vram_gb <= 0:
            options: list[tuple[int | None, int]] = [(None, self.pools.rr)]
        else:
            gpu_opts = self._gpu_options(candidate.vram_gb)
            if not gpu_opts:
                return None
            options = list(gpu_opts)

        for gpu_id, next_rr in options:
            if finishes_before_reservation:
                # Rule (b): releases before the head needs its resources. Any GPU
                # with room is fine, even the one the head is waiting for.
                return gpu_id, next_rr, True
            # Rule (a): admit only if the head still fits at its reservation while
            # this candidate holds its resources (on this GPU) indefinitely.
            if self._head_still_fits_with_hold(head, candidate, gpu_id, reservation):
                return gpu_id, next_rr, False
        return None

    def _head_still_fits_with_hold(
        self,
        head: PendingEntry,
        candidate: PendingEntry,
        candidate_gpu: int | None,
        reservation: float,
    ) -> bool:
        """Does the head still fit at its reservation if ``candidate`` is admitted
        now and holds its resources through the reservation time?
        """
        # Project the live pools to the reservation, then remove what the
        # candidate would hold (it does not free before the reservation).
        # _state_at(inf) frees every finite-duration occupant; unknown ones never
        # free. That is exactly the projection we want for an infinite reservation.
        memory, gpu, slots = self._state_at(reservation)

        memory -= candidate.memory_gb
        slots += 1
        if candidate_gpu is not None and candidate_gpu in gpu:
            gpu[candidate_gpu] -= candidate.vram_gb

        fits, _, _ = self._fits(head, memory, gpu, slots, self.pools.rr)
        return fits
