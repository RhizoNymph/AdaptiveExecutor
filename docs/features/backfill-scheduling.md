# Feature: Reservation-based Backfill Scheduling

## Scope
- Replace strict head-of-line FIFO dispatch with EASY (reservation-based)
  backfill: when the front task cannot be admitted now, later tasks may be
  dispatched ahead of it **only** when doing so cannot delay the front task.
- A pure, deterministic scheduling component
  (`adaptive_executor/scheduling.py`, `plan_dispatch`) that decides which
  pending tasks to dispatch this cycle and on which GPU.
- p90 run-duration estimation (`ResourceEstimate.duration_p90_seconds`) used to
  reason about when running tasks release resources and whether a backfill task
  finishes in time.
- Per-GPU VRAM accounting for reservations and backfill placement.
- Correct interaction with exclusive tasks (`PendingWork.exclusive`): an
  exclusive task runs alone for its entire run — as a blocked head it admits no
  backfill, and once in flight it blocks all dispatch until it finishes.

## Non-scope
- Worker-pool lifecycle, GPU pinning, timeouts, crash retry, persistence — all
  unchanged (see `docs/features/executor.md`). The scheduler only decides
  *ordering/admission*; the executor still owns worker acquisition and I/O.
- No configuration flag: backfill is the default and only behavior.
- The scheduler does not model live per-task memory attribution; it reasons
  about *committed estimates* being released (matching the admission model),
  which is deliberately conservative.

## Key concepts
- **Reservation**: when the head of the queue cannot be admitted now, the
  earliest future time at which the currently running tasks are expected to
  release enough resources (memory, per-GPU VRAM, worker slots) for the head to
  start. Computed from each running task's remaining time.
- **Backfill rules** (a later task may run ahead of a blocked head iff):
  - **(a) resource-disjoint**: it fits in capacity that remains free even after
    setting aside everything the head's reservation requires — so it may run for
    any duration without delaying the head; or
  - **(b) short-enough**: it fits now *and* its expected duration means it
    finishes before the reservation time, releasing its resources before the
    head needs them.

## Data / control flow
1. **submit()** → `LearnedProfile.estimate()` now also computes
   `duration_p90_seconds` (p90 of observed `ResourceObservation.duration_seconds`,
   or `None` when the function has no history).
2. **_maybe_dispatch()** (dispatch thread, under `self.lock`):
   - `_build_dispatch_plan()` snapshots live state into the scheduler's inputs:
     - `Capacity.memory_free_gb` = `memory_total - memory_used - memory_headroom
       - committed_running_memory` (or `math.inf` when no snapshot yet).
     - `Capacity.gpu_free_vram_gb[g]` = `vram_total - vram_used - vram_headroom -
       committed_running_vram_on_g` for each GPU present in the snapshot (via
       `_committed_vram_per_gpu`); `math.inf` per GPU when no snapshot.
     - `RunningEntry.remaining_seconds` = `_running_remaining_seconds(pending)` —
       `duration_p90 - elapsed`, or `None` if duration unknown, not started, or
       the task has **overrun** its estimate (elapsed >= p90).
     - `RunningEntry.exclusive` = `pending.exclusive` — whether this in-flight
       task must run alone (used to gate the whole cycle; see below).
     - `PendingEntry` carries memory/VRAM/CPU/`duration_p90_seconds`/`exclusive`.
   - `plan_dispatch(pending, running, capacity)` returns a `DispatchPlan`
     (ordered `DispatchDecision(pending_id, gpu_id)` + updated `next_gpu_index`).
   - The executor applies `next_gpu_index`, then for each decision obtains a
     worker (`_get_or_spawn_idle_worker`) and, on success, moves the task to
     `in_flight` and sends the `WorkItem`. Dispatched ids are removed from
     `self.pending` (which may become non-contiguous).
3. **plan_dispatch / `_Planner`** (pure):
   0. If **any running entry is exclusive**, return an empty plan immediately —
      an exclusive task in flight owns the whole machine until it leaves the
      running set, so nothing else may be dispatched this cycle.
   1. Greedily admit front tasks that fit now, in FIFO order (identical to strict
      FIFO). Each front admit consumes the live pools and becomes an occupant
      that starts now (remaining = its own p90 duration).
   2. The first task that does not fit now is the **head**.
      - If the head is **exclusive**, stop — no backfill.
      - Otherwise compute the head's reservation (`_compute_reservation`): the
        earliest release time at which the head fits, projecting occupants that
        free by that time (`_state_at`). Infinite if unreachable from finite
        releases.
   3. Scan the rest of the queue in FIFO order (`_backfill`). For each candidate
      that passes the GPU-independent gates (`_fits_nongpu`): classify as rule
      (b) if the reservation is finite and its duration ≤ reservation; else check
      rule (a) via `_head_still_fits_with_hold` (does the head still fit at its
      reservation while this candidate holds its resources indefinitely?). GPU
      candidates are tried in round-robin order (`_gpu_options`) so a task is
      steered onto a GPU disjoint from the head's reservation when possible.

## Conservative fallbacks (explicit)
- **Unknown duration ⇒ infinite.** A running task with unknown/overrun duration
  is assumed to never release; the head cannot count on it freeing resources. A
  backfill task with unknown duration can never satisfy rule (b) — it may only
  backfill via rule (a) (resource-disjoint).
- **Unknown reservation ⇒ rule (a) only.** If the head's reservation is infinite
  (it depends on an unknown-duration running task, or capacity is fundamentally
  insufficient), rule (b) is disabled entirely; only disjoint backfill is
  allowed.
- **Overrun ⇒ slip toward FIFO.** Once a running task's elapsed time reaches its
  p90 estimate, its remaining is reported as unknown, so the reservation
  recomputed next cycle slips later (never earlier), degrading toward FIFO —
  never below it.
- **Exclusive head ⇒ no backfill.** An exclusive head requires the in-flight set
  to drain to empty; admitting anything would extend that drain, so an exclusive
  head blocks all backfill.
- **Exclusive task running ⇒ no dispatch at all.** Once an exclusive task is in
  flight, the scheduler returns an empty plan every cycle until it leaves the
  running set, so it runs alone start to finish (not just at its start). This is
  the OOM-retry case: a task killed under memory pressure is penalized and
  marked exclusive, and must keep the machine to itself for its whole retry.
- **No snapshot ⇒ memory/VRAM non-gating.** Before the first monitor snapshot,
  memory and VRAM are treated as infinite, matching prior startup admission.

## Files and roles
- `adaptive_executor/scheduling.py` — pure scheduler. Inputs: `PendingEntry`,
  `RunningEntry`, `Capacity`. Output: `DispatchPlan(decisions, next_gpu_index)`.
  Entry point `plan_dispatch`; internal `_Planner` implements front admission,
  `_compute_reservation`, `_state_at`, `_backfill`, `_place_backfill`,
  `_head_still_fits_with_hold`, `_fits`/`_fits_nongpu`, `_gpu_options`.
- `adaptive_executor/adaptive_executor.py` — `_maybe_dispatch` (executes the
  plan), `_build_dispatch_plan` (gathers state), `_running_remaining_seconds`,
  reused `_committed_resources` / `_committed_vram_per_gpu`.
- `adaptive_executor/dtypes.py` — `ResourceEstimate.duration_p90_seconds`.
- `adaptive_executor/profiles.py` — `LearnedProfile.estimate()` computes the p90
  duration.
- `tests/test_scheduling.py` — pure unit tests (all scenarios).
- `tests/test_scheduling_exclusive.py` — pure unit tests for exclusive *run*
  isolation (a running exclusive task blocks all dispatch; exclusivity lifts
  when it leaves the running set).
- `tests/sim/test_exclusive_isolation.py` — sim property test: in an OOM-retry
  storm, no dispatch overlaps an exclusive task's run window.
- `tests/test_backfill_integration.py` — integration tests through the real
  executor dispatch path with fake spawning and an injected snapshot.

## Invariants and constraints
- **Never later than FIFO.** The head task never starts later than it would
  under strict FIFO: front tasks are admitted in strict FIFO order, and a task is
  backfilled past a blocked head only when it either finishes before the head's
  reservation (rule b) or fits in capacity disjoint from the reservation (rule
  a). In both cases the head's earliest feasible start is unchanged.
- **Per-GPU safety.** A backfill task never consumes VRAM on the GPU the head is
  reserving unless it finishes before the reservation.
- **Exclusive runs alone, start to finish.** An exclusive task never shares the
  machine once it is in flight: as a blocked head it admits no backfill, and
  while running it forces an empty plan every cycle, so no other task is
  dispatched until it leaves the running set (completes/fails/crashes).
- **Purity/determinism.** `plan_dispatch` is a pure function of its inputs — no
  threads, no clock reads, no I/O — so it is exhaustively unit-testable. The
  executor performs all time reads and state capture before calling it.
- **Model consistency.** Free resources and releases are reasoned about in terms
  of committed estimates (the same quantities admission uses), and unknowns are
  always resolved in the conservative (FIFO-ward) direction.
- **Reservation recomputed every cycle** from current running state, so overruns
  naturally slip the reservation.
