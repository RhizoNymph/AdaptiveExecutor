"""Unit tests for the pure observed-usage commitment accounting.

Covers the core rule ``committed = max(estimate - observed, 0)`` in the three
regimes (estimate > observed, observed > estimate -> 0, observed == estimate ->
0), aggregation over many tasks, and per-GPU maps with a mix of pinned/unpinned
tasks and overrunning (observed > estimate) tasks.
"""

import math

from adaptive_executor.accounting import (
    ResourceUsage,
    committed_gb,
    committed_vram_per_gpu,
    total_committed_gb,
)


# --- committed_gb (the core rule) ------------------------------------------


def test_committed_is_unrealized_remainder_when_estimate_exceeds_observed():
    assert committed_gb(10.0, 4.0) == 6.0


def test_committed_is_zero_when_observed_meets_estimate():
    assert committed_gb(10.0, 10.0) == 0.0


def test_committed_is_zero_when_observed_exceeds_estimate():
    # Overrun: the full real usage is already in the snapshot -> commit nothing.
    assert committed_gb(10.0, 15.0) == 0.0


def test_committed_never_negative():
    assert committed_gb(3.0, 100.0) == 0.0
    assert committed_gb(0.0, 0.0) == 0.0


def test_committed_with_zero_observed_is_the_full_estimate():
    # Baseline (no observation yet / unreadable process) keeps the full estimate.
    assert committed_gb(7.5, 0.0) == 7.5


# --- total_committed_gb (aggregation) --------------------------------------


def test_total_committed_sums_remainders():
    pairs = [(10.0, 4.0), (5.0, 5.0), (8.0, 12.0), (2.0, 0.0)]
    # 6 + 0 + 0 + 2 = 8
    assert total_committed_gb(pairs) == 8.0


def test_total_committed_empty_is_zero():
    assert total_committed_gb([]) == 0.0


# --- committed_vram_per_gpu (per-GPU maps) ---------------------------------


def test_per_gpu_groups_and_credits_observed():
    usages = [
        ResourceUsage(gpu_id=0, estimate_gb=6.0, observed_gb=2.0),  # 4 on gpu0
        ResourceUsage(gpu_id=0, estimate_gb=3.0, observed_gb=3.0),  # 0 on gpu0
        ResourceUsage(gpu_id=1, estimate_gb=8.0, observed_gb=1.0),  # 7 on gpu1
    ]
    assert committed_vram_per_gpu(usages) == {0: 4.0, 1: 7.0}


def test_per_gpu_overrun_contributes_zero():
    usages = [
        ResourceUsage(gpu_id=0, estimate_gb=4.0, observed_gb=9.0),  # overrun -> 0
        ResourceUsage(gpu_id=0, estimate_gb=5.0, observed_gb=1.0),  # 4
    ]
    assert committed_vram_per_gpu(usages) == {0: 4.0}


def test_per_gpu_ignores_unpinned_tasks():
    usages = [
        ResourceUsage(gpu_id=None, estimate_gb=10.0, observed_gb=0.0),
        ResourceUsage(gpu_id=2, estimate_gb=5.0, observed_gb=0.0),
    ]
    assert committed_vram_per_gpu(usages) == {2: 5.0}


def test_per_gpu_empty_is_empty_map():
    assert committed_vram_per_gpu([]) == {}


def test_per_gpu_all_realized_is_zero_not_absent():
    # A fully-realized GPU task still appears in the map with a 0 commitment,
    # since it was pinned; the key documents the pin even at zero remainder.
    usages = [ResourceUsage(gpu_id=1, estimate_gb=4.0, observed_gb=4.0)]
    assert committed_vram_per_gpu(usages) == {1: 0.0}


def test_committed_math_matches_release_projection_identity():
    # The amount the scheduler projects a task to RELEASE (its committed
    # remainder) plus its realized-but-already-in-snapshot usage never exceeds
    # max(estimate, observed) -- i.e. we never credit back more than the estimate
    # when under-realized, nor more than the observed when overrun.
    for estimate, observed in [(10.0, 3.0), (10.0, 10.0), (10.0, 14.0), (0.0, 5.0)]:
        release = committed_gb(estimate, observed)
        assert release <= max(estimate, observed) + 1e-12
        assert not math.isnan(release)
