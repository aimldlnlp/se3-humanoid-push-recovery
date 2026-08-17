import numpy as np

from se3_whole_body_control.control.diagnostic_wbc import (
    PROFILE_DEFINITIONS,
    diagnostic_profile,
)
from experiments.post_touchdown_wbc_ablation import (
    HIGH_UTILIZATION_THRESHOLD,
    _sustained_onset,
    _threshold_metrics,
)


def test_primary_profile_set_is_fixed_and_production_independent():
    assert tuple(PROFILE_DEFINITIONS) == (
        "baseline_landed_support",
        "pose_pruned",
        "centroidal_momentum",
        "joint_limit_guarded",
        "combined_minimal",
    )
    assert not diagnostic_profile("baseline_landed_support").pose_pruned
    assert diagnostic_profile("pose_pruned").pose_pruned
    assert diagnostic_profile("centroidal_momentum").centroidal_momentum
    assert diagnostic_profile("joint_limit_guarded").joint_limit_guarded
    combined = diagnostic_profile("combined_minimal")
    assert combined.pose_pruned and combined.centroidal_momentum and combined.joint_limit_guarded
    assert combined.momentum_decay_time_s == 0.25
    assert combined.joint_guard_rad == 0.02


def test_sustained_onset_requires_two_samples():
    time_s = np.arange(5, dtype=float) * 0.004
    assert _sustained_onset(time_s, np.array([False, True, False, True, True])) == 0.012
    assert _sustained_onset(time_s, np.array([False, True, False, False, False])) is None


def test_high_utilization_metrics_use_declared_threshold_and_duration():
    time_s = np.arange(6, dtype=float) * 0.004
    values = np.array([0.2, 0.96, 0.97, 0.4, 0.95, 0.96])
    metrics = _threshold_metrics(time_s, values, HIGH_UTILIZATION_THRESHOLD)
    assert metrics["onset_s"] == 0.004
    assert metrics["sample_count"] == 4
    assert np.isclose(metrics["duration_s"], 4 * 0.004)
    assert np.isclose(metrics["fraction"], 4 / 6)
