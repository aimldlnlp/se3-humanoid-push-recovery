import numpy as np

from se3_whole_body_control.evaluation.metrics import TrialLog, save_trial_npz
from se3_whole_body_control.evaluation.recovery import RecoveryConfig, classify_recovery


def test_recovery_success():
    t = np.arange(0, 1.0, 0.01)
    zeros = np.zeros_like(t)
    result = classify_recovery(t, zeros, zeros, zeros, np.ones_like(t, dtype=bool), np.ones_like(t, dtype=bool), np.ones_like(t), zeros, np.ones_like(t, dtype=bool), np.ones_like(t), RecoveryConfig())
    assert result.success
    assert result.failure_reason is None


def test_recovery_contact_loss():
    t = np.arange(0, 1.0, 0.01)
    zeros = np.zeros_like(t)
    result = classify_recovery(t, zeros, zeros, zeros, np.zeros_like(t, dtype=bool), np.ones_like(t, dtype=bool), np.ones_like(t), zeros, np.ones_like(t, dtype=bool), np.ones_like(t), RecoveryConfig())
    assert not result.success
    assert result.failure_reason == "CONTACT_LOSS"


def test_recovery_time_starts_after_disturbance():
    t = np.arange(0, 2.0, 0.01)
    zeros = np.zeros_like(t)
    result = classify_recovery(t, zeros, zeros, zeros, np.ones_like(t, dtype=bool), np.ones_like(t, dtype=bool), np.ones_like(t), zeros, np.ones_like(t, dtype=bool), np.ones_like(t), RecoveryConfig(), recovery_start_s=1.0)
    assert result.success
    assert result.recovered_at_s >= 1.0
    assert np.isclose(result.recovery_latency_s, result.recovered_at_s - 1.0)


def test_recovery_before_timeout_remains_success():
    t = np.arange(0, 2.0, 0.01)
    zeros = np.zeros_like(t)
    result = classify_recovery(
        t, zeros, zeros, zeros,
        np.ones_like(t, dtype=bool), np.ones_like(t, dtype=bool),
        np.ones_like(t), zeros, np.ones_like(t, dtype=bool), np.ones_like(t),
        RecoveryConfig(timeout_s=1.0), push_end_s=0.2,
    )
    assert result.success
    assert result.failure_reason is None
    assert result.recovered_at_s is not None
    assert result.recovery_latency_s is not None


def test_serialization(tmp_path):
    log = TrialLog.empty()
    for i in range(2):
        log.append(time_s=float(i), torso_error=[0] * 6, pelvis_error=[0] * 6, com_world=[0, 0, 1], torso_position=[0, 0, 1], torso_rotation_error_rad=0, contact_left=True, contact_right=True, predicted_contact_wrench=[0] * 12, actual_contact_wrench=[0] * 12, actual_friction_utilization=[0, 0], foot_tangent_velocity=[0, 0], foot_xy_displacement=[0, 0], foot_xy_world=[0, 0, 0, 0], foot_support_vertices_world=[0] * 16, foot_cop_world=[0] * 4, control=[0], qp_status="solved", qp_solve_time_s=0.001, push_force=[0, 0, 0], joint_velocity_norm=0, torso_angular_velocity_norm=0, torso_height_m=1, torque_abs_max_Nm=0, torque_utilization=0, qp_success=True, predicted_friction_margin=1, actual_friction_margin=1, qp_message="", qp_slack_norm=0, dynamics_residual_norm=0, contact_acceleration_residual_norm=0)
    path = tmp_path / "trial.npz"
    save_trial_npz(log, path, {"seed": 0})
    assert path.exists()
    data = np.load(path)
    assert data["time_s"].shape == (2,)


def test_physical_slip_detection_uses_measured_contact_metrics():
    t = np.arange(0, 1.0, 0.01)
    zeros = np.zeros_like(t)
    utilization = np.zeros((len(t), 2)); utilization[50:60] = 1.2
    velocity = np.zeros((len(t), 2)); displacement = np.zeros((len(t), 2))
    result = classify_recovery(
        t, zeros, zeros, zeros, np.ones_like(t, dtype=bool), np.ones_like(t, dtype=bool),
        np.ones_like(t), zeros, np.ones_like(t, dtype=bool), np.ones_like(t), RecoveryConfig(),
        actual_friction_utilization=utilization, foot_tangent_velocity=velocity,
        foot_xy_displacement=displacement,
    )
    assert not result.success
    assert result.failure_reason == "SLIP"
