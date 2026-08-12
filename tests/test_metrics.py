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
    assert result.recovery_time_s >= 1.0


def test_serialization(tmp_path):
    log = TrialLog.empty()
    for i in range(2):
        log.append(time_s=float(i), torso_error=[0] * 6, pelvis_error=[0] * 6, com_world=[0, 0, 1], torso_position=[0, 0, 1], torso_rotation_error_rad=0, contact_left=True, contact_right=True, contact_wrench=[0] * 12, control=[0], qp_status="solved", qp_solve_time_s=0.001, push_force=[0, 0, 0], joint_velocity_norm=0, torso_angular_velocity_norm=0, torso_height_m=1, torque_abs_max_Nm=0, qp_success=True, friction_margin=1, qp_message="", qp_slack_norm=0)
    path = tmp_path / "trial.npz"
    save_trial_npz(log, path, {"seed": 0})
    assert path.exists()
    data = np.load(path)
    assert data["time_s"].shape == (2,)
