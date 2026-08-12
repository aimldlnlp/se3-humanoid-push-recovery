import numpy as np

from se3_whole_body_control.control.tasks import pose_task_acceleration
from se3_whole_body_control.geometry.se3 import adjoint_se3, exp_se3, inverse_se3, log_se3


def test_right_invariant_spatial_error_is_equivariant_under_global_frame_change():
    desired = exp_se3(np.array([0.1, -0.05, 0.2, 0.08, -0.03, 0.04]))
    current = desired @ exp_se3(np.array([0.03, -0.02, 0.01, 0.02, 0.01, -0.015]))
    global_frame = exp_se3(np.array([0.7, -0.4, 0.2, -0.12, 0.09, 0.05]))
    error = log_se3(current @ inverse_se3(desired))
    transformed_error = log_se3((global_frame @ current) @ inverse_se3(global_frame @ desired))
    np.testing.assert_allclose(transformed_error, adjoint_se3(global_frame) @ error, atol=2e-8, rtol=2e-8)


def test_spatial_twist_adjoint_matches_conjugation():
    transform = exp_se3(np.array([0.2, 0.1, -0.1, 0.05, -0.04, 0.07]))
    twist = np.array([0.3, -0.2, 0.1, 0.2, 0.1, -0.15])
    conjugated = transform @ np.array([[0, -twist[5], twist[4], twist[0]], [twist[5], 0, -twist[3], twist[1]], [-twist[4], twist[3], 0, twist[2]], [0, 0, 0, 0]]) @ inverse_se3(transform)
    expected = adjoint_se3(transform) @ twist
    np.testing.assert_allclose(np.r_[conjugated[:3, 3], [conjugated[2, 1], conjugated[0, 2], conjugated[1, 0]]], expected, atol=1e-10)


def test_pose_task_global_frame_equivariance():
    """Exercise the production task output, not only isolated SE(3) identities."""
    desired = exp_se3(np.array([0.1, -0.05, 0.2, 0.08, -0.03, 0.04]))
    current = exp_se3(np.array([0.14, -0.08, 0.24, 0.11, -0.02, 0.07]))
    jacobian = np.array(
        [[0.2, -0.1, 0.3, 0.4], [0.1, 0.2, -0.2, 0.1], [-0.3, 0.1, 0.2, 0.5],
         [0.4, 0.3, -0.1, 0.2], [-0.2, 0.1, 0.5, -0.3], [0.1, -0.4, 0.2, 0.2]],
        dtype=float,
    )
    qvel = np.array([0.3, -0.2, 0.1, 0.4])
    global_frame = exp_se3(np.array([0.7, -0.4, 0.2, -0.12, 0.09, 0.05]))
    args = (jacobian, qvel, 180.0, 28.0, 220.0, 32.0)
    J, target, error = pose_task_acceleration(current, desired, *args)
    J_changed, target_changed, error_changed = pose_task_acceleration(
        global_frame @ current,
        global_frame @ desired,
        adjoint_se3(global_frame) @ jacobian,
        qvel,
        180.0,
        28.0,
        220.0,
        32.0,
    )
    Ad = adjoint_se3(global_frame)
    np.testing.assert_allclose(J_changed, Ad @ J, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(error_changed, Ad @ error, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(J_changed @ qvel, Ad @ (J @ qvel), atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(target_changed, Ad @ target, atol=2e-8, rtol=2e-8)
