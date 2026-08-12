import numpy as np

from se3_whole_body_control.geometry.so3 import exp_so3, hat_so3, log_so3, vee_so3


def test_hat_vee_roundtrip():
    w = np.array([0.2, -0.4, 0.7])
    np.testing.assert_allclose(vee_so3(hat_so3(w)), w)


def test_exp_is_rotation():
    R = exp_so3(np.array([0.2, -0.3, 0.4]))
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_exp_log_small_and_regular():
    for w in (np.zeros(3), np.array([1e-8, -2e-8, 3e-8]), np.array([0.4, -0.2, 0.7])):
        np.testing.assert_allclose(exp_so3(log_so3(exp_so3(w))), exp_so3(w), atol=1e-9)


def test_log_near_pi():
    w = (np.pi - 1e-7) * np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    np.testing.assert_allclose(exp_so3(log_so3(exp_so3(w))), exp_so3(w), atol=1e-6)
