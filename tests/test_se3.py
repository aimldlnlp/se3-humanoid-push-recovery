import numpy as np

from se3_whole_body_control.geometry.se3 import (
    adjoint_se3, compose_se3, exp_se3, hat_se3, inverse_se3, log_se3, vee_se3,
)


def test_hat_vee_roundtrip():
    xi = np.array([0.2, -0.1, 0.5, 0.3, -0.2, 0.1])
    np.testing.assert_allclose(vee_se3(hat_se3(xi)), xi)


def test_se3_exp_log_and_inverse():
    xi = np.array([0.4, -0.2, 0.1, 0.3, -0.1, 0.2])
    T = exp_se3(xi)
    np.testing.assert_allclose(exp_se3(log_se3(T)), T, atol=1e-9)
    np.testing.assert_allclose(compose_se3(T, inverse_se3(T)), np.eye(4), atol=1e-9)


def test_pure_translation():
    xi = np.array([1.0, 2.0, -0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(log_se3(exp_se3(xi)), xi, atol=1e-10)


def test_adjoint_identity():
    np.testing.assert_allclose(adjoint_se3(np.eye(4)), np.eye(6))
