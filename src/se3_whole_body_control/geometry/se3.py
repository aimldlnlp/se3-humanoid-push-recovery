"""SE(3) maps using the twist convention ``[linear, angular]``."""

from __future__ import annotations

import numpy as np

from .so3 import exp_so3, hat_so3, log_so3, vee_so3


def _xi(xi: np.ndarray) -> np.ndarray:
    x = np.asarray(xi, dtype=float).reshape(-1)
    if x.shape != (6,):
        raise ValueError(f"expected a 6-vector, got {x.shape}")
    return x


def _T(T: np.ndarray) -> np.ndarray:
    X = np.asarray(T, dtype=float)
    if X.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {X.shape}")
    return X


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """Map ``[v, w]`` to ``[[hat(w), v], [0, 0]]``."""

    x = _xi(xi)
    out = np.zeros((4, 4), dtype=float)
    out[:3, :3] = hat_so3(x[3:])
    out[:3, 3] = x[:3]
    return out


def vee_se3(matrix: np.ndarray) -> np.ndarray:
    M = np.asarray(matrix, dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {M.shape}")
    return np.r_[M[:3, 3], vee_so3(M[:3, :3])]


def _V_matrix(w: np.ndarray) -> np.ndarray:
    W = hat_so3(w)
    theta2 = float(w @ w)
    if theta2 < 1e-14:
        A = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        B = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    else:
        theta = np.sqrt(theta2)
        A = (1.0 - np.cos(theta)) / theta2
        B = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) + A * W + B * (W @ W)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    x = _xi(xi)
    R = exp_so3(x[3:])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = _V_matrix(x[3:]) @ x[:3]
    return T


def _V_inv_matrix(w: np.ndarray) -> np.ndarray:
    W = hat_so3(w)
    theta2 = float(w @ w)
    if theta2 < 1e-12:
        A = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    else:
        theta = np.sqrt(theta2)
        half = 0.5 * theta
        A = (1.0 - half / np.tan(half)) / theta2
    return np.eye(3) - 0.5 * W + A * (W @ W)


def log_se3(T: np.ndarray) -> np.ndarray:
    X = _T(T)
    w = log_so3(X[:3, :3])
    v = _V_inv_matrix(w) @ X[:3, 3]
    return np.r_[v, w]


def compose_se3(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    return _T(T_ab) @ _T(T_bc)


def inverse_se3(T: np.ndarray) -> np.ndarray:
    X = _T(T)
    out = np.eye(4)
    out[:3, :3] = X[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ X[:3, 3]
    return out


def adjoint_se3(T: np.ndarray) -> np.ndarray:
    """Return the adjoint for twists ordered as ``[linear, angular]``."""

    X = _T(T)
    R = X[:3, :3]
    p = X[:3, 3]
    A = np.zeros((6, 6), dtype=float)
    A[:3, :3] = R
    A[:3, 3:] = hat_so3(p) @ R
    A[3:, 3:] = R
    return A
