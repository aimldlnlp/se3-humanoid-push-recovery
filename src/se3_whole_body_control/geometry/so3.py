"""Numerically stable SO(3) exponential and logarithm maps.

Vectors use the right-handed xyz convention. All matrices are active rotation
matrices mapping coordinates from a source frame into a target frame.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-10


def _as_vector(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"expected a 3-vector, got {a.shape}")
    return a


def hat_so3(omega: np.ndarray) -> np.ndarray:
    """Return the skew matrix satisfying ``hat(w) @ x == cross(w, x)``."""

    x, y, z = _as_vector(omega)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee_so3(matrix: np.ndarray) -> np.ndarray:
    """Return the vector represented by a 3x3 skew matrix."""

    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {m.shape}")
    return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=float)


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Compute ``exp(hat(omega))`` with a small-angle Taylor branch."""

    w = _as_vector(omega)
    theta2 = float(w @ w)
    W = hat_so3(w)
    if theta2 < 1e-14:
        # sin(theta)/theta and (1-cos(theta))/theta^2 through second order.
        A = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        B = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        theta = np.sqrt(theta2)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta2
    return np.eye(3) + A * W + B * (W @ W)


def _near_pi_axis(R: np.ndarray) -> np.ndarray:
    """Recover a stable rotation axis for an angle close to pi."""

    diag = np.maximum(np.diag(R), -1.0)
    idx = int(np.argmax(diag))
    axis = np.zeros(3)
    denom = np.sqrt(max(1.0 + diag[idx], _EPS))
    axis[idx] = denom
    other = [(idx + 1) % 3, (idx + 2) % 3]
    axis[other[0]] = (R[other[0], idx] + R[idx, other[0]]) / (2.0 * denom)
    axis[other[1]] = (R[other[1], idx] + R[idx, other[1]]) / (2.0 * denom)
    n = np.linalg.norm(axis)
    if n < _EPS:
        _, _, vh = np.linalg.svd(R - np.eye(3))
        axis = vh[-1]
        n = np.linalg.norm(axis)
    axis /= max(n, _EPS)
    skew = vee_so3(R - R.T)
    if np.linalg.norm(skew) > _EPS and axis @ skew < 0:
        axis = -axis
    return axis


def log_so3(R: np.ndarray) -> np.ndarray:
    """Compute the principal rotation vector for a proper rotation matrix."""

    M = np.asarray(R, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {M.shape}")
    # Project tiny numerical drift to SO(3), without changing normal inputs.
    u, _, vh = np.linalg.svd(M)
    M = u @ vh
    if np.linalg.det(M) < 0:
        u[:, -1] *= -1
        M = u @ vh
    skew = vee_so3(M - M.T)
    sin_theta = 0.5 * np.linalg.norm(skew)
    cos_theta = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arctan2(sin_theta, cos_theta))
    if theta < 1e-8:
        return 0.5 * skew
    if np.pi - theta < 1e-6:
        return theta * _near_pi_axis(M)
    return (theta / (2.0 * np.sin(theta))) * skew
