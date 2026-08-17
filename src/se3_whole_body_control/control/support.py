"""Small geometry helpers used by contact-mode controllers.

This module intentionally has no MuJoCo dependency.  It operates on the
world-frame support vertices supplied by the robot adapter, which keeps the
step decision testable and independent from the visualization layer.
"""

from __future__ import annotations

import numpy as np


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise monotonic-chain hull for finite XY points."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= 1:
        return points.copy()
    unique = np.unique(points, axis=0)
    if len(unique) <= 2:
        return unique

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # ``np.cross`` no longer returns a scalar for 2-D vectors in recent
        # NumPy releases.  The monotonic-chain orientation test is the scalar
        # determinant by definition, so keep it explicit and version-stable.
        oa = a - o
        ob = b - o
        return float(oa[0] * ob[1] - oa[1] * ob[0])

    lower: list[np.ndarray] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def signed_support_margin(point_xy: np.ndarray, hull_xy: np.ndarray) -> float:
    """Return the signed distance to a convex hull, positive inside."""
    point = np.asarray(point_xy, dtype=float).reshape(2)
    hull = np.asarray(hull_xy, dtype=float).reshape(-1, 2)
    if len(hull) < 3 or not np.all(np.isfinite(hull)):
        return float("nan")
    area = 0.5 * np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - hull[:, 1] * np.roll(hull[:, 0], -1))
    if area < 0.0:
        hull = hull[::-1]
    edge = np.roll(hull, -1, axis=0) - hull
    normal = np.column_stack((-edge[:, 1], edge[:, 0]))
    lengths = np.linalg.norm(edge, axis=1)
    distance = np.sum((point[None, :] - hull) * normal, axis=1) / np.maximum(lengths, 1e-12)
    return float(np.min(distance))


def normalize_xy(vector: np.ndarray, minimum: float = 1e-12) -> np.ndarray:
    """Return a unit XY direction, or zero when the input is too small."""
    value = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > minimum else np.zeros(2, dtype=float)
