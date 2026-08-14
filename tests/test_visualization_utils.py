import numpy as np

from se3_whole_body_control.visualization.plots import (
    _active_support_hull,
    _recovery_envelope,
    _recovery_grid,
    convex_hull_2d,
)


def test_convex_hull_returns_double_support_boundary():
    left = np.array([[-0.25, -0.12], [0.05, -0.12], [0.05, 0.12], [-0.25, 0.12]])
    right = left + np.array([0.45, 0.0])
    hull = _active_support_hull(np.stack([left, right]), np.array([True, True]))

    assert hull.shape == (4, 2)
    assert np.allclose(hull.min(axis=0), [-0.25, -0.12])
    assert np.allclose(hull.max(axis=0), [0.50, 0.12])


def test_recovery_grid_and_envelope_preserve_measured_cells():
    rows = [
        {"controller": "pd", "push_magnitude_N": 20, "push_direction_deg": 0, "success": "True"},
        {"controller": "pd", "push_magnitude_N": 40, "push_direction_deg": 0, "success": "False"},
        {"controller": "pd", "push_magnitude_N": 20, "push_direction_deg": 90, "success": 1},
        {"controller": "se3_wbc", "push_magnitude_N": 40, "push_direction_deg": 0, "success": True},
    ]

    magnitudes, directions, grid = _recovery_grid(rows, "pd")
    assert np.array_equal(magnitudes, [20.0, 40.0])
    assert np.array_equal(directions, [0.0, 90.0])
    assert np.array_equal(grid, [[1.0, 1.0], [0.0, np.nan]], equal_nan=True)

    envelope = _recovery_envelope(rows, "pd", directions)
    assert np.array_equal(envelope, [20.0, 20.0], equal_nan=True)


def test_recovery_envelope_leaves_unmeasured_direction_nan():
    rows = [{"controller": "pd", "push_magnitude_N": 20, "push_direction_deg": 0, "success": True}]
    envelope = _recovery_envelope(rows, "pd", np.array([0.0, 45.0]))

    assert envelope[0] == 20.0
    assert np.isnan(envelope[1])
