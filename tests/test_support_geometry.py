import numpy as np

from se3_whole_body_control.control.support import convex_hull_2d, signed_support_margin


def test_signed_support_margin_is_positive_inside_and_negative_outside():
    hull = convex_hull_2d(np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]))
    assert signed_support_margin([0.0, 0.0], hull) > 0.0
    assert signed_support_margin([1.2, 0.0], hull) < 0.0


def test_support_hull_deduplicates_foot_vertices():
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]])
    hull = convex_hull_2d(points)
    assert hull.shape == (4, 2)
    np.testing.assert_allclose(hull.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(hull.max(axis=0), [1.0, 1.0])
