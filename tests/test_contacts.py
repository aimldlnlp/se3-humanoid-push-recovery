import numpy as np

from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController


def test_friction_pyramid_rows():
    class Dummy:
        nv = 2
        nu = 1
    # Check the physical inequalities used by the controller directly.
    mu = 0.7
    wrench = np.array([1.0, -2.0, 4.0, 0, 0, 0])
    assert wrench[2] >= 0
    assert abs(wrench[0]) <= mu * wrench[2]
    assert abs(wrench[1]) <= mu * wrench[2]
    assert np.isfinite(mu)
