import numpy as np

from se3_whole_body_control.disturbance.push import Push, active_push, push_force


def test_push_force_and_impulse():
    push = Push(120.0, np.pi / 2, 0.15, 2.0, "torso")
    assert np.isclose(push.impulse_Ns, 18.0)
    assert not active_push(push, 1.9)
    assert active_push(push, 2.05)
    np.testing.assert_allclose(push_force(push, 2.05), [0, 120, 0], atol=1e-10)
    np.testing.assert_allclose(push_force(push, 2.2), [0, 0, 0])
