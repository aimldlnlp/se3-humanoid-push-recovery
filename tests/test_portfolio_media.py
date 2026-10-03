import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from render_portfolio_reaching import metrics


def test_portfolio_metrics_use_final_goal_not_moving_reference():
    arrays = dict(reach_point_world=np.array([[0,0,0],[.01,0,0]]),
                  reach_goal_world=np.array([.02,0,0]),
                  reach_reference_world=np.zeros((2,3)),
                  torso_rotation_error_rad=np.array([0,np.pi/180]),
                  torque_utilization=np.array([.1,.2]))
    error, torso, torque = metrics(arrays)
    np.testing.assert_allclose(error,[20,10])
    np.testing.assert_allclose(torso,[0,1])
    np.testing.assert_array_equal(torque,arrays['torque_utilization'])
