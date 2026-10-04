import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mujoco')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_pose_origin import body_origin_target
from se3_whole_body_control.control.tasks import pose_task_acceleration
from se3_whole_body_control.geometry.se3 import exp_se3
from se3_whole_body_control.geometry.so3 import hat_so3


@pytest.mark.parametrize('frame', [np.eye(4), exp_se3(np.array([.7,-.4,.2,-.12,.09,.05]))])
def test_corrected_pose_objective_and_acceleration_are_frame_consistent(frame):
    current = exp_se3(np.array([.14,-.08,.9,.11,-.02,.07]))
    desired = exp_se3(np.array([.1,-.05,.85,.08,-.03,.04]))
    J = np.arange(24,dtype=float).reshape(6,4)/20-.5
    qvel, qdd = np.array([.3,-.2,.1,.4]), np.array([-.1,.2,.5,-.3])
    bias = np.array([.02,-.03,.04,.05,-.02,.01])
    gains = (180,28,220,32)
    target = body_origin_target(current,desired,J,qvel,bias,gains)
    rotation = np.zeros((6,6))
    rotation[:3,:3] = rotation[3:,3:] = frame[:3,:3]
    changed_J = rotation@J
    changed = body_origin_target(frame@current,frame@desired,changed_J,qvel,rotation@bias,gains)
    np.testing.assert_allclose(changed,rotation@target,atol=2e-8,rtol=2e-8)
    # The complete weighted pose quadratic, not only its target, is invariant.
    np.testing.assert_allclose(changed_J.T@changed_J,J.T@J,atol=2e-10)
    np.testing.assert_allclose(changed_J.T@changed,J.T@target,atol=2e-8)
    np.testing.assert_allclose(changed@changed,target@target,rtol=2e-8)
    H = np.eye(6)
    H[:3,3:] = hat_so3(current[:3,3])
    velocity = J@qvel
    offset = np.r_[np.cross(velocity[:3],velocity[3:]),np.zeros(3)]
    _, spatial_target, _ = pose_task_acceleration(current,desired,H@J,qvel,*gains)
    residual = H@(J@qdd+bias)+offset-spatial_target
    np.testing.assert_allclose(residual,H@(J@qdd-target),atol=2e-8)
