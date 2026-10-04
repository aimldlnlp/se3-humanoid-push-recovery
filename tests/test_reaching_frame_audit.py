from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('mujoco')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT, load_configs, make_model
from reaching_frame_audit import measure


@pytest.mark.parametrize('body', ['torso', 'pelvis'])
def test_real_body_jacobian_frame_and_acceleration_conventions(body):
    model = make_model(load_configs(ROOT, robot_name='unitree_g1'))
    qpos = model.data.qpos.copy()
    qvel = np.linspace(-0.2, 0.3, model.nv)
    model.reset(qpos, qvel)
    desired = model.body_pose(body)
    result = measure(model, body, desired, (180,28,220,32), np.zeros(model.nv))
    assert result['raw_jacobian_spatial_velocity_error'] > 1e-3
    assert result['raw_target_origin_shift_change'] > 1e-3
    assert result['spatial_target_adjoint_error'] < 1e-8
    assert result['diagnostic_target_difference_norm'] > 1e-3
    np.testing.assert_array_equal(model.data.qpos, qpos)
    np.testing.assert_array_equal(model.data.qvel, qvel)
