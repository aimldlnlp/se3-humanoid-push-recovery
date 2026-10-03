import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT, load_configs, make_model
from reaching_hold_audit import foot_geometry


def test_foot_geometry_uses_world_pose_and_initial_reference():
    model = make_model(load_configs(ROOT, robot_name='unitree_g1'))
    reference = [model.body_pose(name).copy() for name in ('left_foot', 'right_foot')]
    rows = foot_geometry(model, reference)
    assert all(row['body_xy_drift_m']==0 for row in rows)
    for foot, row in enumerate(rows):
        corners = model.support_vertices_local[foot] @ reference[foot][:3, :3].T + reference[foot][:3, 3]
        np.testing.assert_allclose(row['sole_corner_max_z_m'], corners[:, 2].max())
        reference[foot][0, 3] += .01
    changed = foot_geometry(model, reference)
    np.testing.assert_allclose([row['body_xy_drift_m'] for row in changed], .01)
    np.testing.assert_allclose([row['sole_tilt_rad'] for row in changed], [row['sole_tilt_rad'] for row in rows])
