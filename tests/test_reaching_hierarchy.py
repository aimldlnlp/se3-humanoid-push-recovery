import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_hierarchy import solve_level, HierarchicalController
from common import ROOT, load_configs, make_model, prepare_paired_initial_condition
from se3_whole_body_control.config import load_yaml


def test_priority_locks_attained_not_unreachable_target():
    A = np.eye(2)
    low, high = np.array([-1., -1.]), np.array([1., 1.])
    x = solve_level(np.diag([2., 0.]), np.array([-4., 0.]), A, low, high, {})
    assert abs(x[0]-1) < 1e-4
    final = solve_level(2*np.eye(2), np.array([4., -2.]),
                        np.vstack([A, [1., 0.]]), np.r_[low, x[0]-1e-5],
                        np.r_[high, x[0]+1e-5], {})
    assert abs(final[0]-x[0]) < 1e-4
    assert abs(final[1]-1) < 1e-4


def test_hierarchy_retains_original_physical_rows_and_solves():
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    model = make_model(cfg)
    model.reset(initial.qpos, initial.qvel)
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    settings['weight'] = 3000
    settings['contact_velocity_damping_s_inv'] = 20
    controller = HierarchicalController(model, cfg['controller'], settings)
    result = controller.solve()
    assert result.success, result.message
    before, after = controller.baseline_problem, controller.problem
    n = before[2].shape[0]
    np.testing.assert_array_equal(before[2].toarray(), after[2][:n].toarray())
    np.testing.assert_array_equal(before[3], after[3][:n])
    np.testing.assert_array_equal(before[4], after[4][:n])
    assert [v['name'] for v in controller.hierarchy_levels] == ['contact_slack', 'balance', 'reach']
    assert result.diagnostics['constraint_budget_ratio'] <= 1
