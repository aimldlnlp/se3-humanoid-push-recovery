import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_hierarchy import solve_level, solve_hierarchy, HierarchicalController
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
    assert [v['name'] for v in controller.hierarchy_levels] == ['contact_slack', 'balance', 'reach', 'remaining']
    assert result.diagnostics['constraint_budget_ratio'] <= 1
    reconstructed_P = sum(2*matrix.T@matrix for name, matrix, target in controller.levels)
    reconstructed_q = sum(-2*matrix.T@target for name, matrix, target in controller.levels)
    np.testing.assert_allclose(reconstructed_P, before[0], atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(reconstructed_q, before[1], atol=1e-8, rtol=1e-12)
    for level in controller.hierarchy_levels:
        np.testing.assert_allclose(level['matrix']@controller.solution, level['output'], atol=1e-7, rtol=1e-7)


def test_nullspace_preserves_conflicting_priority_and_redundant_equalities():
    A = np.array([[1., 1., 0.], [2., 2., 0.], [1., 0., 0.], [0., 0., 1.]])
    low, high = np.array([1., 2., -1., -1.]), np.array([1., 2., 1., 1.])
    levels = [('high', np.array([[1., 0., 0.]]), np.array([2.])),
              ('low', np.eye(3), np.array([-1., 2., 1.]))]
    x, records = solve_hierarchy(levels, A, low, high, {})
    np.testing.assert_allclose(x, [1., 0., 1.], atol=1e-6)
    np.testing.assert_allclose(records[0]['matrix']@x, records[0]['output'], atol=1e-10)
