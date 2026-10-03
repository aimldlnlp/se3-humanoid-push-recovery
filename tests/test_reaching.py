from pathlib import Path

import numpy as np
import pytest

from se3_whole_body_control.config import load_configs, resolve_model_path
from se3_whole_body_control.control.tasks import ReachTask, quintic_reference
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
from se3_whole_body_control.dynamics.humanoid import HumanoidModel


def test_quintic_rest_endpoints_and_derivatives():
    start, goal = np.zeros(3), np.array([0.1, -0.03, 0.04])
    for t, endpoint in [(-1, start), (0, start), (2, goal), (3, goal)]:
        p, v, a = quintic_reference(start, goal, t, 2)
        np.testing.assert_allclose(p, endpoint)
        np.testing.assert_allclose(v, 0)
        np.testing.assert_allclose(a, 0)
    dt = 1e-5
    p, v, a = quintic_reference(start, goal, 0.7, 2)
    pp, vp, _ = quintic_reference(start, goal, 0.7+dt, 2)
    pm, vm, _ = quintic_reference(start, goal, 0.7-dt, 2)
    np.testing.assert_allclose((pp-pm)/(2*dt), v, atol=1e-9)
    np.testing.assert_allclose((vp-vm)/(2*dt), a, atol=1e-9)
    with pytest.raises(ValueError):
        quintic_reference(start, goal, 0, 0)


def test_gif_preserves_physical_duration(tmp_path):
    from PIL import Image
    from se3_whole_body_control.visualization.video import make_gif
    for i in range(6):
        Image.new('RGB', (32, 32), (i*40, 0, 0)).save(tmp_path/f'frame_{i:06d}.png')
    output = make_gif(tmp_path, tmp_path/'test.gif', fps=30)
    with Image.open(output) as gif:
        delays = []
        for i in range(gif.n_frames):
            gif.seek(i)
            delays.append(gif.info.get('duration', 0))
        assert gif.n_frames == 6 and all(d > 0 for d in delays)
        assert abs(sum(delays)/1000 - 6/30) <= 0.02


def test_production_reach_point_jacobian_bias_and_qp_objective():
    mujoco = pytest.importorskip("mujoco")
    cfg = load_configs(Path(__file__).resolve().parents[1], robot_name="unitree_g1")
    model = HumanoidModel(resolve_model_path(cfg), robot_config=cfg['robot'])
    body, local = 'right_wrist_yaw_link', np.array([0.08, -0.01, 0.02])
    rng = np.random.default_rng(4)
    model.data.qvel[:] = rng.normal(0, 0.1, model.nv)
    mujoco.mj_forward(model.model, model.data)
    q, vel = model.data.qpos.copy(), model.data.qvel.copy()
    p, J, bias = model.attached_point_kinematics(body, local)
    samples = []
    dt = 1e-6
    for sign in (-1, 1):
        next_q = q.copy()
        mujoco.mj_integratePos(model.model, next_q, vel, sign*dt)
        model.reset(next_q, vel)
        samples.append(model.attached_point_kinematics(body, local))
    model.reset(q, vel)
    np.testing.assert_allclose((samples[1][0]-samples[0][0])/(2*dt), J@vel, atol=1e-8)
    np.testing.assert_allclose((samples[1][1]-samples[0][1])@vel/(2*dt), bias, atol=1e-8)
    task = ReachTask(body, local, p+np.array([0.03, 0, 0]), np.zeros(3), np.zeros(3))
    controller = WholeBodyQPController(model, cfg['controller'])
    P0, q0, A0, l0, u0, *_ = controller._build_problem()
    controller.reach_task = task
    P1, q1, A1, l1, u1, *_ = controller._build_problem()
    assert np.linalg.norm(P1-P0) > 0 and np.linalg.norm(q1-q0) > 0
    np.testing.assert_array_equal(A0.toarray(), A1.toarray())
    np.testing.assert_array_equal(l0, l1)
    np.testing.assert_array_equal(u0, u1)
    task.weight *= 10
    P2, q2, A2, l2, u2, *_ = controller._build_problem()
    np.testing.assert_allclose(P2-P0, 10*(P1-P0), atol=1e-9)
    np.testing.assert_allclose(q2-q0, 10*(q1-q0), atol=1e-9)
    np.testing.assert_array_equal(A1.toarray(), A2.toarray())
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(u1, u2)
    for key in ('qp_posture_weight', 'qp_nominal_torque_weight'):
        original = controller.cfg[key]
        controller.cfg[key] = 0
        _, _, A3, l3, u3, *_ = controller._build_problem()
        np.testing.assert_array_equal(A2.toarray(), A3.toarray())
        np.testing.assert_array_equal(l2, l3)
        np.testing.assert_array_equal(u2, u3)
        controller.cfg[key] = original
    result = controller.solve()
    assert result.success, result.message
    assert result.diagnostics['constraint_budget_ratio'] <= 1
    np.testing.assert_array_equal(model.data.qpos, q)
    task.position_world = np.array([np.nan, 0, 0])
    with pytest.raises(ValueError):
        task.acceleration_target(model)


def test_workspace_audit_catches_joint_failure_during_push(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from reaching_workspace import assess_trial
    times = np.arange(0, 5, .004)
    n = len(times)
    summary = {'combined_success': True, 'reach_success': True,
               'recovery': {'failure_reason': None},
               'provenance': {'com_reference_world': [0, 0, 1]}}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    arrays = dict(time_s=times, torso_rotation_error_rad=np.zeros(n),
                  torso_angular_velocity_norm=np.zeros(n), com_world=np.tile([0,0,1], (n,1)),
                  contact_left_post_step=np.ones(n, bool), contact_right_post_step=np.ones(n, bool),
                  torso_height_m=np.ones(n), torque_abs_max_Nm=np.zeros(n), qp_success=np.ones(n, bool),
                  actual_friction_margin=np.ones(n), actual_friction_utilization_post_step=np.zeros((n,2)),
                  foot_tangent_velocity_post_step=np.zeros((n,2)), foot_xy_displacement_post_step=np.zeros((n,2)),
                  torque_utilization=np.zeros(n), joint_limit_violation=np.zeros(n, bool), numerical_valid=np.ones(n, bool))
    np.savez(tmp_path/'trajectory.npz', **arrays)
    assert assess_trial(tmp_path)[1:] == (True, 'PASS')
    arrays['joint_limit_violation'][int(1.55/.004)] = True
    np.savez(tmp_path/'trajectory.npz', **arrays)
    assert assess_trial(tmp_path)[1:] == (False, 'JOINT_LIMIT')


def test_pd_reach_reference_is_feasible_and_does_not_change_plant(monkeypatch):
    pytest.importorskip('mujoco')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from common import ROOT, make_model, prepare_paired_initial_condition
    from reaching_pd import ReachingPDController
    from se3_whole_body_control.config import load_yaml
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    model = make_model(cfg)
    model.reset(initial.qpos, initial.qvel)
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    settings['target_offset_world_m'] = [.05, 0, 0]
    qpos, qvel = model.data.qpos.copy(), model.data.qvel.copy()
    controller = ReachingPDController(model,cfg['controller'],settings)
    assert controller.ik_max_error_m < .001
    np.testing.assert_array_equal(model.data.qpos,qpos)
    np.testing.assert_array_equal(model.data.qvel,qvel)
    np.testing.assert_allclose(controller.joint_references[0], initial.joint_reference, atol=1e-8)


def test_tracking_calibration_rejects_unpaired_reference(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from reaching_tracking_study import verify_pair
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    new.mkdir()
    arrays = dict(time_s=np.arange(3), reach_reference_world=np.zeros((3, 3)),
                  reach_goal_world=np.zeros(3), push_force=np.zeros((3, 3)),
                  qpos_history=np.zeros((3, 2)), qvel_history=np.zeros((3, 2)))
    np.savez(old/'trajectory.npz', **arrays)
    np.savez(new/'trajectory.npz', **arrays)
    verify_pair(old, new)
    arrays['reach_reference_world'][1, 0] = .001
    np.savez(new/'trajectory.npz', **arrays)
    with pytest.raises(AssertionError):
        verify_pair(old, new)


def test_tracking_plots_include_calibration_and_control_views(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from reaching_tracking_study import plot
    for stage in ('validation', 'validation_control'):
        root = tmp_path/stage
        root.mkdir()
        row = dict(trial_id='front_12.5cm_w300_moving_70N', stage=stage,
                   hold_max_error_mm=10, combined_success=True, failure_reason='PASS', reach_weight=300)
        rows = [row]
        if stage == 'validation':
            rows.append({**row, 'stage': 'calibration'})
        plot(root, rows)
        assert (root/'tracking_results.png').stat().st_size > 1000


def test_objective_audit_records_the_production_qp_without_changing_it(monkeypatch):
    pytest.importorskip('mujoco')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from common import ROOT, make_model
    from reaching_conflict_audit import ObjectiveAuditController, OBJECTIVES
    from reach_and_balance import ReachingController
    from se3_whole_body_control.config import load_yaml
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    model = make_model(cfg)
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    audit = ObjectiveAuditController(model, cfg['controller'], settings)
    ordinary = ReachingController(model, cfg['controller'], settings)
    actual, expected = audit._build_problem(), ordinary._build_problem()
    assert len(audit.objectives)==len(OBJECTIVES)
    for index in (0, 1, 3, 4):
        np.testing.assert_array_equal(actual[index], expected[index])
    np.testing.assert_array_equal(actual[2].toarray(), expected[2].toarray())
    reconstructed = np.eye(audit.nx)*1e-9
    linear = np.zeros(audit.nx)
    for A, b, weight in audit.objectives:
        reconstructed += 2*weight*A.T@A
        linear -= 2*weight*A.T@b
    np.testing.assert_allclose(reconstructed, actual[0], atol=1e-9)
    np.testing.assert_allclose(linear, actual[1], atol=1e-9)


def test_balance_guard_reduces_immediately_and_restores_only_after_stable_dwell(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from reach_and_balance import ReachBalanceGuard
    guard = ReachBalanceGuard(3000, 100, .25)
    assert guard.update(0, True)==3000
    assert guard.update(1, False)==100
    assert guard.update(1.004, True)==100
    assert guard.update(1.204, True)==100
    assert guard.update(1.254, True)==100
    assert guard.update(1.379, True)==pytest.approx(1550)
    assert guard.update(1.380, False)==100
    assert guard.update(1.4, True)==100
    assert guard.update(1.9, True)==pytest.approx(3000)
    assert guard.update(2, True)==3000


def test_balance_guard_uses_measured_state_without_force_oracle(monkeypatch):
    pytest.importorskip('mujoco')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    from common import ROOT, make_model, recovery_config
    from reach_and_balance import ReachingController
    from se3_whole_body_control.config import load_yaml
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    model = make_model(cfg)
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    settings['weight'] = 3000
    recovery = recovery_config(cfg)
    settings['balance_guard'] = {key: getattr(recovery, key) for key in (
        'orientation_threshold_rad', 'angular_velocity_threshold_rad_s',
        'com_displacement_threshold_m', 'stable_duration_s')}
    settings['balance_guard']['minimum_weight'] = 100
    controller = ReachingController(model, cfg['controller'], settings)
    def forbidden():
        raise AssertionError('external-force oracle must remain disabled')
    monkeypatch.setattr(model, 'external_generalized_force', forbidden)
    model.data.qvel[3] = 1
    import mujoco
    mujoco.mj_forward(model.model, model.data)
    controller.solve()
    assert controller.guard_risks[-1] and controller.weights[-1]==100
    settings['balance_guard']['track_arm_posture_during_recovery'] = True
    controller = ReachingController(model, cfg['controller'], settings)
    original = controller.q_des.copy()
    model.data.qpos[model.joint_qpos_indices[controller.guard_arm]] += .03
    mujoco.mj_forward(model.model, model.data)
    controller.solve()
    other = np.setdiff1d(np.arange(model.nu), controller.guard_arm)
    np.testing.assert_array_equal(controller.q_des[other], original[other])
    np.testing.assert_array_equal(controller.pd_fallback.q_des[other], original[other])
    np.testing.assert_array_equal(controller.q_des[controller.guard_arm], model.joint_positions()[controller.guard_arm])
    np.testing.assert_array_equal(controller.pd_fallback.q_des[controller.guard_arm], controller.q_des[controller.guard_arm])
