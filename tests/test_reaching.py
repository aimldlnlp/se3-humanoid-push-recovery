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
    result = controller.solve()
    assert result.success, result.message
    assert result.diagnostics['constraint_budget_ratio'] <= 1
    np.testing.assert_array_equal(model.data.qpos, q)
    task.position_world = np.array([np.nan, 0, 0])
    with pytest.raises(ValueError):
        task.acceleration_target(model)
