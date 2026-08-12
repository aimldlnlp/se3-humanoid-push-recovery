from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from se3_whole_body_control.dynamics.humanoid import HumanoidModel
from se3_whole_body_control.control.tasks import com_jacobian
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController


def test_model_contacts_and_jacobians_are_finite():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")

    assert (model.nq, model.nv, model.nu) == (15, 14, 8)
    state = model.state()
    assert state.contact_left and state.contact_right
    assert state.torso_jacobian.shape == (6, model.nv)
    assert state.left_foot_jacobian.shape == (6, model.nv)
    assert state.right_foot_jacobian.shape == (6, model.nv)
    assert np.all(np.isfinite(model.mass_matrix()))
    assert np.all(np.isfinite(model.contact_bias_acceleration()))


def test_com_jacobian_matches_finite_difference():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")
    model.data.qvel[:] = np.linspace(-0.015, 0.015, model.nv)
    mujoco.mj_forward(model.model, model.data)
    dt = 1e-6
    qpos0 = model.data.qpos.copy()
    qvel0 = model.data.qvel.copy()
    com0 = model.center_of_mass().copy()
    qpos_next = model.data.qpos.copy()
    mujoco.mj_integratePos(model.model, qpos_next, model.data.qvel, dt)
    model.data.qpos[:] = qpos_next
    mujoco.mj_forward(model.model, model.data)
    com1 = model.center_of_mass().copy()
    model.data.qpos[:] = qpos0
    model.data.qvel[:] = qvel0
    mujoco.mj_forward(model.model, model.data)
    finite_difference = (com1 - com0) / dt
    np.testing.assert_allclose(finite_difference, com_jacobian(model) @ model.data.qvel, atol=2e-5, rtol=2e-4)


def test_actual_contact_reaction_is_separate_and_upward():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")
    contact = model.actual_contact_data()
    assert contact.contact_flags.all()
    assert contact.wrench_world.shape == (12,)
    assert np.all(contact.wrench_world[[2, 8]] > 0.0)


def test_local_push_application_point_is_rotated_into_world_wrench():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")
    qpos = model.qpos0.copy()
    qpos[3:7] = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    model.reset(qpos=qpos)
    force = np.array([1.0, 0.0, 0.0])
    point_local = np.array([0.1, 0.0, 0.0])
    model.set_external_force("torso", force, point_local)
    body_id = model.body_ids["torso"]
    point_world = np.asarray(model.data.xmat[body_id]).reshape(3, 3) @ point_local
    np.testing.assert_allclose(model.data.xfrc_applied[body_id, 3:], np.cross(point_world, force), atol=1e-10)


def test_qp_residuals_and_support_wrench_limits_are_production_values():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")
    config = {
        "posture_kp": 120.0, "posture_kd": 18.0, "torso_position_kp": 180.0,
        "torso_position_kd": 28.0, "torso_rotation_kp": 220.0, "torso_rotation_kd": 32.0,
        "pelvis_position_kp": 140.0, "pelvis_position_kd": 24.0, "pelvis_rotation_kp": 160.0,
        "pelvis_rotation_kd": 26.0, "com_kp": 70.0, "com_kd": 18.0, "qp_acceleration_weight": 0.02,
        "qp_torque_weight": 0.0005, "qp_nominal_torque_weight": 0.5, "qp_posture_weight": 2.0,
        "qp_torso_weight": 20.0, "qp_pelvis_weight": 8.0, "qp_com_weight": 3.0,
        "qp_slack_weight": 100000.0, "friction_coefficient": 0.7, "max_joint_acceleration": 250.0,
        "support_polygon_x_min_m": -0.115, "support_polygon_x_max_m": 0.225,
        "support_polygon_y_min_m": -0.12, "support_polygon_y_max_m": 0.12,
        "torsional_friction_coefficient": 0.02,
        "solver": {"eps_abs": 1e-3, "eps_rel": 1e-3, "max_iter": 20000, "polish": True},
    }
    result = WholeBodyQPController(model, config).solve()
    assert result.success
    assert result.dynamics_residual_norm < 0.1
    assert result.contact_acceleration_residual_norm < 0.1
    for offset in (0, 6):
        fx, fy, fz, mx, my, mz = result.contact_wrench[offset : offset + 6]
        assert fz >= -1e-3
        assert abs(fx) <= 0.7 * fz + 1e-3
        assert abs(fy) <= 0.7 * fz + 1e-3
        assert -0.12 * fz - 1e-3 <= mx <= 0.12 * fz + 1e-3
        assert -0.225 * fz - 1e-3 <= my <= 0.115 * fz + 1e-3
        assert abs(mz) <= 0.02 * fz + 1e-3


def test_contact_jacobian_directional_finite_difference_is_consistent():
    repo_root = Path(__file__).resolve().parents[1]
    model = HumanoidModel(repo_root / "models" / "humanoid" / "mini_humanoid.xml")
    model.data.qvel[:] = np.linspace(-0.02, 0.02, model.nv)
    mujoco.mj_forward(model.model, model.data)

    dt = 1e-6
    qpos = model.data.qpos.copy()
    qvel = model.data.qvel.copy()
    jacobian_now = model.contact_jacobian()
    qpos_next = qpos.copy()
    mujoco.mj_integratePos(model.model, qpos_next, qvel, dt)
    model.data.qpos[:] = qpos_next
    mujoco.mj_forward(model.model, model.data)
    jacobian_next = model.contact_jacobian()
    model.data.qpos[:] = qpos
    model.data.qvel[:] = qvel
    mujoco.mj_forward(model.model, model.data)

    expected = ((jacobian_next - jacobian_now) / dt) @ qvel
    actual = model.contact_bias_acceleration(dt)
    assert np.allclose(actual, expected, atol=1e-8, rtol=1e-5)
