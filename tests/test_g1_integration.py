from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from se3_whole_body_control.config import load_configs, resolve_model_path
from se3_whole_body_control.control.tasks import com_jacobian
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
from se3_whole_body_control.dynamics.humanoid import HumanoidModel


def _g1():
    root = Path(__file__).resolve().parents[1]
    configs = load_configs(root, robot_name="unitree_g1")
    return HumanoidModel(resolve_model_path(configs), robot_config=configs["robot"])


def test_g1_model_dimensions_and_name_mapping_are_explicit():
    model = _g1()
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.adapter.name == "unitree_g1"
    assert model.body_ids["pelvis"] >= 0
    assert model.body_ids["torso"] >= 0
    assert model.body_ids["left_foot"] >= 0
    assert model.body_ids["right_foot"] >= 0
    assert len(model.foot_contact_geom_ids[0]) == 4
    assert len(model.foot_contact_geom_ids[1]) == 4
    assert len(model.joint_names) == model.nu
    assert len(set(model.joint_names)) == model.nu
    assert model.actuator_limits.shape == (29, 2)
    assert np.all(model.actuator_limits[:, 0] < model.actuator_limits[:, 1])


def test_g1_nominal_standing_has_both_feet_and_finite_kinematics():
    model = _g1()
    state = model.state()
    assert state.contact_left and state.contact_right
    assert state.torso_jacobian.shape == (6, 35)
    assert state.left_foot_jacobian.shape == (6, 35)
    assert state.right_foot_jacobian.shape == (6, 35)
    assert np.all(np.isfinite(model.mass_matrix()))
    assert np.all(np.linalg.eigvalsh(model.mass_matrix()) > 0.0)
    assert np.all(np.isfinite(com_jacobian(model)))
    assert np.all(np.isfinite(model.foot_support_vertices_world()))
    contact = model.actual_contact_data()
    assert contact.contact_flags.all()
    assert np.all(contact.wrench_world[[2, 8]] > 0.0)


def test_g1_com_jacobian_matches_finite_difference():
    model = _g1()
    model.data.qvel[:] = np.linspace(-0.01, 0.01, model.nv)
    mujoco.mj_forward(model.model, model.data)
    dt = 1e-6
    qpos0 = model.data.qpos.copy()
    qvel0 = model.data.qvel.copy()
    com0 = model.center_of_mass().copy()
    qpos_next = qpos0.copy()
    mujoco.mj_integratePos(model.model, qpos_next, qvel0, dt)
    model.data.qpos[:] = qpos_next
    mujoco.mj_forward(model.model, model.data)
    com1 = model.center_of_mass().copy()
    model.data.qpos[:] = qpos0
    model.data.qvel[:] = qvel0
    mujoco.mj_forward(model.model, model.data)
    np.testing.assert_allclose((com1 - com0) / dt, com_jacobian(model) @ qvel0, atol=4e-5, rtol=4e-4)


def test_g1_qp_contact_constraints_and_measured_grf_are_separate():
    root = Path(__file__).resolve().parents[1]
    configs = load_configs(root, robot_name="unitree_g1")
    model = _g1()
    result = WholeBodyQPController(model, configs["controller"], configs["experiments"]["recovery"]).solve()
    assert result.success
    assert result.dynamics_residual_norm < 0.1
    assert result.contact_acceleration_residual_norm < 0.1
    for offset in (0, 6):
        fx, fy, fz, mx, my, mz = result.contact_wrench[offset:offset + 6]
        assert fz >= -1e-3
        assert abs(fx) <= configs["controller"]["friction_coefficient"] * fz + 1e-3
        assert abs(fy) <= configs["controller"]["friction_coefficient"] * fz + 1e-3
        assert configs["controller"]["support_polygon_y_min_m"] * fz - 1e-3 <= mx <= configs["controller"]["support_polygon_y_max_m"] * fz + 1e-3
        assert configs["controller"]["support_polygon_x_min_m"] * fz - 1e-3 <= my <= configs["controller"]["support_polygon_x_max_m"] * fz + 1e-3
    actual = model.actual_contact_data()
    assert actual.contact_flags.all()
    assert np.all(np.isfinite(actual.wrench_world))
    assert np.all(actual.wrench_world[[2, 8]] > 0.0)
    # The QP wrench is a prediction; the plant wrench is measured from the
    # MuJoCo contact solver and is intentionally a separate signal.
    assert result.contact_wrench.shape == (12,)
    assert actual.wrench_world.shape == (12,)


def test_g1_push_application_point_uses_body_local_frame():
    model = _g1()
    qpos = model.qpos0.copy()
    qpos[3:7] = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    model.reset(qpos=qpos)
    force = np.array([1.0, 0.0, 0.0])
    point_local = np.array([0.0, 0.0, 0.1])
    model.set_external_force("torso", force, point_local)
    body_id = model.body_ids["torso"]
    point_world = np.asarray(model.data.xmat[body_id]).reshape(3, 3) @ point_local
    np.testing.assert_allclose(model.data.xfrc_applied[body_id, 3:], np.cross(point_world, force), atol=1e-10)
