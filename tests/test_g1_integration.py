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


@pytest.mark.parametrize('persistent_violation', [False, True])
def test_production_solve_retries_then_rejects_invalid_solved_output(monkeypatch, persistent_violation):
    from types import SimpleNamespace
    import se3_whole_body_control.control.whole_body_qp as qp_module
    original = qp_module.osqp.OSQP
    calls = []
    model = _g1()
    configs = load_configs(Path(__file__).resolve().parents[1], robot_name='unitree_g1')

    class Solver:
        def __init__(self):
            self.real = original()

        def __getattr__(self, name):
            return getattr(self.real, name)

        def solve(self):
            solution = self.real.solve()
            calls.append(solution.info.status)
            x = solution.x.copy()
            if len(calls) == 1 or persistent_violation:
                x[model.nv + model.nu + 2] = -0.12
            return SimpleNamespace(x=x, info=solution.info)

    monkeypatch.setattr(qp_module.osqp, 'OSQP', Solver)
    result = WholeBodyQPController(model, configs['controller']).solve()
    assert calls == ['solved', 'solved']
    assert result.diagnostics['constraint_refinement_count'] == 1
    assert result.success is not persistent_violation
    if persistent_violation:
        assert result.status == 'fallback_pd'
        assert 'unscaled constraint validation failed' in result.message
        assert np.all(result.contact_wrench == 0)
    else:
        assert result.diagnostics['constraint_budget_ratio'] <= 1


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
    for foot_index, foot_name in enumerate(("left_foot", "right_foot")):
        fz = actual.wrench_world[6 * foot_index + 2]
        expected_cop = model.body_pose(foot_name)[:2, 3] + np.array([
            -actual.wrench_world[6 * foot_index + 4] / fz,
            actual.wrench_world[6 * foot_index + 3] / fz,
        ])
        np.testing.assert_allclose(actual.cop_world[foot_index], expected_cop, atol=1e-10)
    # The QP wrench is a prediction; the plant wrench is measured from the
    # MuJoCo contact solver and is intentionally a separate signal.
    assert result.contact_wrench.shape == (12,)
    assert actual.wrench_world.shape == (12,)


def test_g1_qp_single_support_mode_has_explicit_contact_dimension():
    root = Path(__file__).resolve().parents[1]
    configs = load_configs(root, robot_name="unitree_g1")
    model = _g1()
    controller = WholeBodyQPController(model, configs["controller"], configs["experiments"]["recovery"])
    controller.set_active_contacts(("left_foot",))
    result = controller.solve()
    assert result.success
    assert result.contact_wrench.shape == (6,)
    assert result.diagnostics["active_contacts"] == ["left_foot"]
    controller.set_active_contacts(("left_foot", "right_foot"))


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


def test_qp_workspace_reuses_only_compatible_problem_and_contact_mode():
    configs = load_configs(Path(__file__).resolve().parents[1], robot_name="unitree_g1")
    config = configs["controller"]
    config["solver"]["reuse_workspace"] = True
    controller = WholeBodyQPController(_g1(), config)
    first = controller.solve()
    original_solver = controller._solver
    second = controller.solve()
    assert first.success and second.success
    assert not first.diagnostics["workspace_reused"]
    assert second.diagnostics["workspace_reused"]
    assert controller._solver is original_solver
    np.testing.assert_allclose(second.control, first.control, atol=2e-3, rtol=2e-3)
    controller.set_active_contacts(("right_foot",))
    third = controller.solve()
    assert third.success and not third.diagnostics["workspace_reused"]
    assert controller._solver is not original_solver
    assert third.contact_wrench.shape == (6,)
    for key in ("qp_build_time_s", "qp_prepare_time_s", "qp_numerical_solve_time_s"):
        assert 0 <= third.diagnostics[key] <= third.solve_time_s


def test_warm_and_cold_qp_solve_same_changed_state_within_solver_tolerance():
    configs = load_configs(Path(__file__).resolve().parents[1], robot_name="unitree_g1")
    warm_config = configs["controller"]
    warm_config["solver"]["reuse_workspace"] = True
    model = _g1()
    warm = WholeBodyQPController(model, warm_config)
    assert warm.solve().success
    cold_config = {**warm_config, "solver": {**warm_config["solver"], "reuse_workspace": False}}
    cold = WholeBodyQPController(model, cold_config)
    model.data.qvel[:] = np.linspace(-1e-4, 1e-4, model.nv)
    mujoco.mj_forward(model.model, model.data)
    a, b = warm.solve(), cold.solve()
    assert a.success and b.success
    assert a.diagnostics["workspace_reused"]
    np.testing.assert_allclose(a.control, b.control, atol=5e-3, rtol=5e-3)
    assert a.dynamics_residual_norm < 0.1 and b.dynamics_residual_norm < 0.1
    for result in (a, b):
        forces = result.contact_wrench.reshape(-1, 6)
        assert np.all(np.abs(forces[:, 0]) + np.abs(forces[:, 1]) <= 0.7 * forces[:, 2] + 1e-3)
