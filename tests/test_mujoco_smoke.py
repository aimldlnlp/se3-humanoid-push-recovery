from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from se3_whole_body_control.dynamics.humanoid import HumanoidModel


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
