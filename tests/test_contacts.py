import numpy as np
import pytest

from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController


def _accepts(wrench, foot=0, contacts=("left_foot", "right_foot")):
    controller = object.__new__(WholeBodyQPController)
    controller.model = type("ModelDimensions", (), {"nv": 2, "nu": 1})()
    controller.cfg = {}
    controller.mu = 0.7
    controller.set_active_contacts(contacts)
    rows, lower, upper = [], [], []
    start = controller.model.nv + controller.model.nu
    controller._friction_rows(rows, lower, upper, start)
    x = np.zeros(controller.nx)
    x[start + 6 * foot:start + 6 * foot + 6] = wrench
    values = np.asarray(rows) @ x
    return bool(np.all(values >= np.asarray(lower) - 1e-10)
                and np.all(values <= np.asarray(upper) + 1e-10))


@pytest.mark.parametrize("foot", [0, 1])
def test_production_friction_rows_reject_outer_square_corners(foot):
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        assert not _accepts([sx * 0.7, sy * 0.7, 1, 0, 0, 0], foot)
        assert _accepts([sx * 0.35, sy * 0.35, 1, 0, 0, 0], foot)
    assert _accepts([0.7, 0, 1, 0, 0, 0], foot)
    assert not _accepts([0, 0, -1, 0, 0, 0], foot)
    assert _accepts([0, 0, 0, 0, 0, 0], foot)
    assert not _accepts([0.001, 0, 0, 0, 0, 0], foot)


@pytest.mark.parametrize("contacts", [("left_foot",), ("right_foot",)])
def test_single_contact_rows_keep_cop_and_torsion_limits(contacts):
    assert _accepts([0, 0, 1, 0.12, -0.225, 0.02], contacts=contacts)
    for index, value in ((3, 0.121), (4, -0.226), (4, 0.116), (5, 0.021)):
        wrench = np.array([0., 0., 1., 0., 0., 0.])
        wrench[index] = value
        assert not _accepts(wrench, contacts=contacts)


def test_accepted_force_boundary_is_inside_coulomb_circle():
    for angle in np.linspace(0, 2 * np.pi, 73):
        direction = np.array([np.cos(angle), np.sin(angle)])
        force = 0.7 * direction / np.sum(np.abs(direction))
        assert _accepts([*force, 1, 0, 0, 0])
        assert np.linalg.norm(force) <= 0.7 + 1e-12
