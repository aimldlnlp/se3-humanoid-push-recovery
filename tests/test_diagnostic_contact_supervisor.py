from dataclasses import dataclass

import numpy as np

from se3_whole_body_control.control.diagnostic_contact_supervisor import (
    MeasuredContactModeSupervisor,
)
from se3_whole_body_control.control.diagnostic_telemetry import (
    map_contact_wrench_to_canonical_feet,
)


@dataclass
class _Contact:
    contact_flags: np.ndarray
    normal_force_N: np.ndarray


def _contact(left: float | None, right: float | None) -> _Contact:
    values = np.asarray([
        np.nan if left is None else float(left),
        np.nan if right is None else float(right),
    ])
    return _Contact(
        contact_flags=np.isfinite(values) & (values > 0.0),
        normal_force_N=values,
    )


def test_right_support_loss_is_debounced_and_never_emits_empty_mode():
    supervisor = MeasuredContactModeSupervisor()

    first = supervisor.update(_contact(None, None), 0.100)
    second = supervisor.update(_contact(None, None), 0.104)

    assert first.active_contacts == ("right_foot",)
    assert second.active_contacts == ("right_foot",)
    assert second.event == "hold_no_loaded_support"
    assert not second.changed


def test_right_loss_and_left_load_handoff_requires_two_samples():
    supervisor = MeasuredContactModeSupervisor()

    supervisor.update(_contact(None, None), 0.100)
    decision = supervisor.update(_contact(30.0, None), 0.104)
    assert decision.active_contacts == ("right_foot",)
    decision = supervisor.update(_contact(30.0, None), 0.108)

    assert decision.active_contacts == ("left_foot",)
    assert decision.event == "mode_change:left_foot"
    assert decision.loaded_contacts == ("left_foot",)


def test_left_load_promotes_double_support_when_right_remains_loaded():
    supervisor = MeasuredContactModeSupervisor()

    supervisor.update(_contact(None, 40.0), 0.100)
    decision = supervisor.update(_contact(25.0, 40.0), 0.104)
    assert decision.active_contacts == ("right_foot",)
    decision = supervisor.update(_contact(25.0, 40.0), 0.108)

    assert decision.active_contacts == ("left_foot", "right_foot")
    assert decision.event == "mode_change:left_foot+right_foot"


def test_double_support_releases_right_only_after_two_low_samples():
    supervisor = MeasuredContactModeSupervisor(initial_active_contacts=("left_foot", "right_foot"))

    supervisor.update(_contact(25.0, 25.0), 0.100)
    decision = supervisor.update(_contact(25.0, 2.0), 0.104)
    assert decision.active_contacts == ("left_foot", "right_foot")
    decision = supervisor.update(_contact(25.0, 2.0), 0.108)

    assert decision.active_contacts == ("left_foot",)
    assert decision.event == "mode_change:left_foot"


def test_hysteresis_band_does_not_acquire_a_noisy_four_newton_contact():
    supervisor = MeasuredContactModeSupervisor()

    for index in range(5):
        decision = supervisor.update(_contact(4.0, 40.0), 0.100 + 0.004 * index)

    assert decision.active_contacts == ("right_foot",)
    assert decision.loaded_streak[0] == 0
    assert not decision.changed


def test_wrench_slots_follow_contact_set_changes():
    left = np.arange(6.0)
    right = np.arange(10.0, 16.0)

    mapped_right = map_contact_wrench_to_canonical_feet(right, ("right_foot",))
    mapped_both = map_contact_wrench_to_canonical_feet(
        np.r_[left, right], ("left_foot", "right_foot"),
    )
    mapped_left = map_contact_wrench_to_canonical_feet(left, ("left_foot",))

    assert np.allclose(mapped_right[6:12], right)
    assert np.isnan(mapped_right[:6]).all()
    assert np.allclose(mapped_both[:6], left)
    assert np.allclose(mapped_both[6:12], right)
    assert np.allclose(mapped_left[:6], left)
    assert np.isnan(mapped_left[6:12]).all()
