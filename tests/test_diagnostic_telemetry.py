from types import SimpleNamespace

import numpy as np
import pytest

from se3_whole_body_control.control.diagnostic_telemetry import (
    extract_post_step_actuator_telemetry,
    map_contact_wrench_to_canonical_feet,
    post_step_timestamp,
)


def test_active_wrench_slots_are_mapped_to_canonical_left_right_order():
    left = np.arange(10.0, 16.0)
    right = np.arange(20.0, 26.0)
    mapped_right = map_contact_wrench_to_canonical_feet(right, ("right_foot",))
    assert np.all(np.isnan(mapped_right[:6]))
    np.testing.assert_array_equal(mapped_right[6:], right)

    mapped_reversed = map_contact_wrench_to_canonical_feet(
        np.r_[right, left], ("right_foot", "left_foot"),
    )
    np.testing.assert_array_equal(mapped_reversed[:6], left)
    np.testing.assert_array_equal(mapped_reversed[6:], right)


def test_post_step_timestamp_is_explicitly_offset_from_pre_step_time():
    assert post_step_timestamp(0.100, 0.004) == pytest.approx(0.104)
    with pytest.raises(ValueError):
        post_step_timestamp(0.100, 0.0)


def test_actuator_telemetry_preserves_pre_post_timestamp_semantics():
    model = SimpleNamespace(
        nu=2,
        nv=3,
        data=SimpleNamespace(
            time=0.104,
            actuator_force=np.array([3.0, -4.0]),
            qfrc_actuator=np.array([0.0, 3.0, -4.0]),
            ctrl=np.array([3.0, -4.0]),
        ),
        model=SimpleNamespace(actuator_gear=np.ones((2, 1))),
        actuator_limits=np.array([[-10.0, 10.0], [-8.0, 8.0]]),
    )
    telemetry = extract_post_step_actuator_telemetry(
        model,
        commanded_control=np.array([3.0, -4.0]),
        pre_step_time_s=0.100,
        expected_control_timestep_s=0.004,
    )
    assert telemetry["pre_step_time_s"] == pytest.approx(0.100)
    assert telemetry["post_step_time_s"] == pytest.approx(0.104)
    assert telemetry["timestamp_error_s"] == pytest.approx(0.0)
    assert telemetry["timestamp_semantics_valid"] is True
    np.testing.assert_array_equal(telemetry["actual_actuator_force_post_step"], [3.0, -4.0])
    np.testing.assert_allclose(telemetry["actual_actuator_effort_utilization_post_step"], [0.3, 0.5])
