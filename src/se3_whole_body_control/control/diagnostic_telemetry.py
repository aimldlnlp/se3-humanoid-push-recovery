"""Corrected, diagnostic-only telemetry for post-touchdown replay studies.

The historical ``diagnostic_wbc`` module is intentionally not rewritten by
this overlay. This module provides a narrow corrected path for new replay
artifacts: it fixes canonical contact-wrench indexing, exposes QP residual and
actuator-bound proxies, and records MuJoCo post-step actuator/contact timing.
It never changes the production QP formulation or the physical model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .diagnostic_wbc import DiagnosticWholeBodyQPController as _HistoricalDiagnosticController
from se3_whole_body_control.evaluation.metrics import TrialLog
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner as _HistoricalSimulationRunner


CANONICAL_FOOT_NAMES = ("left_foot", "right_foot")


def map_contact_wrench_to_canonical_feet(
    wrench: np.ndarray,
    active_contacts: tuple[str, ...] | list[str],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Map active-contact wrench slots into ``left_foot, right_foot`` order.

    A QP with one active right-foot contact has a six-element wrench whose
    first slot is the right foot. The fixed artifact layout is twelve
    elements, with the right foot in indices 6:12. Inactive slots remain
    ``fill_value`` so they cannot be mistaken for a zero measured wrench.
    """

    names = tuple(str(name) for name in active_contacts)
    if not names or any(name not in CANONICAL_FOOT_NAMES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"active_contacts must be a unique non-empty subset of {CANONICAL_FOOT_NAMES}")
    values = np.asarray(wrench, dtype=float).reshape(-1)
    expected = 6 * len(names)
    if values.size != expected:
        raise ValueError(f"wrench has {values.size} values, expected {expected} for {names}")
    mapped = np.full(12, float(fill_value), dtype=float)
    for source_index, name in enumerate(names):
        target_index = CANONICAL_FOOT_NAMES.index(name)
        mapped[6 * target_index : 6 * target_index + 6] = values[6 * source_index : 6 * source_index + 6]
    return mapped


def post_step_timestamp(pre_step_time_s: float, integrated_duration_s: float) -> float:
    """Return physical simulation time after one integrated control step."""

    pre = float(pre_step_time_s)
    duration = float(integrated_duration_s)
    if not np.isfinite(pre) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("pre-step time and integrated duration must be finite and positive")
    return pre + duration


class CorrectedDiagnosticWholeBodyQPController(_HistoricalDiagnosticController):
    """Historical diagnostic controller with corrected artifact telemetry."""

    def _padded_wrench(self, wrench: np.ndarray) -> np.ndarray:
        # The historical implementation always copied the active wrench into
        # the left slot. Keep QP ordering untouched and only fix canonical
        # serialization used by this diagnostic controller.
        return map_contact_wrench_to_canonical_feet(wrench, self.contact_names)

    @staticmethod
    def _saturation_mask(control: np.ndarray, limits: np.ndarray) -> np.ndarray:
        control = np.asarray(control, dtype=float).reshape(-1)
        limits = np.asarray(limits, dtype=float)
        tolerance = np.maximum(1.0e-6, 1.0e-6 * (limits[:, 1] - limits[:, 0]))
        return (
            (np.abs(control - limits[:, 0]) <= tolerance)
            | (np.abs(control - limits[:, 1]) <= tolerance)
        )

    def solve(self):
        result = super().solve()
        if self.diagnostic_history:
            item = self.diagnostic_history[-1]
            model = self.internal_model
            limits = np.asarray(model.actuator_limits, dtype=float)
            denominator = np.maximum(np.max(np.abs(limits), axis=1), 1.0e-12)
            utilization = np.abs(np.asarray(result.control, dtype=float)) / denominator
            saturated = self._saturation_mask(result.control, limits)
            item["canonical_wrench_foot_order"] = list(CANONICAL_FOOT_NAMES)
            item["commanded_torque_utilization_by_actuator"] = utilization
            item["commanded_torque_saturated_mask"] = saturated
            item["commanded_torque_saturated_count"] = int(np.sum(saturated))
            item["qp_primal_residual"] = float(result.primal_residual)
            item["qp_dual_residual"] = float(result.dual_residual)
        return result

    def history_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().history_arrays()
        history = self.diagnostic_history

        def stack(name: str, shape: tuple[int, ...], fill: float = np.nan) -> np.ndarray:
            values = []
            for item in history:
                value = item.get(name)
                if value is None:
                    values.append(np.full(shape, fill, dtype=float))
                    continue
                value_array = np.asarray(value, dtype=float)
                if value_array.shape != shape:
                    raise ValueError(f"diagnostic field {name!r} has shape {value_array.shape}, expected {shape}")
                values.append(value_array)
            return np.stack(values) if values else np.zeros((0,) + shape, dtype=float)

        def stack_bool(name: str, shape: tuple[int, ...]) -> np.ndarray:
            values = []
            for item in history:
                value = item.get(name)
                if value is None:
                    values.append(np.zeros(shape, dtype=bool))
                    continue
                value_array = np.asarray(value, dtype=bool)
                if value_array.shape != shape:
                    raise ValueError(f"diagnostic field {name!r} has shape {value_array.shape}, expected {shape}")
                values.append(value_array)
            return np.stack(values) if values else np.zeros((0,) + shape, dtype=bool)

        def scalar(name: str, fill: float = np.nan) -> np.ndarray:
            return np.asarray([item.get(name, fill) for item in history], dtype=float)

        arrays.update({
            "canonical_wrench_foot_order_json": np.asarray(json.dumps(CANONICAL_FOOT_NAMES)),
            "commanded_torque_utilization_by_actuator": stack(
                "commanded_torque_utilization_by_actuator", (self.model.nu,),
            ),
            "commanded_torque_saturated_mask": stack_bool(
                "commanded_torque_saturated_mask", (self.model.nu,),
            ),
            "commanded_torque_saturated_count": scalar("commanded_torque_saturated_count", 0.0),
            "qp_primal_residual": scalar("qp_primal_residual"),
            "qp_dual_residual": scalar("qp_dual_residual"),
        })
        return arrays


def _read_data_vector(data, field_name: str, expected_size: int) -> tuple[np.ndarray, bool]:
    value = getattr(data, field_name, None)
    if value is None:
        return np.full(expected_size, np.nan, dtype=float), False
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != expected_size:
        return np.full(expected_size, np.nan, dtype=float), False
    return array.copy(), True


def extract_post_step_actuator_telemetry(
    model,
    *,
    commanded_control: np.ndarray,
    pre_step_time_s: float,
    expected_control_timestep_s: float,
) -> dict[str, np.ndarray | float | bool]:
    """Extract actuator-space and generalized MuJoCo effort after integration.

    ``data.actuator_force`` is kept separate from the requested controller
    command. For the G1 direct ``motor`` actuators, ``gear=1`` and the
    effective generalized effort is numerically the same quantity. The
    artifact still records both the raw actuator force and
    ``data.qfrc_actuator`` so that this assumption is auditable.
    """

    force, force_available = _read_data_vector(model.data, "actuator_force", int(model.nu))
    generalized, generalized_available = _read_data_vector(model.data, "qfrc_actuator", int(model.nv))
    applied_control, applied_control_available = _read_data_vector(model.data, "ctrl", int(model.nu))
    gear_matrix = np.asarray(model.model.actuator_gear, dtype=float)
    gear = gear_matrix[:, 0].reshape(-1) if gear_matrix.ndim == 2 and gear_matrix.shape[0] == model.nu else np.full(model.nu, np.nan)
    limits = np.asarray(model.actuator_limits, dtype=float)
    effective_limit = np.maximum(np.max(np.abs(limits), axis=1) * np.abs(gear), 1.0e-12)
    effective_effort = force * gear
    effort_utilization = np.divide(
        np.abs(effective_effort), effective_limit,
        out=np.full(model.nu, np.nan, dtype=float),
        where=np.isfinite(effective_effort) & np.isfinite(effective_limit),
    )
    pre = float(pre_step_time_s)
    post = float(model.data.time)
    expected_post = post_step_timestamp(pre, expected_control_timestep_s)
    return {
        "pre_step_time_s": pre,
        "post_step_time_s": post,
        "expected_post_step_time_s": expected_post,
        "timestamp_error_s": post - expected_post,
        "timestamp_semantics_valid": bool(np.isfinite(post) and abs(post - expected_post) <= 1.0e-10),
        "commanded_control": np.asarray(commanded_control, dtype=float).copy(),
        "applied_control_post_step": applied_control,
        "applied_control_available": bool(applied_control_available),
        "actual_actuator_force_post_step": force,
        "actual_actuator_force_available": bool(force_available),
        "actual_generalized_actuator_force_post_step": generalized,
        "actual_generalized_actuator_force_available": bool(generalized_available),
        "actuator_gear": gear,
        "effective_actuator_effort_post_step": effective_effort,
        "actual_actuator_effort_utilization_post_step": effort_utilization,
        "effective_actuator_effort_limit": effective_limit,
    }


class _RunRecorder:
    def __init__(self, model, expected_control_timestep_s: float):
        self.model = model
        self.expected_control_timestep_s = float(expected_control_timestep_s)
        self.rows: list[dict] = []

    def record(self, values: dict) -> None:
        self.rows.append(extract_post_step_actuator_telemetry(
            self.model,
            commanded_control=np.asarray(values.get("control", []), dtype=float),
            pre_step_time_s=float(values["time_s"]),
            expected_control_timestep_s=self.expected_control_timestep_s,
        ))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        keys = tuple(self.rows[0])
        output: dict[str, np.ndarray] = {}
        for key in keys:
            values = [row[key] for row in self.rows]
            first = values[0]
            if isinstance(first, (bool, np.bool_)):
                output[key] = np.asarray(values, dtype=bool)
            elif np.asarray(first).shape == ():
                output[key] = np.asarray(values, dtype=float)
            else:
                output[key] = np.stack([np.asarray(value) for value in values])
        return output


_ACTIVE_RECORDER: _RunRecorder | None = None
_TELEMETRY_BY_RUN: dict[int, dict[str, np.ndarray]] = {}
_ORIGINAL_TRIAL_LOG_APPEND = TrialLog.append


def _recording_trial_log_append(self: TrialLog, **values) -> None:
    if _ACTIVE_RECORDER is not None:
        _ACTIVE_RECORDER.record(values)
    _ORIGINAL_TRIAL_LOG_APPEND(self, **values)


if getattr(TrialLog.append, "_corrected_diagnostic_wrapper", False) is not True:
    _recording_trial_log_append._corrected_diagnostic_wrapper = True
    TrialLog.append = _recording_trial_log_append


class InstrumentedSimulationRunner(_HistoricalSimulationRunner):
    """Historical runner plus post-step actuator/timestamp capture."""

    def run(self, *args, **kwargs):
        global _ACTIVE_RECORDER
        recorder = _RunRecorder(self.model, self.control_timestep_s)
        previous = _ACTIVE_RECORDER
        _ACTIVE_RECORDER = recorder
        try:
            run = super().run(*args, **kwargs)
        finally:
            _ACTIVE_RECORDER = previous
        arrays = recorder.arrays()
        expected_rows = len(run.log.arrays()["time_s"])
        if len(arrays.get("pre_step_time_s", [])) != expected_rows:
            raise RuntimeError(
                f"corrected actuator telemetry misalignment: {len(arrays.get('pre_step_time_s', []))} != {expected_rows}"
            )
        _TELEMETRY_BY_RUN[id(run)] = arrays
        return run


def save_run_with_corrected_telemetry(original_save_run: Callable, run, path: Path, metadata: dict | None = None) -> None:
    """Save the historical trial plus a provenance-linked telemetry sidecar."""

    arrays = _TELEMETRY_BY_RUN.pop(id(run), {})
    metadata = dict(metadata or {})
    if arrays:
        root = Path(path).resolve().parents[2]
        telemetry_path = root / "data" / "diagnostic" / f"{Path(path).stem}_actuator_telemetry.npz"
        metadata["corrected_actuator_telemetry"] = {
            "path": telemetry_path.relative_to(root).as_posix(),
            "timestamp_convention": {
                "pre_step_time_s": "TrialLog.time_s; state and controller command before integration",
                "post_step_time_s": "model.data.time after all MuJoCo substeps",
                "post_step_fields": "measured from MuJoCo after the command was integrated",
            },
            "actuator_force_semantics": "data.actuator_force; effective generalized effort is actuator_force * actuator_gear[:,0]",
            "normalized_effort_basis": "effective actuator effort divided by effective model.actuator_limits",
        }
    original_save_run(run, path, metadata)
    if arrays:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(arrays)
        serializable["metadata_json"] = np.asarray(json.dumps(metadata["corrected_actuator_telemetry"], sort_keys=True))
        np.savez_compressed(telemetry_path, **serializable)
