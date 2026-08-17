"""Trial data containers and machine-readable serialization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass
class TrialLog:
    time_s: list[float]
    torso_error: list[list[float]]
    pelvis_error: list[list[float]]
    com_world: list[list[float]]
    torso_position: list[list[float]]
    torso_rotation_error_rad: list[float]
    contact_left: list[bool]
    contact_right: list[bool]
    predicted_contact_wrench: list[list[float]]
    actual_contact_wrench: list[list[float]]
    actual_friction_utilization: list[list[float]]
    foot_tangent_velocity: list[list[float]]
    foot_xy_displacement: list[list[float]]
    foot_xy_world: list[list[float]]
    foot_support_vertices_world: list[list[float]]
    foot_cop_world: list[list[float]]
    control: list[list[float]]
    qp_status: list[str]
    qp_solve_time_s: list[float]
    push_force: list[list[float]]
    joint_velocity_norm: list[float]
    torso_angular_velocity_norm: list[float]
    torso_height_m: list[float]
    torque_abs_max_Nm: list[float]
    torque_utilization: list[float]
    qp_success: list[bool]
    predicted_friction_margin: list[float]
    actual_friction_margin: list[float]
    qp_message: list[str]
    qp_slack_norm: list[float]
    dynamics_residual_norm: list[float]
    contact_acceleration_residual_norm: list[float]
    joint_limit_violation: list[bool]
    numerical_valid: list[bool]
    control_mode: list[str]
    step_phase: list[str]
    swing_foot: list[str]
    support_margin_m: list[float]
    # Contact measured after the control command has been integrated through
    # the MuJoCo substeps.  The original fields above are retained as the
    # pre-step observation for backwards-compatible plots and audits.
    contact_left_post_step: list[bool]
    contact_right_post_step: list[bool]
    actual_contact_wrench_post_step: list[list[float]]
    actual_normal_force_post_step_N: list[list[float]]
    actual_friction_utilization_post_step: list[list[float]]
    foot_tangent_velocity_post_step: list[list[float]]
    foot_xy_displacement_post_step: list[list[float]]
    foot_cop_world_post_step: list[list[float]]
    event_label: list[str]
    event_reason: list[str]
    step_count: list[int]
    planned_foot_target_world: list[list[float]]

    @classmethod
    def empty(cls) -> "TrialLog":
        return cls(**{field: [] for field in cls.__dataclass_fields__})

    def append(self, **values) -> None:
        values = dict(values)
        # Keep simple/legacy log producers row-aligned when they do not know
        # about the post-step telemetry added by the arena. The MuJoCo runner
        # supplies these keys explicitly, so this is only a compatibility
        # fallback and never replaces measured post-step data.
        post_defaults = {
            "contact_left_post_step": values.get("contact_left", False),
            "contact_right_post_step": values.get("contact_right", False),
            "actual_contact_wrench_post_step": values.get("actual_contact_wrench", [0.0] * 12),
            "actual_friction_utilization_post_step": values.get("actual_friction_utilization", [float("nan")] * 2),
            "foot_tangent_velocity_post_step": values.get("foot_tangent_velocity", [float("nan")] * 2),
            "foot_xy_displacement_post_step": values.get("foot_xy_displacement", [float("nan")] * 2),
            "foot_cop_world_post_step": values.get("foot_cop_world", [float("nan")] * 4),
            "event_label": "",
            "event_reason": "",
            "step_count": 0,
            "planned_foot_target_world": [float("nan")] * 3,
        }
        if "actual_normal_force_post_step_N" not in values:
            wrench = np.asarray(values.get("actual_contact_wrench", [0.0] * 12), dtype=float).reshape(-1)
            post_defaults["actual_normal_force_post_step_N"] = [
                float(wrench[2]) if wrench.size >= 3 else 0.0,
                float(wrench[8]) if wrench.size >= 9 else 0.0,
            ]
        for key, default in post_defaults.items():
            values.setdefault(key, default)
        for key, value in values.items():
            getattr(self, key).append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for key, value in asdict(self).items():
            if key in {"qp_status", "control_mode", "step_phase", "swing_foot", "event_label", "event_reason"}:
                out[key] = np.asarray(value, dtype="U64")
            else:
                out[key] = np.asarray(value)
        return out


def save_trial_npz(log: TrialLog, path: str | Path, metadata: dict | None = None, extra_arrays: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = log.arrays()
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    arrays.update(extra_arrays or {})
    np.savez_compressed(path, **arrays)


def summarize_trial(log: TrialLog) -> dict:
    arrays = log.arrays()
    if len(arrays["time_s"]) == 0:
        return {}
    qp_ms = arrays["qp_solve_time_s"] * 1000.0
    finite_qp_ms = qp_ms[np.isfinite(qp_ms)]
    predicted = np.asarray(arrays["predicted_contact_wrench"], dtype=float)
    actual = np.asarray(
        arrays.get("actual_contact_wrench_post_step", arrays["actual_contact_wrench"]),
        dtype=float,
    )
    consistency = {}
    if predicted.ndim == 2 and actual.ndim == 2 and predicted.shape == actual.shape and predicted.shape[1] >= 9:
        predicted_force = predicted[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
        actual_force = actual[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
        valid = np.all(np.isfinite(predicted_force), axis=(1, 2)) & np.all(np.isfinite(actual_force), axis=(1, 2))
        if np.any(valid):
            delta = predicted_force[valid] - actual_force[valid]
            consistency = {
                "qp_measured_vertical_grf_rmse_N_post_step": float(np.sqrt(np.mean(delta[:, :, 2] ** 2))),
                "qp_measured_vertical_grf_bias_N_post_step": float(np.mean(delta[:, :, 2])),
                "qp_measured_force_rmse_N_post_step": float(np.sqrt(np.mean(delta ** 2))),
            }
    com_delta = arrays["com_world"] - arrays["com_world"][0]
    com_horizontal = np.linalg.norm(com_delta[:, :2], axis=1)
    com_3d = np.linalg.norm(com_delta, axis=1)
    force_components = actual[:, [0, 1, 2, 6, 7, 8]] if actual.ndim == 2 and actual.shape[1] >= 9 else np.zeros((0, 6))
    force_resultants = np.linalg.norm(force_components.reshape(-1, 2, 3), axis=2) if len(force_components) else np.zeros(0)
    actual_utilization = np.asarray(
        arrays.get("actual_friction_utilization_post_step", arrays["actual_friction_utilization"]),
        dtype=float,
    )
    return {
        "duration_s": float(arrays["time_s"][-1]),
        "max_torso_error_rad": float(np.max(arrays["torso_rotation_error_rad"])),
        "max_com_displacement_m": float(np.max(com_horizontal)),
        "max_com_3d_displacement_m": float(np.max(com_3d)),
        "max_joint_velocity": float(np.max(arrays["joint_velocity_norm"])),
        "max_qp_solve_time_ms": float(np.max(arrays["qp_solve_time_s"]) * 1000.0),
        "mean_qp_solve_time_ms": float(np.mean(finite_qp_ms)) if finite_qp_ms.size else float("nan"),
        "p95_qp_solve_time_ms": float(np.percentile(finite_qp_ms, 95)) if finite_qp_ms.size else float("nan"),
        "p99_qp_solve_time_ms": float(np.percentile(finite_qp_ms, 99)) if finite_qp_ms.size else float("nan"),
        "qp_deadline_ms": 4.0,
        "qp_deadline_miss_percent": float(np.mean(finite_qp_ms > 4.0) * 100.0) if finite_qp_ms.size else float("nan"),
        "qp_failures": int(np.sum(np.char.startswith(arrays["qp_status"].astype(str), "fallback"))),
        "max_contact_slack_norm": float(np.max(arrays["qp_slack_norm"])),
        "max_actual_contact_force_N": float(np.max(force_resultants)) if force_resultants.size else 0.0,
        "max_actual_contact_component_N": float(np.max(np.abs(force_components))) if force_components.size else 0.0,
        "max_actual_friction_utilization": float(np.max(actual_utilization)) if actual_utilization.size else float("nan"),
        "max_dynamics_residual": float(np.max(arrays["dynamics_residual_norm"])),
        "max_contact_acceleration_residual": float(np.max(arrays["contact_acceleration_residual_norm"])),
        "left_contact_fraction": float(np.mean(arrays["contact_left"])),
        "right_contact_fraction": float(np.mean(arrays["contact_right"])),
        **consistency,
    }
