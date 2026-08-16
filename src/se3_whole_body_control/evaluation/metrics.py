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

    @classmethod
    def empty(cls) -> "TrialLog":
        return cls(**{field: [] for field in cls.__dataclass_fields__})

    def append(self, **values) -> None:
        for key, value in values.items():
            getattr(self, key).append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for key, value in asdict(self).items():
            if key == "qp_status":
                out[key] = np.asarray(value, dtype="U32")
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
    actual = np.asarray(arrays["actual_contact_wrench"], dtype=float)
    consistency = {}
    if predicted.ndim == 2 and actual.ndim == 2 and predicted.shape == actual.shape and predicted.shape[1] >= 9:
        predicted_force = predicted[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
        actual_force = actual[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
        valid = np.all(np.isfinite(predicted_force), axis=(1, 2)) & np.all(np.isfinite(actual_force), axis=(1, 2))
        if np.any(valid):
            delta = predicted_force[valid] - actual_force[valid]
            consistency = {
                "qp_measured_vertical_grf_rmse_N": float(np.sqrt(np.mean(delta[:, :, 2] ** 2))),
                "qp_measured_vertical_grf_bias_N": float(np.mean(delta[:, :, 2])),
                "qp_measured_force_rmse_N": float(np.sqrt(np.mean(delta ** 2))),
            }
    return {
        "duration_s": float(arrays["time_s"][-1]),
        "max_torso_error_rad": float(np.max(arrays["torso_rotation_error_rad"])),
        "max_com_displacement_m": float(np.max(np.linalg.norm(arrays["com_world"] - arrays["com_world"][0], axis=1))),
        "max_joint_velocity": float(np.max(arrays["joint_velocity_norm"])),
        "max_qp_solve_time_ms": float(np.max(arrays["qp_solve_time_s"]) * 1000.0),
        "mean_qp_solve_time_ms": float(np.mean(finite_qp_ms)) if finite_qp_ms.size else float("nan"),
        "p95_qp_solve_time_ms": float(np.percentile(finite_qp_ms, 95)) if finite_qp_ms.size else float("nan"),
        "p99_qp_solve_time_ms": float(np.percentile(finite_qp_ms, 99)) if finite_qp_ms.size else float("nan"),
        "qp_deadline_ms": 4.0,
        "qp_deadline_miss_percent": float(np.mean(finite_qp_ms > 4.0) * 100.0) if finite_qp_ms.size else float("nan"),
        "qp_failures": int(np.sum(np.char.startswith(arrays["qp_status"].astype(str), "fallback"))),
        "max_contact_slack_norm": float(np.max(arrays["qp_slack_norm"])),
        "max_actual_contact_force_N": float(np.max(np.abs(arrays["actual_contact_wrench"][:, [0, 1, 2, 6, 7, 8]]))),
        "max_actual_friction_utilization": float(np.max(arrays["actual_friction_utilization"])),
        "max_dynamics_residual": float(np.max(arrays["dynamics_residual_norm"])),
        "max_contact_acceleration_residual": float(np.max(arrays["contact_acceleration_residual_norm"])),
        "left_contact_fraction": float(np.mean(arrays["contact_left"])),
        "right_contact_fraction": float(np.mean(arrays["contact_right"])),
        **consistency,
    }
