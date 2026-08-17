"""Project captured touchdown states into progressively stricter feasible sets.

This is a diagnostic study, not a stepping-controller change.  It starts from
an exact ``qpos``/``qvel`` snapshot emitted at a confirmed ``TOUCHDOWN`` and
creates the smallest nearby state repairs that can distinguish three failure
mechanisms:

1. the captured state is outside the configured joint limits;
2. the state is joint-limit-consistent but does not form a coherent landed-
   foot/support geometry; and
3. even a nearby state with one-step QP torque/friction headroom cannot be
   stabilized by the existing WBC.

The floating-base pose is held fixed for the first two projections.  The
optional headroom projection may move only the base x/y position inside a
small declared trust region.  The original ``qvel`` and controller context
are preserved in every replay; projected states are therefore diagnostic
counterfactuals and are never reported as physically reachable unless a
separate reachability experiment establishes that fact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_configs,
    make_model,
    write_csv,
    write_execution_manifest,
)
from replay_touchdown_recoverability import (  # noqa: E402
    _json,
    _make_replay_controller,
    _plot_replay,
    _replay,
    _replay_observables,
    _safe_name,
    _stability_summary,
)
from se3_whole_body_control.control.support import convex_hull_2d, signed_support_margin  # noqa: E402
from se3_whole_body_control.dynamics.humanoid import HumanoidModel  # noqa: E402


PROJECTION_STAGES = (
    "original_touchdown",
    "joint_limit_guard",
    "joint_limit_contact_support",
    "joint_limit_contact_headroom",
)


def _state_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value) -> str | float | int:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _load_capture(path: Path, configs: dict) -> dict:
    with np.load(path, allow_pickle=True) as data:
        context = json.loads(str(_scalar(data["context_json"])))
        state_sha = str(_scalar(data["state_sha256"]))
        capture = {
            "input_path": path.resolve(),
            "input_file_sha256": _state_file_sha256(path),
            "qpos": np.asarray(data["qpos"], dtype=float).copy(),
            "qvel": np.asarray(data["qvel"], dtype=float).copy(),
            "capture_com": np.asarray(data["com_world"], dtype=float).copy(),
            "time_s": float(_scalar(data["capture_time_s"])),
            "context": context,
            "state_sha256": state_sha,
            "hybrid_config": copy.deepcopy(configs["experiments"].get("hybrid_recovery", {})),
        }
    if capture["context"].get("swing_foot") not in {"left_foot", "right_foot"}:
        raise ValueError(f"capture context has no valid landed foot: {capture['context']}")
    if capture["qpos"].shape != (36,) or capture["qvel"].shape != (35,):
        raise ValueError(f"unexpected G1 touchdown state shape: {capture['qpos'].shape}, {capture['qvel'].shape}")
    return capture


def _free_joint_qpos_indices(model: HumanoidModel) -> np.ndarray:
    import mujoco

    free = [
        int(j)
        for j in range(model.model.njnt)
        if int(model.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if not free:
        raise ValueError("projection requires a floating-base model")
    start = int(model.model.jnt_qposadr[free[0]])
    return np.arange(start, start + 7, dtype=int)


def _foot_vertices_world(model: HumanoidModel, foot_name: str) -> np.ndarray:
    foot_index = 0 if foot_name == "left_foot" else 1
    local = np.asarray(model.adapter.foot_support_vertices_local[foot_index], dtype=float)
    T = model.body_pose(foot_name)
    homogeneous = np.c_[local, np.ones(len(local))]
    return (homogeneous @ T.T)[:, :3]


def _rotation_angle(R: np.ndarray, R_ref: np.ndarray) -> float:
    relative = np.asarray(R_ref, dtype=float).T @ np.asarray(R, dtype=float)
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _momentum(model: HumanoidModel, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    generalized = np.zeros(model.nv, dtype=float)
    mujoco.mj_mulM(model.model, model.data, generalized, np.asarray(qvel, dtype=float))
    base_position = np.asarray(model.data.xpos[model.body_ids["floating_base"]], dtype=float)
    com = model.center_of_mass()
    return generalized[:3].copy(), generalized[3:6] - np.cross(com - base_position, generalized[:3])


def _joint_info(model: HumanoidModel, qpos: np.ndarray, guard_rad: float) -> dict:
    lo, hi = model.joint_position_limits()
    joints = np.asarray(model.joint_positions(), dtype=float)
    limited = np.asarray(
        [bool(model.model.jnt_limited[j]) for j in model.joint_ids.values()], dtype=bool,
    )
    margins = np.minimum(joints - lo, hi - joints)
    guard_margins = margins - float(guard_rad)
    violation = bool(np.any(limited & (margins < 0.0)))
    # SLSQP/bound clipping can leave a few ulps below the declared bound.
    # Treat only a material violation as a failed guard; the raw configured
    # joint-limit violation remains strict enough to expose the capture error.
    guard_violation = bool(np.any(limited & (guard_margins < -1e-8)))
    names = list(model.joint_ids.keys())
    worst = int(np.argmin(margins)) if len(margins) else -1
    return {
        "joint_limit_violation": violation,
        "joint_guard_violation": guard_violation,
        "min_joint_limit_margin_rad": float(np.min(margins)) if len(margins) else float("nan"),
        "min_joint_guard_margin_rad": float(np.min(guard_margins)) if len(guard_margins) else float("nan"),
        "worst_joint_name": names[worst] if worst >= 0 else "",
        "joint_positions": joints.copy(),
        "joint_margins_rad": margins.copy(),
        "joint_names": names,
        "joint_lower_rad": np.asarray(lo, dtype=float),
        "joint_upper_rad": np.asarray(hi, dtype=float),
        "limited": limited,
    }


def _geometry_metrics(
    model: HumanoidModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    landed_foot: str,
    reference: dict | None = None,
    guard_rad: float = 0.02,
) -> dict:
    model.reset(qpos=np.asarray(qpos, dtype=float), qvel=np.asarray(qvel, dtype=float))
    com = model.center_of_mass().copy()
    foot_pose = model.body_pose(landed_foot).copy()
    pelvis_pose = model.body_pose("pelvis").copy()
    torso_pose = model.body_pose("torso").copy()
    vertices = _foot_vertices_world(model, landed_foot)
    margin = float(signed_support_margin(com[:2], convex_hull_2d(vertices[:, :2])))
    contact = model.actual_contact_data()
    flags = model.contact_flags()
    foot_index = 0 if landed_foot == "left_foot" else 1
    ground_id = model.geom_ids["ground"]
    ground_z = float(model.data.geom_xpos[ground_id][2])
    linear_momentum, angular_momentum = _momentum(model, qvel)
    joint = _joint_info(model, qpos, guard_rad)
    base_indices = _free_joint_qpos_indices(model)
    base_pose_qpos = np.asarray(qpos, dtype=float)[base_indices].copy()
    result = {
        "qpos": np.asarray(qpos, dtype=float).copy(),
        "qvel": np.asarray(qvel, dtype=float).copy(),
        "com_world": com,
        "foot_pose": foot_pose,
        "pelvis_pose": pelvis_pose,
        "torso_pose": torso_pose,
        "foot_vertices_world": vertices,
        "foot_min_z_m": float(np.min(vertices[:, 2])),
        "foot_max_z_m": float(np.max(vertices[:, 2])),
        "foot_ground_gap_min_m": float(np.min(vertices[:, 2]) - ground_z),
        "foot_ground_gap_max_m": float(np.max(vertices[:, 2]) - ground_z),
        "foot_normal_z": float(foot_pose[:3, 2] @ np.array([0.0, 0.0, 1.0])),
        "support_margin_m": margin,
        "geometric_contact_flags": np.asarray(flags, dtype=bool).copy(),
        "actual_contact_flags": np.asarray(contact.contact_flags, dtype=bool).copy(),
        "actual_normal_force_N": np.asarray(contact.normal_force_N, dtype=float).copy(),
        "linear_momentum_world": linear_momentum,
        "centroidal_angular_momentum_world": angular_momentum,
        "base_qpos": base_pose_qpos,
        **joint,
    }
    if reference is not None:
        result.update({
            "qpos_delta_norm": float(np.linalg.norm(np.asarray(qpos) - reference["qpos"])),
            "joint_delta_norm_rad": float(np.linalg.norm(result["joint_positions"] - reference["joint_positions"])),
            "com_delta_norm_m": float(np.linalg.norm(com - reference["com_world"])),
            "com_xy_delta_norm_m": float(np.linalg.norm(com[:2] - reference["com_world"][:2])),
            "pelvis_translation_delta_m": float(np.linalg.norm(pelvis_pose[:3, 3] - reference["pelvis_pose"][:3, 3])),
            "torso_translation_delta_m": float(np.linalg.norm(torso_pose[:3, 3] - reference["torso_pose"][:3, 3])),
            "landed_foot_translation_delta_m": float(np.linalg.norm(foot_pose[:3, 3] - reference["foot_pose"][:3, 3])),
            "landed_foot_xy_delta_m": float(np.linalg.norm(foot_pose[:2, 3] - reference["foot_pose"][:2, 3])),
            "landed_foot_orientation_delta_rad": _rotation_angle(foot_pose[:3, :3], reference["foot_pose"][:3, :3]),
            "base_translation_delta_m": float(np.linalg.norm(base_pose_qpos[:3] - reference["base_qpos"][:3])),
            "linear_momentum_delta_Ns": float(np.linalg.norm(linear_momentum - reference["linear_momentum_world"])),
            "angular_momentum_delta_Nms": float(np.linalg.norm(angular_momentum - reference["centroidal_angular_momentum_world"])),
        })
    return result


def _joint_bound_arrays(model: HumanoidModel, guard_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = model.joint_position_limits()
    limited = np.asarray(
        [bool(model.model.jnt_limited[j]) for j in model.joint_ids.values()], dtype=bool,
    )
    lower = np.asarray(lo, dtype=float).copy()
    upper = np.asarray(hi, dtype=float).copy()
    lower[limited] += float(guard_rad)
    upper[limited] -= float(guard_rad)
    if np.any(lower[limited] > upper[limited]):
        raise ValueError("joint guard is larger than at least one configured joint range")
    return lower, upper, limited


def _clip_to_joint_guard(model: HumanoidModel, qpos: np.ndarray, guard_rad: float) -> np.ndarray:
    lower, upper, limited = _joint_bound_arrays(model, guard_rad)
    projected = np.asarray(qpos, dtype=float).copy()
    values = projected[model.joint_qpos_indices].copy()
    values[limited] = np.clip(values[limited], lower[limited], upper[limited])
    projected[model.joint_qpos_indices] = values
    return projected


def _assemble_qpos(
    original_qpos: np.ndarray,
    model: HumanoidModel,
    joint_values: np.ndarray,
    base_xy: np.ndarray | None = None,
) -> np.ndarray:
    qpos = np.asarray(original_qpos, dtype=float).copy()
    qpos[model.joint_qpos_indices] = np.asarray(joint_values, dtype=float)
    if base_xy is not None:
        base_indices = _free_joint_qpos_indices(model)
        qpos[base_indices[:2]] = np.asarray(base_xy, dtype=float)
    return qpos


def _import_optimizer():
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("scipy.optimize is required on the SSH worker for projection") from exc
    return minimize


def _constraint_values(
    metrics: dict,
    reference: dict,
    ground_z: float,
    support_margin_min_m: float,
    foot_xy_trust_m: float,
    foot_orientation_trust_rad: float,
    foot_height_tolerance_m: float,
) -> np.ndarray:
    """Return SLSQP inequalities in the convention ``value >= 0``."""
    ref_min_z = float(reference["foot_min_z_m"])
    ref_foot_pose = reference["foot_pose"]
    ref_vertices = reference["foot_vertices_world"]
    # Keep the repair close to the measured landing pose.  Contact height is
    # referenced to the measured touchdown geometry, while the ground plane
    # prevents a projection from hiding penetration.
    return np.asarray([
        metrics["min_joint_guard_margin_rad"],
        metrics["foot_min_z_m"] - (ground_z - 0.001),
        metrics["foot_min_z_m"] - (ref_min_z - foot_height_tolerance_m),
        (ref_min_z + foot_height_tolerance_m) - metrics["foot_min_z_m"],
        metrics["foot_normal_z"] - math.cos(foot_orientation_trust_rad),
        math.cos(
            _rotation_angle(metrics["foot_pose"][:3, :3], ref_foot_pose[:3, :3])
        ) - math.cos(foot_orientation_trust_rad),
        foot_xy_trust_m - float(np.linalg.norm(metrics["foot_pose"][:2, 3] - ref_foot_pose[:2, 3])),
        0.015 - float(np.max(metrics["foot_vertices_world"][:, 2]) - np.min(metrics["foot_vertices_world"][:, 2])),
        metrics["support_margin_m"] - float(support_margin_min_m),
        # Keep the support footprint from changing by more than a small
        # amount relative to the measured contact footprint.
        0.10 - float(np.linalg.norm(np.mean(metrics["foot_vertices_world"], axis=0)[:2] - np.mean(ref_vertices, axis=0)[:2])),
    ], dtype=float)


def _projection_objective(
    qpos: np.ndarray,
    original_qpos: np.ndarray,
    model: HumanoidModel,
    qvel: np.ndarray,
    reference: dict,
    base_xy: np.ndarray | None = None,
    base_xy_reference: np.ndarray | None = None,
) -> float:
    joint_values = np.asarray(qpos[model.joint_qpos_indices], dtype=float)
    ref_joint_values = np.asarray(original_qpos[model.joint_qpos_indices], dtype=float)
    joint_scale = np.maximum(0.05, 0.25 * (reference["joint_upper_rad"] - reference["joint_lower_rad"]))
    objective = float(np.sum(((joint_values - ref_joint_values) / joint_scale) ** 2))
    metrics = _geometry_metrics(model, qpos, qvel, reference["landed_foot"], reference=reference)
    objective += float(np.sum(((metrics["com_world"] - reference["com_world"]) / np.array([0.02, 0.02, 0.03])) ** 2))
    objective += float((metrics["landed_foot_translation_delta_m"] / 0.02) ** 2)
    objective += float((metrics["landed_foot_orientation_delta_rad"] / 0.10) ** 2)
    if base_xy is not None and base_xy_reference is not None:
        objective += float(np.sum(((np.asarray(base_xy) - np.asarray(base_xy_reference)) / 0.01) ** 2))
    return objective


def _project_joint_contact(
    model: HumanoidModel,
    capture: dict,
    reference: dict,
    guard_rad: float,
    foot_xy_trust_m: float,
    foot_orientation_trust_rad: float,
    foot_height_tolerance_m: float,
) -> dict:
    minimize = _import_optimizer()
    original_qpos = np.asarray(capture["qpos"], dtype=float)
    qvel = np.asarray(capture["qvel"], dtype=float)
    joint0 = original_qpos[model.joint_qpos_indices].copy()
    lower, upper, limited = _joint_bound_arrays(model, guard_rad)
    bounds = [
        (float(lower[i]), float(upper[i])) if bool(limited[i]) else (None, None)
        for i in range(len(joint0))
    ]
    ground_z = float(reference["ground_z_m"])

    def make_qpos(values):
        return _assemble_qpos(original_qpos, model, values)

    def objective(values):
        return _projection_objective(make_qpos(values), original_qpos, model, qvel, reference)

    def constraints(values):
        metrics = _geometry_metrics(model, make_qpos(values), qvel, reference["landed_foot"], reference=reference, guard_rad=guard_rad)
        return _constraint_values(
            metrics,
            reference,
            ground_z,
            support_margin_min_m=0.0,
            foot_xy_trust_m=foot_xy_trust_m,
            foot_orientation_trust_rad=foot_orientation_trust_rad,
            foot_height_tolerance_m=foot_height_tolerance_m,
        )

    x0 = _clip_to_joint_guard(model, original_qpos, guard_rad)[model.joint_qpos_indices]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": 120, "ftol": 1e-8, "disp": False},
    )
    projected_qpos = make_qpos(result.x if np.all(np.isfinite(result.x)) else x0)
    metrics = _geometry_metrics(model, projected_qpos, qvel, reference["landed_foot"], reference=reference, guard_rad=guard_rad)
    values = constraints(projected_qpos[model.joint_qpos_indices])
    feasible = bool(np.all(np.isfinite(values)) and np.min(values) >= -1e-5 and np.all(np.isfinite(projected_qpos)))
    return {
        "qpos": projected_qpos,
        "qvel": qvel.copy(),
        "metrics": metrics,
        "feasible": feasible,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(getattr(result, "nit", -1)),
        "solver_objective": float(result.fun) if np.isfinite(result.fun) else float("nan"),
        "max_constraint_violation": float(max(0.0, -float(np.min(values)))) if len(values) else float("nan"),
        "constraint_values": values,
    }


def _qp_headroom_metrics(model: HumanoidModel, configs: dict, capture: dict, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    landed = str(capture["context"]["swing_foot"])
    model.reset(qpos=qpos, qvel=qvel)
    controller, active_contacts = _make_replay_controller(
        model, configs, capture["context"], "landed_support_momentum_capture",
    )
    result = controller.solve()
    limits = np.maximum(np.max(np.abs(model.actuator_limits), axis=1), 1e-9)
    torque_utilization = float(np.max(np.abs(result.control) / limits)) if len(result.control) else float("nan")
    friction_utilization = 10.0
    if result.success and len(result.contact_wrench):
        mu = float(configs["controller"].get("friction_coefficient", 0.7))
        utilizations = []
        for index in range(len(active_contacts)):
            wrench = np.asarray(result.contact_wrench[6 * index : 6 * index + 6], dtype=float)
            fz = max(float(wrench[2]), 0.0)
            utilizations.append(max(abs(float(wrench[0])), abs(float(wrench[1]))) / max(mu * fz, 1e-9))
        friction_utilization = float(max(utilizations, default=10.0))
    if not np.isfinite(torque_utilization):
        torque_utilization = 10.0
    if not np.isfinite(friction_utilization):
        friction_utilization = 10.0
    return {
        "qp_status": str(result.status),
        "qp_success": bool(result.success),
        "qp_torque_utilization": torque_utilization,
        "qp_friction_utilization": friction_utilization,
        "qp_contact_slack_norm": float(result.contact_slack_norm),
        "qp_dynamics_residual_norm": float(result.dynamics_residual_norm),
        "qp_contact_acceleration_residual_norm": float(result.contact_acceleration_residual_norm),
        "qp_friction_margin": float(result.friction_margin),
        "qp_message": str(result.message),
    }


def _project_headroom(
    model: HumanoidModel,
    capture: dict,
    reference: dict,
    seed_projection: dict,
    configs: dict,
    guard_rad: float,
    foot_xy_trust_m: float,
    foot_orientation_trust_rad: float,
    foot_height_tolerance_m: float,
    base_xy_trust_m: float,
) -> dict:
    minimize = _import_optimizer()
    original_qpos = np.asarray(capture["qpos"], dtype=float)
    qvel = np.asarray(capture["qvel"], dtype=float)
    joint0 = np.asarray(seed_projection["qpos"][model.joint_qpos_indices], dtype=float)
    lower, upper, limited = _joint_bound_arrays(model, guard_rad)
    base_indices = _free_joint_qpos_indices(model)
    base0 = np.asarray(original_qpos[base_indices[:2]], dtype=float)
    x0 = np.r_[joint0, base0]
    bounds = [
        (float(lower[i]), float(upper[i])) if bool(limited[i]) else (None, None)
        for i in range(len(joint0))
    ]
    bounds.extend([(float(value - base_xy_trust_m), float(value + base_xy_trust_m)) for value in base0])
    ground_z = float(reference["ground_z_m"])

    def make_qpos(values):
        return _assemble_qpos(original_qpos, model, values[:-2], base_xy=values[-2:])

    def objective(values):
        return _projection_objective(
            make_qpos(values), original_qpos, model, qvel, reference,
            base_xy=values[-2:], base_xy_reference=base0,
        )

    def constraints(values):
        qpos = make_qpos(values)
        metrics = _geometry_metrics(model, qpos, qvel, reference["landed_foot"], reference=reference, guard_rad=guard_rad)
        geometry = _constraint_values(
            metrics,
            reference,
            ground_z,
            support_margin_min_m=0.005,
            foot_xy_trust_m=foot_xy_trust_m,
            foot_orientation_trust_rad=foot_orientation_trust_rad,
            foot_height_tolerance_m=foot_height_tolerance_m,
        )
        qp = _qp_headroom_metrics(model, configs, capture, qpos, qvel)
        return np.r_[
            geometry,
            0.90 - qp["qp_torque_utilization"],
            0.90 - qp["qp_friction_utilization"],
            1.0 if qp["qp_success"] else -1.0,
        ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": 60, "ftol": 1e-7, "disp": False},
    )
    values = result.x if np.all(np.isfinite(result.x)) else x0
    projected_qpos = make_qpos(values)
    metrics = _geometry_metrics(model, projected_qpos, qvel, reference["landed_foot"], reference=reference, guard_rad=guard_rad)
    qp = _qp_headroom_metrics(model, configs, capture, projected_qpos, qvel)
    constraint_values = constraints(values)
    feasible = bool(
        np.all(np.isfinite(constraint_values))
        and np.min(constraint_values) >= -1e-5
        and np.all(np.isfinite(projected_qpos))
    )
    return {
        "qpos": projected_qpos,
        "qvel": qvel.copy(),
        "metrics": metrics,
        "qp_metrics": qp,
        "feasible": feasible,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(getattr(result, "nit", -1)),
        "solver_objective": float(result.fun) if np.isfinite(result.fun) else float("nan"),
        "max_constraint_violation": float(max(0.0, -float(np.min(constraint_values)))) if len(constraint_values) else float("nan"),
        "constraint_values": constraint_values,
        "base_xy_trust_region_m": float(base_xy_trust_m),
    }


def _state_summary(stage: str, state: dict, capture: dict, reference: dict, qp: dict | None = None) -> dict:
    metrics = state["metrics"]
    row = {
        "stage": stage,
        "input_state_sha256": capture["state_sha256"],
        "projected_state_sha256": state.get("state_sha256", ""),
        "projection_feasible": bool(state.get("feasible", True)),
        "is_reference_state": bool(state.get("is_reference_state", False)),
        "solver_success": bool(state.get("solver_success", True)),
        "solver_message": state.get("solver_message", ""),
        "solver_iterations": int(state.get("solver_iterations", 0)),
        "solver_objective": float(state.get("solver_objective", 0.0)),
        "max_constraint_violation": float(state.get("max_constraint_violation", 0.0)),
        "joint_limit_violation": bool(metrics["joint_limit_violation"]),
        "joint_guard_violation": bool(metrics["joint_guard_violation"]),
        "min_joint_limit_margin_rad": float(metrics["min_joint_limit_margin_rad"]),
        "min_joint_guard_margin_rad": float(metrics["min_joint_guard_margin_rad"]),
        "worst_joint_name": metrics["worst_joint_name"],
        "support_margin_m": float(metrics["support_margin_m"]),
        "geometric_contact_landed": bool(metrics["geometric_contact_flags"][0 if reference["landed_foot"] == "left_foot" else 1]),
        "actual_contact_landed_after_forward": bool(metrics["actual_contact_flags"][0 if reference["landed_foot"] == "left_foot" else 1]),
        "foot_ground_gap_min_m": float(metrics["foot_ground_gap_min_m"]),
        "foot_ground_gap_max_m": float(metrics["foot_ground_gap_max_m"]),
        "foot_normal_z": float(metrics["foot_normal_z"]),
        "com_x_m": float(metrics["com_world"][0]),
        "com_y_m": float(metrics["com_world"][1]),
        "com_z_m": float(metrics["com_world"][2]),
        "qpos_delta_norm": float(metrics.get("qpos_delta_norm", 0.0)),
        "joint_delta_norm_rad": float(metrics.get("joint_delta_norm_rad", 0.0)),
        "com_delta_norm_m": float(metrics.get("com_delta_norm_m", 0.0)),
        "com_xy_delta_norm_m": float(metrics.get("com_xy_delta_norm_m", 0.0)),
        "pelvis_translation_delta_m": float(metrics.get("pelvis_translation_delta_m", 0.0)),
        "torso_translation_delta_m": float(metrics.get("torso_translation_delta_m", 0.0)),
        "landed_foot_translation_delta_m": float(metrics.get("landed_foot_translation_delta_m", 0.0)),
        "landed_foot_xy_delta_m": float(metrics.get("landed_foot_xy_delta_m", 0.0)),
        "landed_foot_orientation_delta_rad": float(metrics.get("landed_foot_orientation_delta_rad", 0.0)),
        "base_translation_delta_m": float(metrics.get("base_translation_delta_m", 0.0)),
        "linear_momentum_norm_Ns": float(np.linalg.norm(metrics["linear_momentum_world"])),
        "centroidal_angular_momentum_norm_Nms": float(np.linalg.norm(metrics["centroidal_angular_momentum_world"])),
        "qvel_preserved_exactly": True,
        "qp_status": "",
        "qp_success": False,
        "qp_torque_utilization": float("nan"),
        "qp_friction_utilization": float("nan"),
        "qp_contact_slack_norm": float("nan"),
        "qp_dynamics_residual_norm": float("nan"),
        "qp_contact_acceleration_residual_norm": float("nan"),
        "qp_friction_margin": float("nan"),
        "qp_message": "",
    }
    if qp:
        row.update(qp)
    return row


def _state_digest(qpos: np.ndarray, qvel: np.ndarray, stage: str, input_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(qpos, dtype="<f8").tobytes())
    digest.update(np.asarray(qvel, dtype="<f8").tobytes())
    digest.update(str(stage).encode("utf-8"))
    digest.update(str(input_sha).encode("utf-8"))
    return digest.hexdigest()


def _save_projected_state(path: Path, stage: str, state: dict, capture: dict, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=np.asarray(state["qpos"], dtype=float),
        qvel=np.asarray(state["qvel"], dtype=float),
        stage=np.asarray(stage),
        input_state_sha256=np.asarray(capture["state_sha256"]),
        projected_state_sha256=np.asarray(state["state_sha256"]),
        projection_summary_json=np.asarray(_json(summary)),
    )


def _replay_projected_state(
    configs: dict,
    capture: dict,
    state: dict,
    stage: str,
    variant: str,
    output_root: Path,
    projection_summary: dict,
    replay_duration_s: float,
    seed: int,
) -> dict:
    replay_capture = dict(capture)
    replay_capture["qpos"] = np.asarray(state["qpos"], dtype=float).copy()
    replay_capture["qvel"] = np.asarray(state["qvel"], dtype=float).copy()
    # Keep the original controller context and original CoM reference.  The
    # projection is therefore tested as a state repair, not as a target retune.
    model, run, active_contacts, initial_qvel = _replay(
        configs, replay_capture, variant, replay_duration_s, seed,
    )
    observables = _replay_observables(model, run, replay_capture, active_contacts)
    summary = _stability_summary(run, observables, replay_capture, active_contacts, variant, configs)
    capture_name = _safe_name(Path(str(capture["input_path"])).stem)
    trial_id = f"{capture_name}_{stage}_{variant}"
    data_root = output_root / "data" / "trials"
    figures_root = output_root / "figures"
    observable_path = data_root / f"{_safe_name(trial_id)}_observables.npz"
    trial_path = data_root / f"{_safe_name(trial_id)}.npz"
    np.savez_compressed(
        observable_path,
        **observables,
        replay_time_s=np.asarray(run.log.arrays()["time_s"], dtype=float),
    )
    metadata = {
        "experiment": "project_touchdown_states",
        "phase": "replay",
        "trial_id": trial_id,
        "projection_stage": stage,
        "replay_variant": variant,
        "external_push_removed": True,
        "planning_removed": True,
        "swing_generation_removed": True,
        "landing_gate_removed": True,
        "input_state_path": str(capture["input_path"]),
        "input_state_file_sha256": capture["input_file_sha256"],
        "input_touchdown_state_sha256": capture["state_sha256"],
        "projected_state_sha256": state["state_sha256"],
        "projected_state_summary": projection_summary,
        "controller_context_preserved": True,
        "qvel_preserved_exactly": True,
        "active_contacts": list(active_contacts),
        "initial_generalized_velocity_norm": float(np.linalg.norm(initial_qvel)),
        "observable_path": str(observable_path.relative_to(output_root).as_posix()),
        "diagnostic_summary": summary,
    }
    from common import save_run  # local import keeps module import lightweight

    save_run(run, trial_path, metadata)
    _plot_replay(
        np.asarray(run.log.arrays()["time_s"], dtype=float),
        run.log.arrays(),
        observables,
        summary,
        figures_root / f"{_safe_name(trial_id)}.png",
    )
    row = dict(summary)
    row.update({
        "trial_id": trial_id,
        "projection_stage": stage,
        "replay_variant": variant,
        "input_touchdown_state_sha256": capture["state_sha256"],
        "projected_state_sha256": state["state_sha256"],
        "projection_feasible": bool(state.get("feasible", True)),
        "projection_qpos_delta_norm": projection_summary["qpos_delta_norm"],
        "projection_com_delta_norm_m": projection_summary["com_delta_norm_m"],
        "projection_landed_foot_translation_delta_m": projection_summary["landed_foot_translation_delta_m"],
        "projection_support_margin_m": projection_summary["support_margin_m"],
        "projection_min_joint_guard_margin_rad": projection_summary["min_joint_guard_margin_rad"],
    })
    return row


def _projection_for_capture(
    configs: dict,
    capture: dict,
    guard_rad: float,
    foot_xy_trust_m: float,
    foot_orientation_trust_rad: float,
    foot_height_tolerance_m: float,
    base_xy_trust_m: float,
) -> tuple[list[dict], dict[str, dict], dict]:
    landed = str(capture["context"]["swing_foot"])
    model = make_model(configs)
    original_metrics = _geometry_metrics(model, capture["qpos"], capture["qvel"], landed, guard_rad=guard_rad)
    reference = dict(original_metrics)
    reference["landed_foot"] = landed
    reference["ground_z_m"] = float(model.data.geom_xpos[model.geom_ids["ground"]][2])
    reference["joint_lower_rad"] = original_metrics["joint_lower_rad"].copy()
    reference["joint_upper_rad"] = original_metrics["joint_upper_rad"].copy()

    states: dict[str, dict] = {}
    original = {
        "qpos": capture["qpos"].copy(),
        "qvel": capture["qvel"].copy(),
        "metrics": original_metrics,
        "feasible": True,
        "is_reference_state": True,
        "solver_success": True,
        "solver_message": "captured touchdown state; no projection",
        "solver_iterations": 0,
        "solver_objective": 0.0,
        "max_constraint_violation": 0.0,
    }
    original["state_sha256"] = capture["state_sha256"]
    states["original_touchdown"] = original

    joint_qpos = _clip_to_joint_guard(model, capture["qpos"], guard_rad)
    joint_metrics = _geometry_metrics(model, joint_qpos, capture["qvel"], landed, reference=reference, guard_rad=guard_rad)
    states["joint_limit_guard"] = {
        "qpos": joint_qpos,
        "qvel": capture["qvel"].copy(),
        "metrics": joint_metrics,
        "feasible": bool(not joint_metrics["joint_guard_violation"]),
        "solver_success": True,
        "solver_message": "deterministic per-joint clipping into configured limits minus guard",
        "solver_iterations": 0,
        "solver_objective": float(np.linalg.norm(joint_qpos - capture["qpos"])),
        "max_constraint_violation": 0.0,
    }

    contact_projection = _project_joint_contact(
        model,
        capture,
        reference,
        guard_rad,
        foot_xy_trust_m,
        foot_orientation_trust_rad,
        foot_height_tolerance_m,
    )
    states["joint_limit_contact_support"] = contact_projection

    qp_initial = _qp_headroom_metrics(
        model,
        configs,
        capture,
        contact_projection["qpos"],
        capture["qvel"],
    )
    headroom_needed = bool(
        contact_projection["feasible"]
        and (
            not qp_initial["qp_success"]
            or qp_initial["qp_torque_utilization"] > 0.90
            or qp_initial["qp_friction_utilization"] > 0.90
            or contact_projection["metrics"]["support_margin_m"] < 0.005
        )
    )
    if headroom_needed:
        headroom_projection = _project_headroom(
            model,
            capture,
            reference,
            contact_projection,
            configs,
            guard_rad,
            foot_xy_trust_m,
            foot_orientation_trust_rad,
            foot_height_tolerance_m,
            base_xy_trust_m,
        )
        states["joint_limit_contact_headroom"] = headroom_projection
        headroom_note = "attempted because stage-2 one-step QP screen lacked declared headroom"
    else:
        states["joint_limit_contact_headroom"] = {
            **contact_projection,
            "feasible": False,
            "solver_success": False,
            "solver_message": "skipped: stage-2 one-step QP screen already passed declared headroom",
            "qp_metrics": qp_initial,
        }
        headroom_note = "skipped because stage-2 one-step QP screen passed declared headroom"

    rows = []
    for stage in PROJECTION_STAGES:
        state = states[stage]
        qp = state.get("qp_metrics")
        if stage == "joint_limit_contact_support":
            state["qp_metrics"] = qp_initial
            qp = qp_initial
        state["state_sha256"] = (
            capture["state_sha256"]
            if stage == "original_touchdown"
            else _state_digest(state["qpos"], state["qvel"], stage, capture["state_sha256"])
        )
        rows.append(_state_summary(stage, state, capture, reference, qp=qp))
    return rows, states, {
        "reference": reference,
        "headroom_needed": headroom_needed,
        "headroom_note": headroom_note,
        "stage2_qp_screen": qp_initial,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--replay-duration", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-guard-rad", type=float, default=0.02)
    parser.add_argument("--foot-xy-trust-m", type=float, default=0.05)
    parser.add_argument("--foot-orientation-trust-deg", type=float, default=12.0)
    parser.add_argument("--foot-height-tolerance-m", type=float, default=0.003)
    parser.add_argument("--base-xy-trust-m", type=float, default=0.03)
    args = parser.parse_args()

    configs = load_configs(ROOT)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output root: {output_root}")
    for directory in (
        output_root / "inputs",
        output_root / "data" / "projected_states",
        output_root / "data" / "trials",
        output_root / "figures",
        output_root / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    captures = [_load_capture(path.resolve(), configs) for path in args.states]
    input_records = []
    for capture in captures:
        destination = output_root / "inputs" / f"{_safe_name(capture['input_path'].stem)}.npz"
        shutil.copy2(capture["input_path"], destination)
        input_records.append({
            "original_path": str(capture["input_path"]),
            "archived_input_path": str(destination.relative_to(output_root).as_posix()),
            "input_file_sha256": capture["input_file_sha256"],
            "touchdown_state_sha256": capture["state_sha256"],
            "capture_time_s": capture["time_s"],
            "landed_foot": capture["context"]["swing_foot"],
        })

    command = " ".join(sys.argv)
    source = {
        "source_commit": os.environ.get("SE3_SOURCE_VERSION", "unknown"),
        "source_tree_sha256": os.environ.get("SE3_SOURCE_TREE_SHA256", "unknown"),
        "remote_source_root": os.environ.get("SE3_SOURCE_ROOT", "unknown"),
        "execution_environment_id": os.environ.get("SE3_EXECUTION_ENV", "unknown"),
    }
    write_execution_manifest(
        output_root / "logs" / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "project_touchdown_states",
            "question": "Does a nearby constraint-consistent touchdown state exist that the existing WBC can stabilize?",
            "run_id": output_root.name,
            "source_provenance": source,
            "input_states": input_records,
            "projection_stages": list(PROJECTION_STAGES),
            "projection_definitions": {
                "joint_limit_guard": "actuated joint qpos clipped to configured range with a declared 0.02 rad guard; qvel unchanged",
                "joint_limit_contact_support": "nearest SLSQP projection with base pose fixed, joint guard, landed-foot height/orientation/XY trust region, ground non-penetration, and landed-foot CoM support margin >= 0",
                "joint_limit_contact_headroom": "conditional nearest SLSQP projection with the same geometric constraints, base XY trust region <= 0.03 m, one-step landed-support QP torque/friction utilization <= 0.90, and successful QP",
            },
            "projection_parameters": {
                "joint_guard_rad": float(args.joint_guard_rad),
                "foot_xy_trust_m": float(args.foot_xy_trust_m),
                "foot_orientation_trust_deg": float(args.foot_orientation_trust_deg),
                "foot_height_tolerance_m": float(args.foot_height_tolerance_m),
                "base_xy_trust_m": float(args.base_xy_trust_m),
            },
            "state_preservation": {
                "qvel": "exactly preserved for every projection and replay",
                "controller_context": "exactly preserved; original capture CoM remains the replay reference",
                "external_push": False,
                "planning_removed": True,
                "swing_generation_removed": True,
                "landing_gate_removed": True,
            },
            "interpretation_warning": "Projected states are diagnostic counterfactuals. A stable replay does not establish that the projection is reachable from the original swing trajectory.",
            "command": command,
        },
    )

    projection_rows = []
    replay_rows = []
    study_records = []
    for capture in captures:
        rows, states, study = _projection_for_capture(
            configs,
            capture,
            float(args.joint_guard_rad),
            float(args.foot_xy_trust_m),
            float(np.deg2rad(args.foot_orientation_trust_deg)),
            float(args.foot_height_tolerance_m),
            float(args.base_xy_trust_m),
        )
        capture_name = _safe_name(capture["input_path"].stem)
        for row in rows:
            row.update({
                "capture_id": capture_name,
                "landed_foot": capture["context"]["swing_foot"],
                "source_commit": source["source_commit"],
            })
            projection_rows.append(row)
            stage = str(row["stage"])
            state = states[stage]
            state_path = output_root / "data" / "projected_states" / f"{capture_name}_{stage}.npz"
            _save_projected_state(state_path, stage, state, capture, row)
            row["projected_state_path"] = str(state_path.relative_to(output_root).as_posix())

            # The comparison uses the current production double-support
            # interpretation and the existing landed-support momentum-capture
            # diagnostic for every state.  No gains are changed.
            for variant in ("double_support_current", "landed_support_momentum_capture"):
                replay_rows.append(_replay_projected_state(
                    configs,
                    capture,
                    state,
                    stage,
                    variant,
                    output_root,
                    row,
                    float(args.replay_duration),
                    int(args.seed),
                ))
        study_records.append({
            "capture_id": capture_name,
            "input_state_sha256": capture["state_sha256"],
            "headroom_needed": study["headroom_needed"],
            "headroom_note": study["headroom_note"],
            "stage2_qp_screen": study["stage2_qp_screen"],
        })

    write_csv(projection_rows, output_root / "projection_summary.csv")
    write_csv(replay_rows, output_root / "replay_summary.csv")
    summary = {
        "run_id": output_root.name,
        "capture_count": len(captures),
        "projection_count": len(projection_rows),
        "replay_count": len(replay_rows),
        "projection_stages": list(PROJECTION_STAGES),
        "strict_success_count": int(sum(bool(row.get("custom_stable_final_window", False)) for row in replay_rows)),
        "classifier_success_count": int(sum(bool(row.get("success", False)) for row in replay_rows)),
        "study_records": study_records,
        "source_provenance": source,
        "artifact_note": "Original input snapshots, projected qpos/qvel states, optimization summaries, raw replay runs, observables, figures, CSVs, and manifest are retained in this new root.",
    }
    (output_root / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
