"""Map reachable and stabilizable touchdown states without changing production control.

The study has two deliberately separate layers:

1. Run the existing one-step MuJoCo pipeline with a small, signed first-foot
   target offset along the measured disturbance direction.  This tests whether
   the plant can actually bring the swing foot to a nearby target and obtain a
   sustained load-bearing touchdown.
2. Replay every confirmed touchdown from its exact logged ``qpos``/``qvel``
   state with planning, swing generation, and the external push removed.  The
   replay compares the current double-support interpretation with a landed-foot
   momentum-capture controller and two velocity counterfactuals.

The target offsets are an oracle diagnostic only.  They are not production
parameters and no result is called a recovery unless it satisfies the strict
post-touchdown stability criterion recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_configs,
    make_model,
    make_push,
    recovery_config,
    save_run,
    write_csv,
    write_execution_manifest,
)
from diagnose_step_viability import DiagnosticHybridController  # noqa: E402
from replay_touchdown_recoverability import (  # noqa: E402
    _capture_controller_context,
    _plot_replay,
    _replay,
    _replay_observables,
    _safe_name,
    _state_digest,
    _stability_summary,
    _write_touchdown_snapshot,
)
from se3_whole_body_control.config import resolve_model_path  # noqa: E402
from se3_whole_body_control.control.support import convex_hull_2d, signed_support_margin  # noqa: E402
from se3_whole_body_control.control.tasks import com_jacobian  # noqa: E402
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner  # noqa: E402


DEFAULT_OFFSETS_M = (-0.10, -0.05, 0.0, 0.05, 0.10)
REPLAY_PLAN = (
    ("double_support_current", 1.0),
    ("landed_support_momentum_capture", 1.0),
    ("landed_support_momentum_capture", 0.5),
    ("landed_support_zero_momentum", 0.0),
)


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-12)


class OracleFootPlacementController(DiagnosticHybridController):
    """Diagnostic first-step target alternative; production code is untouched."""

    def __init__(self, *args, foot_offset_m: float, **kwargs):
        self.foot_offset_m = float(foot_offset_m)
        self.touchdown_context: dict | None = None
        self.first_step_target_world: np.ndarray | None = None
        self.first_step_support_center_world: np.ndarray | None = None
        self.first_step_command_reach_m: float | None = None
        super().__init__(*args, **kwargs)

    def reset_trial(self) -> None:
        super().reset_trial()
        self.touchdown_context = None
        self.first_step_target_world = None
        self.first_step_support_center_world = None
        self.first_step_command_reach_m = None

    def _make_step_target(self, swing_foot: str, support_foot: str, direction_xy: np.ndarray) -> np.ndarray:
        target = super()._make_step_target(swing_foot, support_foot, direction_xy)
        if self.step_count == 0 and abs(self.foot_offset_m) > 0.0:
            target[:2, 3] = self._clamp_target_to_reach(
                target[:2, 3] + normalize_direction(direction_xy) * self.foot_offset_m,
                support_foot,
            )
        if self.step_count == 0:
            support_center = self.model.body_pose(support_foot)[:3, 3].copy()
            self.first_step_target_world = target[:3, 3].copy()
            self.first_step_support_center_world = support_center
            self.first_step_command_reach_m = float(np.linalg.norm(target[:2, 3] - support_center[:2]))
        return target

    def solve(self):
        result = super().solve()
        if self.touchdown_context is None and str(result.diagnostics.get("event_label") or "") == "TOUCHDOWN":
            event = next(
                (item for item in reversed(self.events) if item.get("label") == "TOUCHDOWN"),
                {},
            )
            self.touchdown_context = _capture_controller_context(
                self,
                event,
                self.model.center_of_mass().copy(),
            )
            self.touchdown_context.update({
                "oracle_foot_offset_m": float(self.foot_offset_m),
                "target_command_world": self.first_step_target_world.tolist() if self.first_step_target_world is not None else [],
                "support_center_at_target_world": self.first_step_support_center_world.tolist() if self.first_step_support_center_world is not None else [],
                "target_command_reach_m": self.first_step_command_reach_m,
            })
            self.touchdown_context["capture_controller_time_s"] = float(self.model.data.time)
        return result


def _source_metadata() -> dict:
    return {
        "source_commit": os.environ.get("SE3_SOURCE_VERSION", "unknown"),
        "source_tree_sha256": os.environ.get("SE3_SOURCE_TREE_SHA256", "unknown"),
        "source_tree_hash_algorithm": os.environ.get("SE3_SOURCE_TREE_HASH_ALGORITHM", "unknown"),
        "remote_source_root": os.environ.get("SE3_SOURCE_ROOT", "unknown"),
        "execution_environment_id": os.environ.get("SE3_EXECUTION_ENV", "unknown"),
    }


def _event_list(run) -> list[dict]:
    return [
        dict(event)
        for event in run.metadata.get("controller_summary", {}).get("events", [])
    ]


def _event_index(run, label: str) -> int | None:
    labels = run.log.arrays().get("event_label", np.asarray([], dtype=str)).astype(str)
    indices = np.flatnonzero(labels == str(label))
    return int(indices[0]) if len(indices) else None


def _event_time(run, label: str) -> float | None:
    for event in _event_list(run):
        if event.get("label") == label:
            return float(event.get("time_s", np.nan))
    return None


def _target_error_series(run) -> np.ndarray:
    arrays = run.log.arrays()
    target = np.asarray(
        arrays.get("planned_foot_target_world", np.full((len(arrays["time_s"]), 3), np.nan)),
        dtype=float,
    )
    foot_xy = np.asarray(
        arrays.get("foot_xy_world", np.full((len(arrays["time_s"]), 4), np.nan)),
        dtype=float,
    ).reshape(-1, 2, 2)
    swing = arrays.get("swing_foot", np.asarray([""] * len(target))).astype(str)
    errors = np.full(len(target), np.nan, dtype=float)
    for index in range(len(target)):
        if not np.all(np.isfinite(target[index])):
            continue
        foot_index = 0 if swing[index] == "left_foot" else 1 if swing[index] == "right_foot" else -1
        if foot_index >= 0:
            errors[index] = float(np.linalg.norm(foot_xy[index, foot_index] - target[index, :2]))
    return errors


def _state_momentum(model, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    com_velocity = com_jacobian(model) @ qvel
    generalized_momentum = np.zeros(model.nv, dtype=float)
    mujoco.mj_mulM(model.model, model.data, generalized_momentum, qvel)
    total_mass = float(np.sum(model.model.body_mass))
    linear_momentum = generalized_momentum[:3].copy()
    base_position = np.asarray(model.data.xpos[model.body_ids["floating_base"]], dtype=float)
    centroidal_angular_momentum = generalized_momentum[3:6] - np.cross(
        model.center_of_mass() - base_position,
        generalized_momentum[:3],
    )
    consistency_error = np.asarray(
        [np.linalg.norm(linear_momentum - total_mass * com_velocity)],
        dtype=float,
    )
    return com_velocity, linear_momentum, np.r_[centroidal_angular_momentum, consistency_error]


def _capture_geometry_metrics(model, capture: dict) -> dict:
    arrays = capture["run"].log.arrays()
    index = int(capture["index"])
    model.reset(qpos=capture["qpos"], qvel=capture["qvel"])
    com = model.center_of_mass().copy()
    com_velocity, linear_momentum, angular_and_error = _state_momentum(model, capture["qvel"])
    contact = model.actual_contact_data()
    context = capture["context"]
    landed_foot = str(context["swing_foot"])
    support_foot = str(context["support_foot"])
    landed_index = 0 if landed_foot == "left_foot" else 1
    support_index = 0 if support_foot == "left_foot" else 1
    vertices = np.asarray(model.foot_support_vertices_world(), dtype=float)
    landed_margin = float(signed_support_margin(com[:2], convex_hull_2d(vertices[landed_index])))
    old_support_margin = float(signed_support_margin(com[:2], convex_hull_2d(vertices[support_index])))
    target = np.asarray(arrays["planned_foot_target_world"][index], dtype=float)
    foot_xy = np.asarray(arrays["foot_xy_world"][index], dtype=float).reshape(2, 2)
    actual_landing_error = float(np.linalg.norm(foot_xy[landed_index] - target[:2])) if np.all(np.isfinite(target)) else float("nan")
    errors = _target_error_series(capture["run"])
    swing_complete_index = _event_index(capture["run"], "SWING_COMPLETE")
    swing_complete_error = float(errors[swing_complete_index]) if swing_complete_index is not None else float("nan")
    landing_tolerance = float(capture["hybrid_config"].get("landing_position_tolerance_m", 0.08))
    command_reach = float(context.get("target_command_reach_m", float("nan")))
    lo, hi = model.joint_position_limits()
    q = model.joint_positions()
    limited = np.asarray(
        [bool(model.model.jnt_limited[j]) for j in model.joint_ids.values()],
        dtype=bool,
    )
    joint_margin = np.minimum(q - lo, hi - q)
    finite_joint_margin = joint_margin[limited] if np.any(limited) else joint_margin
    post_friction = np.asarray(arrays["actual_friction_utilization_post_step"][index], dtype=float)
    post_normal = np.asarray(arrays["actual_normal_force_post_step_N"][index], dtype=float)
    return {
        "capture_state_sha256": capture["state_sha256"],
        "capture_time_s": float(capture["time_s"]),
        "touchdown_confirmed": True,
        "swing_complete_confirmed": swing_complete_index is not None,
        "command_reach_m": command_reach,
        "command_within_max_reach": bool(
            np.isfinite(command_reach)
            and command_reach <= float(capture["hybrid_config"].get("max_step_reach_m", np.nan)) + 1e-9,
        ),
        "swing_complete_target_error_m": swing_complete_error,
        "reference_foot_error_within_landing_tolerance": bool(
            np.isfinite(swing_complete_error)
            and swing_complete_error <= landing_tolerance,
        ),
        "touchdown_target_error_m": actual_landing_error,
        "geometric_landing_reachable": bool(
            np.isfinite(actual_landing_error)
            and actual_landing_error <= landing_tolerance,
        ),
        "landed_foot": landed_foot,
        "support_foot": support_foot,
        "landed_foot_support_margin_m": landed_margin,
        "old_support_foot_margin_m": old_support_margin,
        "capture_com_world": com.tolist(),
        "capture_com_velocity_world_m_s": com_velocity.tolist(),
        "capture_linear_momentum_world_Ns": linear_momentum.tolist(),
        "capture_linear_momentum_norm_Ns": float(np.linalg.norm(linear_momentum)),
        "capture_centroidal_angular_momentum_world_Nms": angular_and_error[:3].tolist(),
        "capture_centroidal_angular_momentum_norm_Nms": float(np.linalg.norm(angular_and_error[:3])),
        "capture_momentum_consistency_error_Ns": float(angular_and_error[3]),
        "capture_contact_flags_pre": np.asarray(contact.contact_flags, dtype=bool).tolist(),
        "capture_normal_force_pre_N": np.asarray(contact.normal_force_N, dtype=float).tolist(),
        "capture_normal_force_post_N": post_normal.tolist(),
        "capture_friction_utilization_post": post_friction.tolist(),
        "capture_max_friction_utilization_post": float(np.nanmax(post_friction)),
        "capture_torque_utilization": float(arrays["torque_utilization"][index]),
        "capture_qp_slack_norm": float(arrays["qp_slack_norm"][index]),
        "capture_joint_limit_violation": bool(arrays["joint_limit_violation"][index]),
        "capture_min_joint_limit_margin_rad": float(np.min(finite_joint_margin)) if len(finite_joint_margin) else float("nan"),
        "capture_torso_height_m": float(arrays["torso_height_m"][index]),
        "capture_torso_angular_velocity_rad_s": float(arrays["torso_angular_velocity_norm"][index]),
    }


def _run_capture(configs: dict, magnitude: float, direction: float, offset_m: float, duration: float, seed: int) -> dict:
    model = make_model(configs)
    hybrid_config = copy.deepcopy(configs["experiments"].get("hybrid_recovery", {}))
    hybrid_config["max_steps"] = 1
    controller = OracleFootPlacementController(
        model,
        configs["controller"],
        configs["experiments"].get("recovery", {}),
        hybrid_config,
        diagnostic_variant="one_step_current",
        foot_offset_m=offset_m,
    )
    push = make_push(configs, magnitude=magnitude, direction_deg=direction)
    runner = SimulationRunner(
        model,
        controller,
        duration_s=duration,
        control_timestep_s=configs["robot"]["control_timestep"],
        warmup_duration_s=configs["robot"].get("warmup_duration_s", 0.4),
        warmup_reanchor=True,
    )
    run = runner.run(
        push=push,
        recovery_config=recovery_config(configs),
        classify=True,
        seed=seed,
    )
    touchdown_index = _event_index(run, "TOUCHDOWN")
    if touchdown_index is None or controller.touchdown_context is None:
        return {
            "model": model,
            "controller": controller,
            "run": run,
            "push": push,
            "hybrid_config": hybrid_config,
            "offset_m": float(offset_m),
            "target_command_reach_m": controller.first_step_command_reach_m,
            "target_command_world": controller.first_step_target_world.tolist() if controller.first_step_target_world is not None else [],
            "support_center_at_target_world": controller.first_step_support_center_world.tolist() if controller.first_step_support_center_world is not None else [],
            "index": None,
            "event": None,
            "time_s": float("nan"),
            "qpos": None,
            "qvel": None,
            "capture_com": None,
            "context": None,
            "state_sha256": "",
        }
    arrays = run.log.arrays()
    event = next(
        (item for item in _event_list(run) if item.get("label") == "TOUCHDOWN"),
        {},
    )
    qpos = np.asarray(run.qpos_history[touchdown_index], dtype=float).copy()
    qvel = np.asarray(run.qvel_history[touchdown_index], dtype=float).copy()
    capture_com = np.asarray(arrays["com_world"][touchdown_index], dtype=float).copy()
    context = dict(controller.touchdown_context)
    state_sha256 = _state_digest(qpos, qvel, context)
    return {
        "model": model,
        "controller": controller,
        "run": run,
        "push": push,
        "hybrid_config": hybrid_config,
        "offset_m": float(offset_m),
        "target_command_reach_m": controller.first_step_command_reach_m,
        "target_command_world": controller.first_step_target_world.tolist() if controller.first_step_target_world is not None else [],
        "support_center_at_target_world": controller.first_step_support_center_world.tolist() if controller.first_step_support_center_world is not None else [],
        "index": touchdown_index,
        "event": event,
        "time_s": float(arrays["time_s"][touchdown_index]),
        "qpos": qpos,
        "qvel": qvel,
        "capture_com": capture_com,
        "context": context,
        "state_sha256": state_sha256,
    }


def _capture_row(capture: dict, magnitude: float, direction: float, trial_id: str, seed: int) -> dict:
    run = capture["run"]
    arrays = run.log.arrays()
    events = _event_list(run)
    errors = _target_error_series(run)
    swing_index = _event_index(run, "SWING_COMPLETE")
    target_values = np.asarray(arrays["planned_foot_target_world"], dtype=float)
    finite_target = target_values[np.all(np.isfinite(target_values), axis=1)]
    target = finite_target[0] if len(finite_target) else np.full(3, np.nan)
    row = {
        "trial_id": trial_id,
        "push_magnitude_N": float(magnitude),
        "push_direction_deg": float(direction),
        "oracle_foot_offset_m": float(capture["offset_m"]),
        "seed": int(seed),
        "swing_complete": bool(swing_index is not None),
        "swing_complete_time_s": _event_time(run, "SWING_COMPLETE") or "",
        "touchdown_confirmed": bool(capture["index"] is not None),
        "touchdown_time_s": capture["time_s"] if capture["index"] is not None else "",
        "recovery_classifier_success": bool(run.recovery.success) if run.recovery else False,
        "recovery_classifier_failure_reason": run.recovery.failure_reason if run.recovery else "NOT_CLASSIFIED",
        "target_x_m": float(target[0]),
        "target_y_m": float(target[1]),
        "target_error_at_swing_complete_m": float(errors[swing_index]) if swing_index is not None else float("nan"),
        "max_target_error_m": float(np.nanmax(errors)) if np.any(np.isfinite(errors)) else float("nan"),
        "event_count": len(events),
        "events_json": json.dumps(events, sort_keys=True),
    }
    command_reach = float(capture.get("target_command_reach_m", float("nan")))
    row["command_reach_m"] = command_reach
    row["command_within_max_reach"] = bool(
        np.isfinite(command_reach)
        and command_reach <= float(capture["hybrid_config"].get("max_step_reach_m", np.nan)) + 1e-9,
    )
    if capture["index"] is not None:
        row.update(_capture_geometry_metrics(capture["model"], capture))
    else:
        row.update({
            "capture_state_sha256": "",
            "capture_time_s": "",
            "touchdown_confirmed": False,
            "swing_complete_confirmed": bool(swing_index is not None),
            "swing_complete_target_error_m": float(errors[swing_index]) if swing_index is not None else float("nan"),
            "reference_foot_error_within_landing_tolerance": bool(
                swing_index is not None
                and np.isfinite(errors[swing_index])
                and errors[swing_index] <= float(capture["hybrid_config"].get("landing_position_tolerance_m", 0.08)),
            ),
            "touchdown_target_error_m": float("nan"),
            "geometric_landing_reachable": "unknown_without_touchdown",
            "landed_foot": "",
            "support_foot": "",
            "landed_foot_support_margin_m": float("nan"),
            "old_support_foot_margin_m": float("nan"),
            "capture_com_world": "",
            "capture_com_velocity_world_m_s": "",
            "capture_linear_momentum_world_Ns": "",
            "capture_linear_momentum_norm_Ns": float("nan"),
            "capture_centroidal_angular_momentum_world_Nms": "",
            "capture_centroidal_angular_momentum_norm_Nms": float("nan"),
            "capture_momentum_consistency_error_Ns": float("nan"),
            "capture_contact_flags_pre": "",
            "capture_normal_force_pre_N": "",
            "capture_normal_force_post_N": "",
            "capture_friction_utilization_post": "",
            "capture_max_friction_utilization_post": float("nan"),
            "capture_torque_utilization": float("nan"),
            "capture_qp_slack_norm": float("nan"),
            "capture_joint_limit_violation": False,
            "capture_min_joint_limit_margin_rad": float("nan"),
            "capture_torso_height_m": float("nan"),
            "capture_torso_angular_velocity_rad_s": float("nan"),
        })
    return row


def _strict_replay_summary(summary: dict, run, observables: dict[str, np.ndarray], capture: dict) -> dict:
    arrays = run.log.arrays()
    time_s = np.asarray(arrays["time_s"], dtype=float)
    stable_duration = float(capture["hybrid_config"].get("stable_duration_s", 0.25))
    final_window = time_s >= (time_s[-1] - stable_duration) if len(time_s) else np.zeros(0, dtype=bool)
    joint_limit = np.asarray(arrays["joint_limit_violation"], dtype=bool)
    support_margin = np.asarray(observables["support_margin_m"], dtype=float)
    strict_joint_window = bool(np.any(final_window) and not np.any(joint_limit[final_window]))
    strict_support_window = bool(
        np.any(final_window)
        and np.all(np.isfinite(support_margin[final_window]))
        and np.all(support_margin[final_window] >= -0.005),
    )
    strict = bool(
        summary["custom_stable_final_window"]
        and strict_joint_window
        and strict_support_window
    )
    return {
        **summary,
        "joint_limit_violation_any": bool(np.any(joint_limit)),
        "joint_limit_violation_final_window": bool(np.any(joint_limit[final_window])) if np.any(final_window) else False,
        "strict_joint_limit_free_final_window": strict_joint_window,
        "strict_support_consistent_final_window": strict_support_window,
        "strict_dynamically_stabilizable": strict,
        "strict_criterion": "existing custom stable final window + no joint-limit violation + active-support margin >= -0.005 m over final 0.25 s",
    }


def _replay_row(
    configs: dict,
    capture: dict,
    capture_id: str,
    variant: str,
    qvel_scale: float,
    output_root: Path,
    data_root: Path,
    figures_root: Path,
    source: dict,
    command: str,
    seed: int,
    duration: float,
) -> dict:
    replay_capture = dict(capture)
    replay_capture["qvel"] = np.asarray(capture["qvel"], dtype=float) * float(qvel_scale)
    replay_capture["replay_qvel_scale"] = float(qvel_scale)
    model, run, active_contacts, initial_qvel = _replay(
        configs,
        replay_capture,
        variant,
        duration,
        seed,
    )
    observables = _replay_observables(model, run, replay_capture, active_contacts)
    summary = _strict_replay_summary(
        _stability_summary(run, observables, replay_capture, active_contacts, variant, configs),
        run,
        observables,
        replay_capture,
    )
    replay_name = f"{capture_id}_{variant}_qvel{qvel_scale:g}"
    observable_path = data_root / f"{_safe_name(replay_name)}_observables.npz"
    np.savez_compressed(
        observable_path,
        **observables,
        replay_time_s=np.asarray(run.log.arrays()["time_s"], dtype=float),
    )
    metadata = {
        **source,
        "run_id": output_root.name,
        "trial_id": replay_name,
        "experiment": "map_touchdown_viability",
        "phase": "exact_touchdown_replay",
        "replay_variant": variant,
        "qvel_scale": float(qvel_scale),
        "config": configs,
        "active_contacts": list(active_contacts),
        "external_push_removed": True,
        "planning_removed": True,
        "swing_generation_removed": True,
        "landing_gate_removed": True,
        "capture_trial_id": capture_id,
        "capture_state_sha256": capture["state_sha256"],
        "capture_state_path": str((Path("data") / "touchdown_states" / f"{_safe_name(capture_id)}.npz").as_posix()),
        "initial_state": "exact_touchdown_qpos_qvel_scaled" if qvel_scale != 0.0 else "exact_touchdown_qpos_with_qvel_zero_counterfactual",
        "initial_generalized_velocity_norm": float(np.linalg.norm(initial_qvel)),
        "observable_path": str(observable_path.relative_to(output_root)),
        "summary": summary,
        "command": command,
    }
    trial_path = data_root / f"{_safe_name(replay_name)}.npz"
    save_run(run, trial_path, metadata)
    _plot_replay(
        np.asarray(run.log.arrays()["time_s"], dtype=float),
        run.log.arrays(),
        observables,
        summary,
        figures_root / f"{_safe_name(replay_name)}.png",
    )
    return {
        "capture_trial_id": capture_id,
        "capture_state_sha256": capture["state_sha256"],
        "replay_trial_id": replay_name,
        "replay_variant": variant,
        "qvel_scale": float(qvel_scale),
        "initial_generalized_velocity_norm": float(np.linalg.norm(initial_qvel)),
        **summary,
    }


def _plot_map(capture_rows: list[dict], replay_rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not capture_rows:
        return
    magnitudes = sorted({float(row["push_magnitude_N"]) for row in capture_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for magnitude in magnitudes:
        selected = [row for row in capture_rows if float(row["push_magnitude_N"]) == magnitude]
        xs = [float(row["oracle_foot_offset_m"]) for row in selected]
        contact = [float(row["touchdown_confirmed"]) for row in selected]
        geom = [
            float(row["geometric_landing_reachable"])
            if row["geometric_landing_reachable"] in {True, "True", "true"}
            else np.nan
            for row in selected
        ]
        axes[0].plot(xs, geom, "o-", label=f"{magnitude:g} N: landing target reached")
        axes[0].plot(xs, contact, "x--", label=f"{magnitude:g} N: touchdown")
    axes[0].set_title("Reachability through MuJoCo pipeline")
    axes[0].set_xlabel("oracle foot offset along push [m]")
    axes[0].set_ylabel("boolean indicator")
    axes[0].set_yticks([0, 1], ["no", "yes"])
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=10)
    variants = sorted({str(row["replay_variant"]) for row in replay_rows})
    for variant in variants:
        selected = [row for row in replay_rows if row["replay_variant"] == variant and float(row["qvel_scale"]) in (0.5, 1.0)]
        for magnitude in magnitudes:
            points = []
            for replay in selected:
                capture = next(
                    (row for row in capture_rows if row["trial_id"] == replay["capture_trial_id"]),
                    None,
                )
                if capture is not None and float(capture["push_magnitude_N"]) == magnitude:
                    points.append((float(capture["oracle_foot_offset_m"]), float(replay["strict_dynamically_stabilizable"])))
            if points:
                points.sort()
                label = f"{magnitude:g} N {variant}"
                axes[1].plot([p[0] for p in points], [p[1] for p in points], "o-", label=label)
    axes[1].set_title("Strict post-touchdown stabilizability")
    axes[1].set_xlabel("oracle foot offset along push [m]")
    axes[1].set_yticks([0, 1], ["no", "yes"])
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[70.0, 75.0])
    parser.add_argument("--direction", type=float, default=0.0)
    parser.add_argument("--offsets", type=float, nargs="+", default=list(DEFAULT_OFFSETS_M))
    parser.add_argument("--capture-duration", type=float, default=3.20)
    parser.add_argument("--replay-duration", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = load_configs(ROOT)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output root: {output_root}")
    data_root = output_root / "data" / "trials"
    state_root = output_root / "data" / "touchdown_states"
    figures_root = output_root / "figures"
    logs_root = output_root / "logs"
    for path in (data_root, state_root, figures_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    source = _source_metadata()
    write_execution_manifest(
        logs_root / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "map_touchdown_viability",
            "question": "Does a meaningful touchdown state region exist that is both reachable through the MuJoCo swing/contact pipeline and stabilizable by the existing SE(3) WBC?",
            "run_id": output_root.name,
            "model_path": str(resolve_model_path(configs)),
            "magnitudes_N": [float(value) for value in args.magnitudes],
            "direction_deg": float(args.direction),
            "oracle_offsets_m": [float(value) for value in args.offsets],
            "offset_definition": "signed displacement of the first swing-foot target along the normalized measured push direction after the existing target heuristic and reach clamp",
            "command_reach_definition": "target distance from the measured support-foot center must be <= hybrid_config.max_step_reach_m",
            "reference_completion_definition": "the measured swing foot error at the existing SWING_COMPLETE event; this is reported separately because SWING_COMPLETE is emitted when the reference trajectory finishes",
            "geometric_landing_reach_definition": "for a confirmed touchdown, the measured swing foot is within landing_position_tolerance_m of its target at the TOUCHDOWN row; without touchdown the geometric landing result is unknown, not a proven failure",
            "contact_reach_definition": "the existing landing gate emits a sustained-load TOUCHDOWN event",
            "replay_plan": [
                {"variant": variant, "qvel_scale": float(scale)}
                for variant, scale in REPLAY_PLAN
            ],
            "strict_dynamic_stability_definition": "existing custom stable final window plus no joint-limit violation and active-support margin >= -0.005 m for the final configured stable duration",
            "counterfactual_warning": "qvel_scale=0.5 and landed_support_zero_momentum are counterfactual ablations, not physically reachable recovery claims",
            "production_controller_changed": False,
            "source_provenance": source,
            "source_tree_hash_algorithm": source.get("source_tree_hash_algorithm", "unknown"),
            "command": command,
        },
    )

    capture_rows: list[dict] = []
    replay_rows: list[dict] = []
    for magnitude in args.magnitudes:
        for offset in args.offsets:
            capture_id = f"capture_{magnitude:g}N_{args.direction:g}deg_offset{offset:+.3f}_seed{args.seed}"
            capture = _run_capture(
                configs,
                float(magnitude),
                float(args.direction),
                float(offset),
                args.capture_duration,
                args.seed,
            )
            capture_path = data_root / f"{_safe_name(capture_id)}.npz"
            capture_metadata = {
                **source,
                "run_id": output_root.name,
                "trial_id": capture_id,
                "experiment": "map_touchdown_viability",
                "phase": "oracle_foot_target_capture",
                "config": configs,
                "hybrid_config": capture["hybrid_config"],
                "oracle_foot_offset_m": float(offset),
                "push": capture["push"].__dict__,
                "capture_state_sha256": capture["state_sha256"],
                "capture_event": capture["event"],
                "events": _event_list(capture["run"]),
                "command": command,
            }
            save_run(capture["run"], capture_path, capture_metadata)
            row = _capture_row(capture, float(magnitude), float(args.direction), capture_id, args.seed)
            row["artifact_path"] = str(capture_path.relative_to(output_root))
            capture_rows.append(row)
            print(json.dumps({
                "phase": "capture",
                "trial_id": capture_id,
                "oracle_foot_offset_m": float(offset),
                "touchdown_confirmed": bool(capture["index"] is not None),
                "state_sha256": capture["state_sha256"],
            }))
            if capture["index"] is None:
                continue
            state_path = state_root / f"{_safe_name(capture_id)}.npz"
            _write_touchdown_snapshot(state_path, capture)
            for variant, qvel_scale in REPLAY_PLAN:
                replay_rows.append(
                    _replay_row(
                        configs,
                        capture,
                        capture_id,
                        variant,
                        qvel_scale,
                        output_root,
                        data_root,
                        figures_root,
                        source,
                        command,
                        args.seed,
                        args.replay_duration,
                    )
                )

    write_csv(capture_rows, output_root / "capture_summary.csv")
    write_csv(replay_rows, output_root / "replay_summary.csv")
    _plot_map(capture_rows, replay_rows, figures_root / "touchdown_viability_map.png")

    replay_by_capture = {}
    for row in replay_rows:
        replay_by_capture.setdefault(row["capture_trial_id"], []).append(row)
    viability_rows = []
    for row in capture_rows:
        related = replay_by_capture.get(row["trial_id"], [])
        strict_exact = [
            value for value in related
            if float(value["qvel_scale"]) == 1.0
            and value["replay_variant"] in {"double_support_current", "landed_support_momentum_capture"}
        ]
        strict_zero = [
            value for value in related
            if value["replay_variant"] == "landed_support_zero_momentum"
        ]
        viability_rows.append({
            "trial_id": row["trial_id"],
            "push_magnitude_N": row["push_magnitude_N"],
            "oracle_foot_offset_m": row["oracle_foot_offset_m"],
            "command_within_max_reach": row["command_within_max_reach"],
            "geometric_landing_reachable": row["geometric_landing_reachable"],
            "touchdown_confirmed": row["touchdown_confirmed"],
            "strict_exact_replay_success_any": any(bool(value["strict_dynamically_stabilizable"]) for value in strict_exact),
            "strict_zero_momentum_counterfactual_success": any(bool(value["strict_dynamically_stabilizable"]) for value in strict_zero),
            "capture_state_sha256": row["capture_state_sha256"],
        })
    write_csv(viability_rows, output_root / "viability_summary.csv")

    summary = {
        "run_id": output_root.name,
        "capture_count": len(capture_rows),
        "confirmed_touchdown_count": int(sum(bool(row["touchdown_confirmed"]) for row in capture_rows)),
        "geometric_landing_reached_count": int(sum(row["geometric_landing_reachable"] is True for row in capture_rows)),
        "replay_count": len(replay_rows),
        "strict_dynamic_success_count": int(sum(bool(row["strict_dynamically_stabilizable"]) for row in replay_rows)),
        "strict_dynamic_success_exact_qvel_count": int(sum(
            bool(row["strict_dynamically_stabilizable"])
            for row in replay_rows
            if float(row["qvel_scale"]) == 1.0
        )),
        "classifier_success_count": int(sum(bool(row["success"]) for row in replay_rows)),
        "counterfactual_zero_momentum_success_count": int(sum(
            bool(row["strict_dynamically_stabilizable"])
            for row in replay_rows
            if row["replay_variant"] == "landed_support_zero_momentum"
        )),
        "offsets_m": [float(value) for value in args.offsets],
        "magnitudes_N": [float(value) for value in args.magnitudes],
        "artifact_note": "All capture runs, failed cases, exact touchdown states, replay observables, plots, CSV summaries, and provenance metadata are retained under this new output root.",
        "source_provenance": source,
    }
    (logs_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
