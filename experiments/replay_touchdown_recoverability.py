"""Replay confirmed touchdown states to diagnose post-touchdown recoverability.

This study deliberately removes the external push, stepping planner, swing
trajectory, and landing gate from the replay problem.  It starts a fresh
MuJoCo model from the exact ``qpos``/``qvel`` row logged when the current
one-step diagnostic emitted ``TOUCHDOWN`` and compares only contact/control
assumptions that are explicit in the existing WBC:

* the production post-touchdown double-support interpretation;
* double support while holding the measured touchdown CoM;
* physically consistent landed-foot-only support while holding that CoM;
* landed-foot support with the already-configured transfer CoM gains as a
  diagnostic momentum-capture upper bound; and
* the same landed-foot replay with ``qvel=0`` as a non-physical static/
  momentum counterfactual.

The last two variants are diagnostics, not recovery claims.  No variant is
called recovered unless the normal project recovery classifier and the replay
stability checks support that conclusion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from se3_whole_body_control.config import resolve_model_path  # noqa: E402
from se3_whole_body_control.control.support import convex_hull_2d, signed_support_margin  # noqa: E402
from se3_whole_body_control.control.tasks import com_jacobian  # noqa: E402
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController  # noqa: E402
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner  # noqa: E402


REPLAY_VARIANTS = (
    "double_support_current",
    "double_support_hold_com",
    "landed_support_hold_com",
    "landed_support_momentum_capture",
    "landed_support_zero_momentum",
)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _state_digest(qpos: np.ndarray, qvel: np.ndarray, context: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(qpos, dtype="<f8").tobytes())
    digest.update(np.asarray(qvel, dtype="<f8").tobytes())
    digest.update(_json(context).encode("utf-8"))
    return digest.hexdigest()


def _source_metadata() -> dict:
    return {
        "source_commit": os.environ.get("SE3_SOURCE_VERSION", "unknown"),
        "source_tree_sha256": os.environ.get("SE3_SOURCE_TREE_SHA256", "unknown"),
        "remote_source_root": os.environ.get("SE3_SOURCE_ROOT", "unknown"),
        "execution_environment_id": os.environ.get("SE3_EXECUTION_ENV", "unknown"),
    }


def _first_touchdown(run) -> tuple[int, dict]:
    labels = run.log.arrays()["event_label"].astype(str)
    indices = np.flatnonzero(labels == "TOUCHDOWN")
    if len(indices) == 0:
        raise RuntimeError("capture run did not emit a confirmed TOUCHDOWN event")
    index = int(indices[0])
    events = run.metadata.get("controller_summary", {}).get("events", [])
    event = next((item for item in events if item.get("label") == "TOUCHDOWN"), {})
    return index, dict(event)


def _capture_controller_context(controller, event: dict, capture_com: np.ndarray) -> dict:
    step_event = controller.step_event
    return {
        "q_des": np.asarray(controller.q_des, dtype=float).tolist(),
        "T_des_torso": np.asarray(controller.T_des_torso, dtype=float).tolist(),
        "T_des_pelvis": np.asarray(controller.T_des_pelvis, dtype=float).tolist(),
        "com_des": np.asarray(controller.com_des, dtype=float).tolist(),
        "capture_com": np.asarray(capture_com, dtype=float).tolist(),
        "active_contacts_at_capture": list(controller.contact_names),
        "mode_at_capture": str(controller.mode),
        "step_phase_at_capture": str(controller.step_phase),
        "step_count_at_capture": int(controller.step_count),
        "swing_foot": str(event.get("swing_foot") or (step_event.swing_foot if step_event else "")),
        "support_foot": str(event.get("support_foot") or (step_event.support_foot if step_event else "")),
        "step_index": int(event.get("step_index", step_event.step_index if step_event else 0)),
        "event": event,
    }


def _run_capture(configs: dict, magnitude: float, direction: float, duration: float, seed: int):
    model = make_model(configs)
    hybrid_config = copy.deepcopy(configs["experiments"].get("hybrid_recovery", {}))
    hybrid_config["max_steps"] = 1
    controller = DiagnosticHybridController(
        model,
        configs["controller"],
        configs["experiments"].get("recovery", {}),
        hybrid_config,
        diagnostic_variant="one_step_current",
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
    index, event = _first_touchdown(run)
    arrays = run.log.arrays()
    qpos = np.asarray(run.qpos_history[index], dtype=float).copy()
    qvel = np.asarray(run.qvel_history[index], dtype=float).copy()
    capture_com = np.asarray(arrays["com_world"][index], dtype=float).copy()
    context = _capture_controller_context(controller, event, capture_com)
    state_sha256 = _state_digest(qpos, qvel, context)
    capture = {
        "model": model,
        "controller": controller,
        "run": run,
        "push": push,
        "hybrid_config": hybrid_config,
        "index": index,
        "event": event,
        "time_s": float(arrays["time_s"][index]),
        "qpos": qpos,
        "qvel": qvel,
        "capture_com": capture_com,
        "context": context,
        "state_sha256": state_sha256,
    }
    return capture


def _write_touchdown_snapshot(path: Path, capture: dict) -> None:
    arrays = capture["run"].log.arrays()
    index = int(capture["index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=np.asarray(capture["qpos"], dtype=float),
        qvel=np.asarray(capture["qvel"], dtype=float),
        capture_time_s=np.asarray(capture["time_s"], dtype=float),
        com_world=np.asarray(capture["capture_com"], dtype=float),
        actual_contact_wrench=np.asarray(arrays["actual_contact_wrench"][index], dtype=float),
        actual_contact_wrench_post_step=np.asarray(arrays["actual_contact_wrench_post_step"][index], dtype=float),
        actual_normal_force_post_step_N=np.asarray(arrays["actual_normal_force_post_step_N"][index], dtype=float),
        actual_friction_utilization_post_step=np.asarray(arrays["actual_friction_utilization_post_step"][index], dtype=float),
        torque_utilization=np.asarray(arrays["torque_utilization"][index], dtype=float),
        qp_slack_norm=np.asarray(arrays["qp_slack_norm"][index], dtype=float),
        context_json=np.asarray(_json(capture["context"])),
        state_sha256=np.asarray(capture["state_sha256"]),
    )


def _make_replay_controller(model, configs: dict, context: dict, variant: str):
    controller = WholeBodyQPController(model, configs["controller"], configs["experiments"].get("recovery", {}))
    controller.q_des = np.asarray(context["q_des"], dtype=float).copy()
    controller.pd_fallback.q_des = controller.q_des.copy()
    controller.T_des_torso = np.asarray(context["T_des_torso"], dtype=float).copy()
    controller.T_des_pelvis = np.asarray(context["T_des_pelvis"], dtype=float).copy()
    capture_com = np.asarray(context["capture_com"], dtype=float)
    production_com = np.asarray(context["com_des"], dtype=float)
    landed_foot = str(context["swing_foot"])
    if landed_foot not in {"left_foot", "right_foot"}:
        raise RuntimeError(f"invalid landed foot in capture context: {landed_foot!r}")

    if variant == "double_support_current":
        active_contacts = ("left_foot", "right_foot")
        controller.com_des = production_com.copy()
    elif variant == "double_support_hold_com":
        active_contacts = ("left_foot", "right_foot")
        controller.com_des = capture_com.copy()
    else:
        active_contacts = (landed_foot,)
        controller.com_des = capture_com.copy()
    controller.set_active_contacts(active_contacts)
    controller.set_swing_target(None)
    # These attributes are consumed only by the existing recovery classifier;
    # the replay controller itself has no hybrid state machine.
    controller.allows_single_support = len(active_contacts) == 1
    controller.requires_final_double_support = len(active_contacts) == 2

    if variant == "landed_support_momentum_capture":
        hybrid_config = configs["experiments"].get("hybrid_recovery", {})
        controller.com_task_weight_override = float(hybrid_config["transfer_com_weight"])
        controller.com_task_kp_override = float(hybrid_config["transfer_com_kp"])
        controller.com_task_kd_override = float(hybrid_config["transfer_com_kd"])
    return controller, active_contacts


def _replay(configs: dict, capture: dict, variant: str, duration: float, seed: int):
    model = make_model(configs)
    controller, active_contacts = _make_replay_controller(model, configs, capture["context"], variant)
    initial_qvel = np.asarray(capture["qvel"], dtype=float).copy()
    if variant == "landed_support_zero_momentum":
        initial_qvel[:] = 0.0
    runner = SimulationRunner(
        model,
        controller,
        duration_s=duration,
        control_timestep_s=configs["robot"]["control_timestep"],
        warmup_duration_s=0.0,
        warmup_reanchor=False,
    )
    context = capture["context"]
    run = runner.run(
        push=None,
        recovery_config=recovery_config(configs),
        classify=True,
        seed=seed,
        initial_qpos=np.asarray(capture["qpos"], dtype=float),
        initial_qvel=initial_qvel,
        desired_torso=np.asarray(context["T_des_torso"], dtype=float),
        desired_pelvis=np.asarray(context["T_des_pelvis"], dtype=float),
        com_reference=np.asarray(context["capture_com"], dtype=float),
    )
    return model, run, active_contacts, initial_qvel


def _finite_max(values: np.ndarray, default: float = float("nan")) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else default


def _finite_min(values: np.ndarray, default: float = float("nan")) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else default


def _longest_true_duration(time_s: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not np.any(mask):
        return 0.0
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return float(max((time_s[end] - time_s[start] for start, end in zip(starts, ends)), default=0.0))


def _support_margin_series(arrays: dict[str, np.ndarray], active_contacts: tuple[str, ...]) -> np.ndarray:
    com = np.asarray(arrays["com_world"], dtype=float)
    vertices = np.asarray(arrays["foot_support_vertices_world"], dtype=float).reshape(-1, 2, 4, 2)
    indices = [0 if name == "left_foot" else 1 for name in active_contacts]
    margins = np.full(len(com), np.nan, dtype=float)
    for index in range(len(com)):
        selected = vertices[index, indices].reshape(-1, 2)
        hull = convex_hull_2d(selected)
        margins[index] = signed_support_margin(com[index, :2], hull)
    return margins


def _replay_observables(model, run, capture: dict, active_contacts: tuple[str, ...]) -> dict[str, np.ndarray]:
    import mujoco

    qpos = np.asarray(run.qpos_history, dtype=float)
    qvel = np.asarray(run.qvel_history, dtype=float)
    if len(qpos) != len(qvel):
        raise RuntimeError("qpos_history and qvel_history are not row aligned")
    n = len(qpos)
    com_velocity = np.full((n, 3), np.nan, dtype=float)
    linear_momentum = np.full((n, 3), np.nan, dtype=float)
    linear_momentum_from_com_velocity = np.full((n, 3), np.nan, dtype=float)
    linear_momentum_consistency_error = np.full(n, np.nan, dtype=float)
    centroidal_angular_momentum = np.full((n, 3), np.nan, dtype=float)
    torso_velocity = np.full((n, 6), np.nan, dtype=float)
    root_body_id = int(model.body_ids["floating_base"])
    total_mass = float(np.sum(model.model.body_mass))
    for index, (state_qpos, state_qvel) in enumerate(zip(qpos, qvel)):
        model.reset(qpos=state_qpos, qvel=state_qvel)
        com_velocity[index] = com_jacobian(model) @ state_qvel
        # MuJoCo exposes ``subtree_linvel``/``subtree_angmom`` in the Python
        # bindings, but for this model they remain zero even after
        # mj_comVel.  Use the generalized momentum produced by the public
        # mass-matrix multiply instead.  For a free joint, its first three
        # components are total linear momentum and the next three are
        # angular momentum about the floating-base origin.  Translating that
        # moment to the system CoM gives a physically meaningful centroidal
        # angular momentum.  The linear part is checked against Jcom*qvel.
        generalized_momentum = np.zeros(model.nv, dtype=float)
        mujoco.mj_mulM(model.model, model.data, generalized_momentum, state_qvel)
        linear_momentum[index] = generalized_momentum[:3]
        linear_momentum_from_com_velocity[index] = total_mass * com_velocity[index]
        linear_momentum_consistency_error[index] = float(
            np.linalg.norm(linear_momentum[index] - linear_momentum_from_com_velocity[index]),
        )
        base_position = np.asarray(model.data.xpos[root_body_id], dtype=float)
        centroidal_angular_momentum[index] = generalized_momentum[3:6] - np.cross(
            model.center_of_mass() - base_position,
            generalized_momentum[:3],
        )
        torso_velocity[index] = model.body_velocity("torso")
    arrays = run.log.arrays()
    support_margin = _support_margin_series(arrays, active_contacts)
    return {
        "com_velocity_world": com_velocity,
        "linear_momentum_world": linear_momentum,
        "linear_momentum_from_com_velocity_world": linear_momentum_from_com_velocity,
        "linear_momentum_consistency_error_Ns": linear_momentum_consistency_error,
        "centroidal_angular_momentum_world": centroidal_angular_momentum,
        "torso_velocity_world": torso_velocity,
        "support_margin_m": support_margin,
    }


def _stability_summary(run, observables: dict[str, np.ndarray], capture: dict, active_contacts: tuple[str, ...], variant: str, configs: dict) -> dict:
    arrays = run.log.arrays()
    recovery_cfg = recovery_config(configs)
    time_s = np.asarray(arrays["time_s"], dtype=float)
    com = np.asarray(arrays["com_world"], dtype=float)
    com_reference = np.asarray(capture["capture_com"], dtype=float)
    com_displacement = np.linalg.norm(com[:, :2] - com_reference[:2], axis=1)
    com_speed = np.linalg.norm(observables["com_velocity_world"][:, :2], axis=1)
    momentum_norm = np.linalg.norm(observables["linear_momentum_world"], axis=1)
    angular_momentum_norm = np.linalg.norm(observables["centroidal_angular_momentum_world"], axis=1)
    friction = np.asarray(arrays["actual_friction_utilization_post_step"], dtype=float)
    friction_max = np.nanmax(friction, axis=1) if len(friction) else np.zeros(0)
    normal_force = np.asarray(arrays["actual_normal_force_post_step_N"], dtype=float)
    loaded = normal_force >= float(capture["hybrid_config"].get("contact_force_threshold_N", 5.0))
    loaded_required = np.all(loaded[:, [0, 1]], axis=1) if len(active_contacts) == 2 else loaded[:, 0 if active_contacts[0] == "left_foot" else 1]
    stable_mask = (
        (np.asarray(arrays["torso_rotation_error_rad"], dtype=float) <= float(recovery_cfg.orientation_threshold_rad))
        & (np.asarray(arrays["torso_angular_velocity_norm"], dtype=float) <= float(recovery_cfg.angular_velocity_threshold_rad_s))
        & (com_displacement <= float(recovery_cfg.com_displacement_threshold_m))
        & (np.asarray(arrays["torso_height_m"], dtype=float) >= float(recovery_cfg.torso_ground_height_m))
        & np.asarray(arrays["qp_success"], dtype=bool)
        & np.asarray(arrays["numerical_valid"], dtype=bool)
        & (np.asarray(arrays["torque_utilization"], dtype=float) <= 1.05)
        & (friction_max <= float(recovery_cfg.friction_utilization_threshold))
        & loaded_required
    )
    stable_duration = _longest_true_duration(time_s, stable_mask)
    stable_duration_required = float(recovery_cfg.stable_duration_s)
    final_window = time_s >= (time_s[-1] - stable_duration_required) if len(time_s) else np.zeros(0, dtype=bool)
    custom_success = bool(len(time_s) and np.all(stable_mask[final_window]) and np.any(final_window))
    recovery = run.recovery
    post = time_s >= 0.25 if len(time_s) else np.zeros(0, dtype=bool)
    return {
        "variant": variant,
        "success": bool(recovery.success) if recovery is not None else False,
        "failure_reason": recovery.failure_reason if recovery is not None else "NOT_CLASSIFIED",
        "recovered_at_s": recovery.recovered_at_s if recovery is not None and recovery.recovered_at_s is not None else "",
        "custom_stable_final_window": custom_success,
        "custom_longest_stable_duration_s": stable_duration,
        "final_support_margin_m": float(observables["support_margin_m"][-1]) if len(time_s) else float("nan"),
        "min_support_margin_m": _finite_min(observables["support_margin_m"]),
        "final_com_speed_m_s": float(com_speed[-1]) if len(com_speed) else float("nan"),
        "max_com_speed_m_s": _finite_max(com_speed),
        "final_linear_momentum_Ns": float(momentum_norm[-1]) if len(momentum_norm) else float("nan"),
        "max_linear_momentum_Ns": _finite_max(momentum_norm),
        "final_centroidal_angular_momentum_Nms": float(angular_momentum_norm[-1]) if len(angular_momentum_norm) else float("nan"),
        "max_centroidal_angular_momentum_Nms": _finite_max(angular_momentum_norm),
        "final_torso_angular_velocity_rad_s": float(arrays["torso_angular_velocity_norm"][-1]) if len(time_s) else float("nan"),
        "max_torso_angular_velocity_rad_s": _finite_max(arrays["torso_angular_velocity_norm"]),
        "final_com_displacement_m": float(com_displacement[-1]) if len(time_s) else float("nan"),
        "max_com_displacement_m": _finite_max(com_displacement),
        "loaded_contact_fraction": float(np.mean(loaded_required)) if len(loaded_required) else 0.0,
        "loaded_contact_fraction_final_window": float(np.mean(loaded_required[final_window])) if np.any(final_window) else 0.0,
        "max_force_N": _finite_max(np.linalg.norm(np.asarray(arrays["actual_contact_wrench_post_step"], dtype=float)[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3), axis=2)) if len(time_s) else float("nan"),
        "max_friction_utilization": _finite_max(friction_max),
        "max_torque_utilization": _finite_max(arrays["torque_utilization"]),
        "max_qp_slack_norm": _finite_max(arrays["qp_slack_norm"]),
        "max_dynamics_residual_norm": _finite_max(arrays["dynamics_residual_norm"]),
        "max_contact_acceleration_residual_norm": _finite_max(arrays["contact_acceleration_residual_norm"]),
        "max_qp_solve_time_ms": _finite_max(np.asarray(arrays["qp_solve_time_s"], dtype=float) * 1000.0),
        "post_touchdown_com_speed_max_0p25s_m_s": _finite_max(com_speed[post]),
        "post_touchdown_momentum_max_0p25s_Ns": _finite_max(momentum_norm[post]),
    }


def _plot_replay(time_s: np.ndarray, arrays: dict[str, np.ndarray], observables: dict[str, np.ndarray], summary: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    com_speed = np.linalg.norm(observables["com_velocity_world"][:, :2], axis=1)
    linear_momentum = np.linalg.norm(observables["linear_momentum_world"], axis=1)
    angular_momentum = np.linalg.norm(observables["centroidal_angular_momentum_world"], axis=1)
    friction = np.nanmax(np.asarray(arrays["actual_friction_utilization_post_step"], dtype=float), axis=1)
    normal_force = np.asarray(arrays["actual_normal_force_post_step_N"], dtype=float)
    axes[0].plot(time_s, com_speed, label="CoM speed |Jcom qvel|", color="#1769aa")
    axes[0].plot(time_s, linear_momentum, label="linear momentum norm", color="#d95f02")
    axes[0].set_ylabel("speed / momentum")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot(time_s, angular_momentum, label="centroidal angular momentum", color="#7b3294")
    axes[1].plot(time_s, observables["torso_velocity_world"][:, 3:], label="torso angular velocity norm", color="#238b45")
    axes[1].set_ylabel("angular state")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].plot(time_s, normal_force[:, 0], label="left normal force", color="#1b9e77")
    axes[2].plot(time_s, normal_force[:, 1], label="right normal force", color="#d95f02")
    axes[2].set_ylabel("GRF z [N]")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[3].plot(time_s, arrays["qp_slack_norm"], label="QP slack", color="#d95f02")
    axes[3].plot(time_s, arrays["torque_utilization"], label="torque utilization", color="#7b3294")
    axes[3].plot(time_s, friction, label="friction utilization", color="#1b9e77")
    axes[3].set_ylabel("constraint usage")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[4].plot(time_s, observables["support_margin_m"], label="active support margin", color="#1769aa")
    axes[4].axhline(0.0, color="#444", linewidth=0.8)
    axes[4].set_ylabel("support [m]")
    axes[4].set_xlabel("replay time [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].set_title(f"{summary['variant']} | classifier={summary['success']} | custom={summary['custom_stable_final_window']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[70.0, 75.0])
    parser.add_argument("--direction", type=float, default=0.0)
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
            "experiment": "replay_touchdown_recoverability",
            "question": "Can confirmed touchdown states be stabilized after removing planning, swing generation, and external push?",
            "run_id": output_root.name,
            "model_path": str(resolve_model_path(configs)),
            "magnitudes_N": [float(value) for value in args.magnitudes],
            "direction_deg": float(args.direction),
            "capture_duration_s": float(args.capture_duration),
            "replay_duration_s": float(args.replay_duration),
            "replay_variants": list(REPLAY_VARIANTS),
            "capture_event_semantics": "qpos/qvel are the pre-integration state at the control row whose controller diagnostics emitted TOUCHDOWN after the configured sustained-load gate.",
            "external_push_in_replay": False,
            "source_provenance": source,
            "counterfactual_definitions": {
                "landed_support_momentum_capture": "landed-foot-only WBC with the existing configured transfer CoM weight/Kp/Kd, not tuned for this study",
                "landed_support_zero_momentum": "same touchdown qpos with qvel set to zero; non-physical upper bound used only to isolate the effect of incoming momentum",
            },
            "interpretation_rule": "A successful replay is diagnostic evidence only; it is not an adaptive stepping claim because no planner, swing, push, or landing transition is active in replay.",
            "command": command,
        },
    )

    rows: list[dict] = []
    captures: list[dict] = []
    for magnitude in args.magnitudes:
        capture = _run_capture(configs, float(magnitude), float(args.direction), args.capture_duration, args.seed)
        capture_id = f"touchdown_{magnitude:g}N_{args.direction:g}deg_seed{args.seed}"
        state_path = state_root / f"{_safe_name(capture_id)}.npz"
        _write_touchdown_snapshot(state_path, capture)
        arrays = capture["run"].log.arrays()
        capture_metadata = {
            **source,
            "run_id": output_root.name,
            "trial_id": capture_id,
            "experiment": "replay_touchdown_recoverability",
            "phase": "capture",
            "diagnostic_variant": "one_step_current_max_steps_1",
            "config": configs,
            "hybrid_config": capture["hybrid_config"],
            "push": capture["push"].__dict__,
            "capture_index": int(capture["index"]),
            "capture_time_s": float(capture["time_s"]),
            "capture_event": capture["event"],
            "capture_state_path": str(state_path.relative_to(output_root)),
            "capture_state_sha256": capture["state_sha256"],
            "controller_context": capture["context"],
            "capture_contact_pre": {
                "flags": np.asarray([arrays["contact_left"][capture["index"]], arrays["contact_right"][capture["index"]]], dtype=bool).tolist(),
                "wrench_world": np.asarray(arrays["actual_contact_wrench"][capture["index"]], dtype=float).tolist(),
            },
            "capture_contact_post_step": {
                "flags": np.asarray([arrays["contact_left_post_step"][capture["index"]], arrays["contact_right_post_step"][capture["index"]]], dtype=bool).tolist(),
                "normal_force_N": np.asarray(arrays["actual_normal_force_post_step_N"][capture["index"]], dtype=float).tolist(),
                "wrench_world": np.asarray(arrays["actual_contact_wrench_post_step"][capture["index"]], dtype=float).tolist(),
            },
            "command": command,
        }
        save_run(capture["run"], data_root / f"{_safe_name(capture_id)}_capture.npz", capture_metadata)
        _plot_replay(
            np.asarray(arrays["time_s"], dtype=float),
            arrays,
            _replay_observables(capture["model"], capture["run"], capture, ("left_foot", "right_foot")),
            {"variant": "capture_one_step_current", "success": False, "custom_stable_final_window": False},
            figures_root / f"{_safe_name(capture_id)}_capture.png",
        )
        captures.append(capture)
        print(json.dumps({"phase": "capture", "trial_id": capture_id, "touchdown_time_s": capture["time_s"], "state_sha256": capture["state_sha256"]}))

        for variant in REPLAY_VARIANTS:
            model, run, active_contacts, initial_qvel = _replay(
                configs, capture, variant, args.replay_duration, args.seed,
            )
            observables = _replay_observables(model, run, capture, active_contacts)
            summary = _stability_summary(run, observables, capture, active_contacts, variant, configs)
            trial_id = f"{capture_id}_{variant}"
            observable_path = data_root / f"{_safe_name(trial_id)}_observables.npz"
            np.savez_compressed(
                observable_path,
                **observables,
                replay_time_s=np.asarray(run.log.arrays()["time_s"], dtype=float),
            )
            metadata = {
                **source,
                "run_id": output_root.name,
                "trial_id": trial_id,
                "experiment": "replay_touchdown_recoverability",
                "phase": "replay",
                "replay_variant": variant,
                "config": configs,
                "active_contacts": list(active_contacts),
                "external_push_removed": True,
                "planning_removed": True,
                "swing_generation_removed": True,
                "landing_gate_removed": True,
                "capture_trial_id": capture_id,
                "capture_state_sha256": capture["state_sha256"],
                "capture_state_path": str(state_path.relative_to(output_root)),
                "initial_state": "exact_touchdown_qpos_qvel" if variant != "landed_support_zero_momentum" else "exact_touchdown_qpos_with_qvel_zero_counterfactual",
                "initial_qvel_norm_rad_s": float(np.linalg.norm(initial_qvel)),
                "replay_target_policy": "captured_com_des" if variant == "double_support_current" else "hold_capture_com",
                "momentum_capture_policy": "existing_transfer_com_gains" if variant == "landed_support_momentum_capture" else "none",
                "observable_path": str(observable_path.relative_to(output_root)),
                "diagnostic_summary": summary,
                "command": command,
            }
            save_run(run, data_root / f"{_safe_name(trial_id)}.npz", metadata)
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
                "magnitude_N": float(magnitude),
                "direction_deg": float(args.direction),
                "capture_time_s": float(capture["time_s"]),
                "capture_state_sha256": capture["state_sha256"],
                "active_contacts": "+".join(active_contacts),
                "initial_qvel_norm_rad_s": float(np.linalg.norm(initial_qvel)),
                "source_commit": source["source_commit"],
            })
            rows.append(row)
            print(json.dumps({"phase": "replay", "trial_id": trial_id, "success": row["success"], "failure_reason": row["failure_reason"], "custom_stable_final_window": row["custom_stable_final_window"]}))

    write_csv(rows, output_root / "replay_summary.csv")
    summary = {
        "run_id": output_root.name,
        "capture_count": len(captures),
        "replay_count": len(rows),
        "replay_variants": list(REPLAY_VARIANTS),
        "classifier_success_count": int(sum(bool(row["success"]) for row in rows)),
        "custom_success_count": int(sum(bool(row["custom_stable_final_window"]) for row in rows)),
        "artifact_note": "All capture states, qpos/qvel histories, replay observables, JSON metadata, CSV summaries, and figures are retained under this new output root.",
        "source_provenance": source,
    }
    (logs_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
