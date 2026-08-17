"""Mechanism-identification replay study for post-touchdown WBC authority.

This experiment starts from the already projected 75 N touchdown state and
removes push, planning, swing generation, and landing transitions.  It first
replays the current landed-support WBC as a V4 comparability gate, then runs
only the five pre-declared diagnostic profiles.  No production stepping or
production WBC code is changed by this harness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    _json_safe,
    load_configs,
    make_model,
    recovery_config,
    save_run,
    write_csv,
    write_execution_manifest,
)
from replay_touchdown_recoverability import (  # noqa: E402
    _replay_observables,
    _stability_summary,
)
from se3_whole_body_control.config import resolve_model_path  # noqa: E402
from se3_whole_body_control.control.diagnostic_wbc import (  # noqa: E402
    DiagnosticWholeBodyQPController,
    PROFILE_DEFINITIONS,
)
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner  # noqa: E402


PROFILES = (
    "baseline_landed_support",
    "pose_pruned",
    "centroidal_momentum",
    "joint_limit_guarded",
    "combined_minimal",
)
HIGH_UTILIZATION_THRESHOLD = 0.95
MIN_SUSTAINED_SAMPLES = 2
SUPPORT_MARGIN_THRESHOLD_M = -0.005
MOMENTUM_GROWTH_FRACTION = 0.05
QP_SLACK_SPIKE_MIN = 1.0
QP_SLACK_SPIKE_MULTIPLIER = 3.0
BASELINE_SERIES_ATOL = 2.0e-3
BASELINE_SERIES_RTOL = 2.0e-3
BASELINE_SUMMARY_ATOL = 2.0e-2
BASELINE_SUMMARY_RTOL = 2.0e-3


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value))


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata() -> dict:
    source_commit = os.environ.get("SE3_SOURCE_VERSION", "")
    if not source_commit:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            source_commit = "unknown"
    return {
        "source_commit": source_commit,
        "source_tree_sha256": os.environ.get("SE3_SOURCE_TREE_SHA256", "unknown"),
        "remote_source_root": os.environ.get("SE3_SOURCE_ROOT", "unknown"),
        "execution_environment_id": os.environ.get("SE3_EXECUTION_ENV", "unknown"),
    }


def _canonical_config_sha(configs: dict) -> str:
    portable = _json_safe(copy.deepcopy(configs))
    # Match experiments.common.execution_manifest exactly so the diagnostic
    # manifest can be compared with the historical V4 provenance hash.
    return hashlib.sha256(json.dumps(portable, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _raw_config_hashes() -> dict[str, str]:
    paths = {
        "controller": ROOT / "configs" / "controller.yaml",
        "experiments": ROOT / "configs" / "experiments.yaml",
        "robot_catalog": ROOT / "configs" / "robot.yaml",
        "robot_profile": ROOT / "configs" / "robots" / "unitree_g1.yaml",
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.exists()}


def _scalar(data, key: str):
    value = data[key]
    return value.item() if np.asarray(value).shape == () else value


def _load_projected_capture(
    projected_path: Path,
    capture_path: Path,
    configs: dict,
) -> tuple[dict, dict]:
    if not projected_path.exists():
        raise FileNotFoundError(f"projected touchdown state not found: {projected_path}")
    if not capture_path.exists():
        raise FileNotFoundError(f"capture state context not found: {capture_path}")
    with np.load(capture_path, allow_pickle=False) as capture_data:
        capture_qpos = np.asarray(capture_data["qpos"], dtype=float)
        capture_qvel = np.asarray(capture_data["qvel"], dtype=float)
        capture_com = np.asarray(capture_data["com_world"], dtype=float)
        context = json.loads(str(_scalar(capture_data, "context_json")))
        input_state_sha = str(_scalar(capture_data, "state_sha256"))
    with np.load(projected_path, allow_pickle=False) as state_data:
        qpos = np.asarray(state_data["qpos"], dtype=float)
        qvel = np.asarray(state_data["qvel"], dtype=float)
        stage = str(_scalar(state_data, "stage"))
        projected_state_sha = str(_scalar(state_data, "projected_state_sha256"))
        projection_summary = json.loads(str(_scalar(state_data, "projection_summary_json")))

    if stage != "joint_limit_contact_support":
        raise ValueError(f"expected joint_limit_contact_support, got {stage!r}")
    if not bool(projection_summary.get("projection_feasible", False)):
        raise ValueError("projected touchdown state is not marked feasible")
    if str(projection_summary.get("projected_state_sha256")) != projected_state_sha:
        raise ValueError("projected state hash disagrees with projection summary")
    if not np.allclose(qvel, capture_qvel, atol=0.0, rtol=0.0):
        raise ValueError("projected replay qvel is not exactly the captured qvel")
    if capture_qpos.shape != (36,) or qpos.shape != (36,) or qvel.shape != (35,):
        raise ValueError(f"unexpected G1 state shape: qpos={qpos.shape}, qvel={qvel.shape}")
    if context.get("swing_foot") not in {"left_foot", "right_foot"}:
        raise ValueError(f"invalid landed foot in capture context: {context.get('swing_foot')!r}")

    capture = {
        "qpos": qpos.copy(),
        "qvel": qvel.copy(),
        "capture_com": capture_com.copy(),
        "context": context,
        "hybrid_config": copy.deepcopy(configs["experiments"]["hybrid_recovery"]),
        "state_sha256": input_state_sha,
        "projected_state_sha256": projected_state_sha,
        "projection_stage": stage,
        "capture_path": str(capture_path),
        "projected_path": str(projected_path),
        "capture_qpos": capture_qpos,
        "capture_qvel": capture_qvel,
    }
    provenance = {
        "capture_file_sha256": _sha256_file(capture_path),
        "projected_file_sha256": _sha256_file(projected_path),
        "input_touchdown_state_sha256": input_state_sha,
        "projected_state_sha256": projected_state_sha,
        "projection_stage": stage,
        "projection_summary": projection_summary,
        "landed_foot": context["swing_foot"],
        "capture_time_s": context.get("capture_controller_time_s"),
        "qvel_preserved_exactly": True,
        "controller_context_preserved": True,
    }
    return capture, provenance


def _make_replay_controller(model, configs: dict, capture: dict, profile: str):
    controller = DiagnosticWholeBodyQPController(
        model,
        configs["controller"],
        configs["experiments"].get("recovery", {}),
        profile=profile,
        control_timestep_s=float(configs["robot"]["control_timestep"]),
    )
    context = capture["context"]
    controller.q_des = np.asarray(context["q_des"], dtype=float).copy()
    controller.pd_fallback.q_des = controller.q_des.copy()
    controller.T_des_torso = np.asarray(context["T_des_torso"], dtype=float).copy()
    controller.T_des_pelvis = np.asarray(context["T_des_pelvis"], dtype=float).copy()
    controller.com_des = np.asarray(capture["capture_com"], dtype=float).copy()
    landed_foot = str(context["swing_foot"])
    controller.set_active_contacts((landed_foot,))
    controller.set_swing_target(None)
    controller.allows_single_support = True
    controller.requires_final_double_support = False

    # Match the V4 landed_support_momentum_capture diagnostic exactly.  These
    # are existing transfer gains, not a new ablation gain sweep.
    hybrid_config = configs["experiments"].get("hybrid_recovery", {})
    controller.com_task_weight_override = float(hybrid_config["transfer_com_weight"])
    controller.com_task_kp_override = float(hybrid_config["transfer_com_kp"])
    controller.com_task_kd_override = float(hybrid_config["transfer_com_kd"])
    return controller, (landed_foot,)


def _run_profile(configs: dict, capture: dict, profile: str, duration_s: float, seed: int):
    model = make_model(configs)
    controller, active_contacts = _make_replay_controller(model, configs, capture, profile)
    runner = SimulationRunner(
        model,
        controller,
        duration_s=float(duration_s),
        control_timestep_s=float(configs["robot"]["control_timestep"]),
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
        initial_qvel=np.asarray(capture["qvel"], dtype=float),
        desired_torso=np.asarray(context["T_des_torso"], dtype=float),
        desired_pelvis=np.asarray(context["T_des_pelvis"], dtype=float),
        com_reference=np.asarray(capture["capture_com"], dtype=float),
    )
    arrays = run.log.arrays()
    observables = _replay_observables(model, run, capture, active_contacts)
    summary = _stability_summary(run, observables, capture, active_contacts, profile, configs)
    strict_window = _strict_stable_window(arrays, observables, run, configs)
    summary["strict_success"] = bool(strict_window)
    summary["active_contacts"] = "+".join(active_contacts)
    summary["initial_projected_state_sha256"] = capture["projected_state_sha256"]
    summary["diagnostic_history_rows"] = len(controller.diagnostic_history)
    if len(controller.diagnostic_history) != len(arrays["time_s"]):
        raise RuntimeError(
            f"diagnostic history misalignment for {profile}: "
            f"{len(controller.diagnostic_history)} != {len(arrays['time_s'])}"
        )
    event_metrics = _event_metrics(arrays, observables, active_contacts, configs)
    summary.update(event_metrics)
    return model, controller, run, active_contacts, observables, summary, event_metrics


def _strict_stable_window(arrays: dict[str, np.ndarray], observables: dict[str, np.ndarray], run, configs: dict) -> bool:
    if run.recovery is None or not bool(run.recovery.success):
        return False
    time_s = np.asarray(arrays["time_s"], dtype=float)
    if not len(time_s):
        return False
    stable_duration = float(recovery_config(configs).stable_duration_s)
    final_window = time_s >= time_s[-1] - stable_duration
    if not np.any(final_window):
        return False
    joint_ok = not np.any(np.asarray(arrays["joint_limit_violation"], dtype=bool)[final_window])
    support_ok = np.all(np.asarray(observables["support_margin_m"], dtype=float)[final_window] >= SUPPORT_MARGIN_THRESHOLD_M)
    return bool(joint_ok and support_ok)


def _finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else float("nan")


def _finite_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else float("nan")


def _sample_dt(time_s: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time_s, dtype=float))
    finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.median(finite)) if len(finite) else 0.004


def _sustained_onset(time_s: np.ndarray, mask: np.ndarray, samples: int = MIN_SUSTAINED_SAMPLES) -> float | None:
    time_s = np.asarray(time_s, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < samples:
        return None
    for index in range(len(mask) - samples + 1):
        if bool(np.all(mask[index : index + samples])):
            return float(time_s[index])
    return None


def _threshold_metrics(time_s: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (values >= float(threshold))
    dt = _sample_dt(time_s)
    duration = float(np.sum(mask) * dt)
    total = float(len(mask) * dt)
    return {
        "onset_s": _sustained_onset(time_s, mask),
        "duration_s": duration,
        "fraction": duration / total if total > 0.0 else float("nan"),
        "peak": _finite_max(values),
        "sample_count": int(np.sum(mask)),
        "mask": mask,
    }


def _event_time(time_s: np.ndarray, mask: np.ndarray) -> float | None:
    return _sustained_onset(np.asarray(time_s, dtype=float), np.asarray(mask, dtype=bool))


def _momentum_growth_time(time_s: np.ndarray, norm: np.ndarray) -> tuple[float | None, str]:
    values = np.asarray(norm, dtype=float)
    if len(values) == 0 or not np.isfinite(values[0]):
        return None, "invalid"
    threshold = abs(float(values[0])) * (1.0 + MOMENTUM_GROWTH_FRACTION)
    threshold = max(threshold, abs(float(values[0])) + 1.0e-3)
    onset = _sustained_onset(time_s, np.isfinite(values) & (values >= threshold))
    return onset, "linear_or_angular_norm_5pct"


def _event_metrics(
    arrays: dict[str, np.ndarray],
    observables: dict[str, np.ndarray],
    active_contacts: tuple[str, ...],
    configs: dict,
) -> dict:
    time_s = np.asarray(arrays["time_s"], dtype=float)
    torque = np.asarray(arrays["torque_utilization"], dtype=float)
    friction_values = np.asarray(arrays["actual_friction_utilization_post_step"], dtype=float)
    friction = np.full(len(time_s), np.nan, dtype=float)
    if friction_values.ndim == 2 and len(friction_values):
        finite_friction = np.where(np.isfinite(friction_values), friction_values, -np.inf)
        friction = np.max(finite_friction, axis=1)
        friction[~np.isfinite(friction)] = np.nan

    torque_events = _threshold_metrics(time_s, torque, HIGH_UTILIZATION_THRESHOLD)
    friction_events = _threshold_metrics(time_s, friction, HIGH_UTILIZATION_THRESHOLD)
    support = np.asarray(observables["support_margin_m"], dtype=float)
    linear_momentum = np.linalg.norm(np.asarray(observables["linear_momentum_world"], dtype=float), axis=1)
    angular_momentum = np.linalg.norm(np.asarray(observables["centroidal_angular_momentum_world"], dtype=float), axis=1)
    linear_onset, linear_label = _momentum_growth_time(time_s, linear_momentum)
    angular_onset, angular_label = _momentum_growth_time(time_s, angular_momentum)
    momentum_candidates = [(value, label) for value, label in ((linear_onset, "linear_momentum_growth"), (angular_onset, "angular_momentum_growth")) if value is not None]
    momentum_growth_onset = min(momentum_candidates, default=(None, ""))[0]
    momentum_growth_signal = min(momentum_candidates, default=(None, ""))[1]

    foot_index = 0 if active_contacts[0] == "left_foot" else 1
    contact_flags = np.asarray(arrays["contact_left_post_step" if foot_index == 0 else "contact_right_post_step"], dtype=bool)
    normal_force = np.asarray(arrays["actual_normal_force_post_step_N"], dtype=float)[:, foot_index]
    contact_threshold = float(configs["experiments"]["hybrid_recovery"].get("contact_force_threshold_N", 5.0))
    contact_loaded = contact_flags & (normal_force >= contact_threshold)
    joint_violation = np.asarray(arrays["joint_limit_violation"], dtype=bool)
    support_crossing = np.isfinite(support) & (support < 0.0)
    qp_slack = np.asarray(arrays["qp_slack_norm"], dtype=float)
    finite_slack = qp_slack[np.isfinite(qp_slack)]
    slack_baseline = float(np.median(finite_slack[: min(10, len(finite_slack))])) if len(finite_slack) else 0.0
    slack_threshold = max(QP_SLACK_SPIKE_MIN, QP_SLACK_SPIKE_MULTIPLIER * slack_baseline)
    qp_slack_event = np.isfinite(qp_slack) & (qp_slack >= slack_threshold)

    named_events = {
        "torque_high_utilization": torque_events["onset_s"],
        "friction_high_utilization": friction_events["onset_s"],
        "momentum_growth": momentum_growth_onset,
        "support_margin_crossing": _event_time(time_s, support_crossing),
        "joint_limit_violation": _event_time(time_s, joint_violation),
        "contact_loss": _event_time(time_s, ~contact_loaded),
        "qp_slack_spike": _event_time(time_s, qp_slack_event),
    }
    event_order = [name for name, value in sorted(named_events.items(), key=lambda pair: (float("inf") if pair[1] is None else pair[1], pair[0])) if value is not None]
    authority_names = {
        "torque_high_utilization",
        "friction_high_utilization",
        "joint_limit_violation",
        "qp_slack_spike",
        "support_margin_crossing",
        "contact_loss",
        "momentum_growth",
    }
    first_authority = next((name for name in event_order if name in authority_names), "")
    return {
        "high_utilization_threshold": HIGH_UTILIZATION_THRESHOLD,
        "torque_high_util_onset_s": torque_events["onset_s"],
        "torque_high_util_duration_s": torque_events["duration_s"],
        "torque_high_util_fraction": torque_events["fraction"],
        "torque_high_util_peak": torque_events["peak"],
        "torque_high_util_sample_count": torque_events["sample_count"],
        "friction_high_util_onset_s": friction_events["onset_s"],
        "friction_high_util_duration_s": friction_events["duration_s"],
        "friction_high_util_fraction": friction_events["fraction"],
        "friction_high_util_peak": friction_events["peak"],
        "friction_high_util_sample_count": friction_events["sample_count"],
        "momentum_growth_onset_s": momentum_growth_onset,
        "momentum_growth_signal": momentum_growth_signal,
        "linear_momentum_growth_onset_s": linear_onset,
        "angular_momentum_growth_onset_s": angular_onset,
        "support_margin_crossing_s": named_events["support_margin_crossing"],
        "joint_limit_violation_onset_s": named_events["joint_limit_violation"],
        "contact_loss_onset_s": named_events["contact_loss"],
        "qp_slack_spike_onset_s": named_events["qp_slack_spike"],
        "qp_slack_spike_threshold": slack_threshold,
        "first_authority_event": first_authority,
        "event_order": json.dumps(event_order),
        "event_times_json": json.dumps(named_events, sort_keys=True),
        "max_linear_momentum_Ns": _finite_max(linear_momentum),
        "max_centroidal_angular_momentum_Nms": _finite_max(angular_momentum),
        "min_support_margin_m": _finite_min(support),
        "max_qp_slack_norm": _finite_max(qp_slack),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _series_for_comparison(run_arrays: dict[str, np.ndarray], observables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    friction = np.asarray(run_arrays["actual_friction_utilization_post_step"], dtype=float)
    friction = np.max(np.where(np.isfinite(friction), friction, -np.inf), axis=1)
    friction[~np.isfinite(friction)] = np.nan
    return {
        "torque_utilization": np.asarray(run_arrays["torque_utilization"], dtype=float),
        "friction_utilization": friction,
        "qp_slack_norm": np.asarray(run_arrays["qp_slack_norm"], dtype=float),
        "support_margin_m": np.asarray(observables["support_margin_m"], dtype=float),
        "linear_momentum_norm": np.linalg.norm(np.asarray(observables["linear_momentum_world"], dtype=float), axis=1),
        "angular_momentum_norm": np.linalg.norm(np.asarray(observables["centroidal_angular_momentum_world"], dtype=float), axis=1),
    }


def _baseline_comparison(
    reference_json: Path,
    reference_observables: Path,
    run,
    observables: dict[str, np.ndarray],
    summary: dict,
) -> dict:
    reference_trial = reference_json.with_suffix(".npz")
    if not reference_trial.exists():
        raise FileNotFoundError(f"V4 reference trial NPZ not found beside {reference_json}")
    reference_meta = json.loads(reference_json.read_text(encoding="utf-8"))
    reference_run = _load_npz(reference_trial)
    reference_obs = _load_npz(reference_observables)
    current_arrays = run.log.arrays()
    current_series = _series_for_comparison(current_arrays, observables)
    reference_series = _series_for_comparison(reference_run, reference_obs)
    series_diffs = {}
    series_ok = True
    for name in current_series:
        current = current_series[name]
        reference = reference_series[name]
        if len(current) != len(reference):
            series_diffs[name] = {"length_current": len(current), "length_reference": len(reference), "max_abs_error": float("inf"), "ok": False}
            series_ok = False
            continue
        finite = np.isfinite(current) & np.isfinite(reference)
        if np.any(finite):
            delta = np.abs(current[finite] - reference[finite])
            scale = np.maximum(np.abs(reference[finite]), 1.0)
            max_abs = float(np.max(delta))
            max_rel = float(np.max(delta / scale))
            ok = bool(np.allclose(current[finite], reference[finite], atol=BASELINE_SERIES_ATOL, rtol=BASELINE_SERIES_RTOL))
        else:
            max_abs = float("nan")
            max_rel = float("nan")
            ok = True
        series_diffs[name] = {"max_abs_error": max_abs, "max_relative_error": max_rel, "ok": ok}
        series_ok &= ok

    reference_summary = reference_meta.get("diagnostic_summary", {})
    summary_fields = (
        "failure_reason",
        "max_linear_momentum_Ns",
        "max_centroidal_angular_momentum_Nms",
        "min_support_margin_m",
        "max_torque_utilization",
        "max_friction_utilization",
        "max_qp_slack_norm",
    )
    summary_diffs = {}
    summary_ok = True
    for field in summary_fields:
        current = summary.get(field)
        reference = reference_summary.get(field)
        if field == "failure_reason":
            ok = current == reference
            diff = None
        elif current is None or reference is None or not np.isfinite(float(current)) or not np.isfinite(float(reference)):
            ok = current == reference
            diff = None
        else:
            diff = float(current) - float(reference)
            ok = bool(np.isclose(float(current), float(reference), atol=BASELINE_SUMMARY_ATOL, rtol=BASELINE_SUMMARY_RTOL))
        summary_diffs[field] = {"current": current, "reference": reference, "difference": diff, "ok": ok}
        summary_ok &= ok

    active_reference = reference_meta.get("active_contacts", ["right_foot"])
    active_current = summary.get("active_contacts", "")
    active_ok = active_current == "+".join(active_reference) if isinstance(active_reference, list) else True
    comparable = bool(series_ok and summary_ok and active_ok)
    return {
        "comparable": comparable,
        "reference_json": str(reference_json),
        "reference_observables": str(reference_observables),
        "reference_trial": str(reference_trial),
        "current_profile": summary.get("variant"),
        "current_active_contacts": active_current,
        "reference_active_contacts": active_reference,
        "series": series_diffs,
        "summary": summary_diffs,
        "active_contacts_match": active_ok,
        "criteria": {
            "series_atol": BASELINE_SERIES_ATOL,
            "series_rtol": BASELINE_SERIES_RTOL,
            "summary_atol": BASELINE_SUMMARY_ATOL,
            "summary_rtol": BASELINE_SUMMARY_RTOL,
        },
    }


def _save_diagnostic_artifact(path: Path, controller: DiagnosticWholeBodyQPController, time_s: np.ndarray) -> None:
    arrays = controller.history_arrays()
    arrays["time_s"] = np.asarray(time_s, dtype=float)
    np.savez_compressed(path, **arrays)


def _plot_profile(
    time_s: np.ndarray,
    arrays: dict[str, np.ndarray],
    observables: dict[str, np.ndarray],
    diagnostic_arrays: dict[str, np.ndarray],
    summary: dict,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    friction = np.asarray(arrays["actual_friction_utilization_post_step"], dtype=float)
    friction = np.max(np.where(np.isfinite(friction), friction, -np.inf), axis=1)
    friction[~np.isfinite(friction)] = np.nan
    linear = np.linalg.norm(np.asarray(observables["linear_momentum_world"], dtype=float), axis=1)
    angular = np.linalg.norm(np.asarray(observables["centroidal_angular_momentum_world"], dtype=float), axis=1)
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(time_s, arrays["torque_utilization"], label="commanded torque utilization")
    axes[0].plot(time_s, friction, label="actual friction utilization")
    axes[0].axhline(HIGH_UTILIZATION_THRESHOLD, color="black", linestyle="--", linewidth=0.8, label="0.95 threshold")
    axes[0].set_ylabel("utilization")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot(time_s, linear, label="linear momentum norm")
    axes[1].plot(time_s, angular, label="centroidal angular momentum norm")
    axes[1].set_ylabel("momentum")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].plot(time_s, observables["support_margin_m"], label="support margin")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(SUPPORT_MARGIN_THRESHOLD_M, color="red", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("support [m]")
    axes[3].plot(time_s, arrays["qp_slack_norm"], label="QP slack")
    axes[3].plot(
        time_s,
        diagnostic_arrays["predicted_min_joint_limit_margin_rad"],
        label="predicted joint margin [rad]",
    )
    axes[3].set_ylabel("constraint state")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[4].plot(time_s, arrays["actual_normal_force_post_step_N"][:, 0], label="left normal force")
    axes[4].plot(time_s, arrays["actual_normal_force_post_step_N"][:, 1], label="right normal force")
    axes[4].set_ylabel("GRF z [N]")
    axes[4].set_xlabel("post-touchdown time [s]")
    axes[4].legend(fontsize=8, loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].set_title(
        f"{summary['variant']} | strict={summary['strict_success']} | "
        f"first authority={summary.get('first_authority_event', '')}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [str(row["profile"]) for row in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(x - 0.18, [float(row.get("max_torque_utilization", np.nan)) for row in rows], width=0.36, label="max torque")
    axes[0].bar(x + 0.18, [float(row.get("max_friction_utilization", np.nan)) for row in rows], width=0.36, label="max friction")
    axes[0].axhline(HIGH_UTILIZATION_THRESHOLD, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("peak utilization")
    axes[0].legend()
    axes[1].bar(x, [float(row.get("max_linear_momentum_Ns", np.nan)) for row in rows], label="linear momentum")
    axes[1].set_ylabel("max |p| [Ns]")
    axes[2].bar(x, [float(row.get("min_support_margin_m", np.nan)) for row in rows], label="min support margin")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("min support [m]")
    axes[2].set_xticks(x, names, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--projected-state", type=Path, required=True)
    parser.add_argument("--capture-state", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reference-observables", type=Path, required=True)
    parser.add_argument("--replay-duration", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.replay_duration <= 0.0:
        raise SystemExit("--replay-duration must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output root: {output_root}")

    configs = load_configs(ROOT)
    capture, state_provenance = _load_projected_capture(
        args.projected_state.resolve(),
        args.capture_state.resolve(),
        configs,
    )
    if not args.reference_json.exists() or not args.reference_observables.exists():
        raise FileNotFoundError("V4 baseline reference artifacts are required")

    data_root = output_root / "data" / "trials"
    diagnostic_root = output_root / "data" / "diagnostic"
    figures_root = output_root / "figures"
    logs_root = output_root / "logs"
    for path in (data_root, diagnostic_root, figures_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)

    source = _source_metadata()
    profile_definitions = {name: vars(PROFILE_DEFINITIONS[name]) for name in PROFILES}
    profile_hash = hashlib.sha256(_json(profile_definitions).encode("utf-8")).hexdigest()
    command = " ".join(sys.argv)
    write_execution_manifest(
        logs_root / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "post_touchdown_wbc_ablation",
            "question": "Which post-touchdown WBC task or physical authority mechanism fails first at the 75 N projected touchdown state?",
            "run_id": output_root.name,
            "command": command,
            "replay_duration_s": float(args.replay_duration),
            "profiles": list(PROFILES),
            "profile_definitions": profile_definitions,
            "profile_definitions_sha256": profile_hash,
            "high_utilization_threshold": HIGH_UTILIZATION_THRESHOLD,
            "momentum_growth_fraction": MOMENTUM_GROWTH_FRACTION,
            "support_margin_strict_threshold_m": SUPPORT_MARGIN_THRESHOLD_M,
            "external_push_removed": True,
            "planning_removed": True,
            "swing_generation_removed": True,
            "landing_gate_removed": True,
            "production_controller_unchanged": True,
            "source_provenance": source,
            "config_canonical_sha256": _canonical_config_sha(configs),
            "raw_config_sha256": _raw_config_hashes(),
            "model_path": str(resolve_model_path(configs)),
            "model_sha256": _sha256_file(resolve_model_path(configs)),
            "state_provenance": state_provenance,
            "reference_v4_json": str(args.reference_json.resolve()),
            "reference_v4_observables": str(args.reference_observables.resolve()),
            "interpretation_rule": "Temporal event order is diagnostic evidence, not proof of causality; no profile is called recovered without the strict criterion.",
        },
    )

    rows: list[dict] = []
    baseline_gate = None
    for profile in PROFILES:
        model, controller, run, active_contacts, observables, summary, event_metrics = _run_profile(
            configs,
            capture,
            profile,
            args.replay_duration,
            args.seed,
        )
        arrays = run.log.arrays()
        trial_id = f"touchdown_75N_forward_offset0_{profile}"
        safe_trial = _safe_name(trial_id)
        trial_path = data_root / f"{safe_trial}.npz"
        observable_path = diagnostic_root / f"{safe_trial}_observables.npz"
        controller_path = diagnostic_root / f"{safe_trial}_controller.npz"
        metadata = {
            "experiment": "post_touchdown_wbc_ablation",
            "phase": "replay",
            "run_id": output_root.name,
            "trial_id": trial_id,
            "profile": profile,
            "profile_definition": vars(PROFILE_DEFINITIONS[profile]),
            "config": configs,
            "source_provenance": source,
            "config_canonical_sha256": _canonical_config_sha(configs),
            "raw_config_sha256": _raw_config_hashes(),
            "model_path": str(resolve_model_path(configs)),
            "model_sha256": _sha256_file(resolve_model_path(configs)),
            "projected_state": state_provenance,
            "active_contacts": list(active_contacts),
            "initial_state": "exact_projected_touchdown_qpos_qvel",
            "external_push_removed": True,
            "planning_removed": True,
            "swing_generation_removed": True,
            "landing_gate_removed": True,
            "observable_path": str(observable_path.relative_to(output_root).as_posix()),
            "controller_diagnostic_path": str(controller_path.relative_to(output_root).as_posix()),
            "diagnostic_summary": summary,
            "event_metrics": event_metrics,
            "command": command,
        }
        np.savez_compressed(
            observable_path,
            **observables,
            replay_time_s=np.asarray(arrays["time_s"], dtype=float),
        )
        _save_diagnostic_artifact(controller_path, controller, np.asarray(arrays["time_s"], dtype=float))
        save_run(run, trial_path, metadata)
        _plot_profile(
            np.asarray(arrays["time_s"], dtype=float),
            arrays,
            observables,
            controller.history_arrays(),
            summary,
            figures_root / f"{safe_trial}.png",
        )
        row = dict(summary)
        row.update({
            "trial_id": trial_id,
            "profile": profile,
            "projected_state_sha256": capture["projected_state_sha256"],
            "source_commit": source["source_commit"],
            "active_contacts": "+".join(active_contacts),
        })
        rows.append(row)

        if profile == "baseline_landed_support":
            baseline_gate = _baseline_comparison(
                args.reference_json.resolve(),
                args.reference_observables.resolve(),
                run,
                observables,
                summary,
            )
            (logs_root / "baseline_comparison.json").write_text(
                json.dumps(baseline_gate, indent=2, default=str), encoding="utf-8",
            )
            if not bool(baseline_gate["comparable"]):
                (logs_root / "summary.json").write_text(
                    json.dumps({
                        "run_id": output_root.name,
                        "status": "STOPPED_BASELINE_NOT_COMPARABLE",
                        "baseline_comparison": baseline_gate,
                        "source_provenance": source,
                    }, indent=2, default=str), encoding="utf-8",
                )
                print(json.dumps(baseline_gate, indent=2, default=str))
                raise SystemExit(2)

    write_csv(rows, output_root / "replay_summary.csv")
    write_csv(
        [
            {
                "profile": row.get("profile"),
                "first_authority_event": row.get("first_authority_event"),
                "event_order": row.get("event_order"),
                "torque_high_util_onset_s": row.get("torque_high_util_onset_s"),
                "torque_high_util_duration_s": row.get("torque_high_util_duration_s"),
                "torque_high_util_fraction": row.get("torque_high_util_fraction"),
                "friction_high_util_onset_s": row.get("friction_high_util_onset_s"),
                "friction_high_util_duration_s": row.get("friction_high_util_duration_s"),
                "friction_high_util_fraction": row.get("friction_high_util_fraction"),
                "momentum_growth_onset_s": row.get("momentum_growth_onset_s"),
                "support_margin_crossing_s": row.get("support_margin_crossing_s"),
                "joint_limit_violation_onset_s": row.get("joint_limit_violation_onset_s"),
                "contact_loss_onset_s": row.get("contact_loss_onset_s"),
                "qp_slack_spike_onset_s": row.get("qp_slack_spike_onset_s"),
            }
            for row in rows
        ],
        logs_root / "authority_events.csv",
    )
    _plot_comparison(rows, figures_root / "profile_comparison.png")
    final_summary = {
        "run_id": output_root.name,
        "status": "COMPLETED",
        "profile_count": len(rows),
        "profiles": list(PROFILES),
        "strict_success_count": int(sum(bool(row.get("strict_success", False)) for row in rows)),
        "baseline_gate": baseline_gate,
        "source_provenance": source,
        "state_provenance": state_provenance,
        "artifact_note": "All negative and positive replay results, controller diagnostics, event metrics, metadata, and figures are retained under this new root.",
    }
    (logs_root / "summary.json").write_text(json.dumps(final_summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
