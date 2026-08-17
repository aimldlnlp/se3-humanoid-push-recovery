"""Diagnose why the measured hybrid stepping branch does not recover.

This is an upper-bound diagnostic, not a controller tuning sweep.  It keeps
the MuJoCo plant and SE(3) QP unchanged while comparing:

* the validated fixed-foot WBC;
* the current bounded hybrid controller;
* the current first-step target with the second step disabled;
* a first-step target based on the linear-inverted-pendulum capture point; and
* a deliberately optimistic reach-bound target.

The last two variants are diagnostic alternatives.  They must not be used as
headline recovery results unless the robot reaches a stable final state under
the normal recovery classifier.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    flatten_result,
    load_configs,
    make_model,
    make_push,
    recovery_config,
    run_trial,
    save_run,
    write_csv,
    write_execution_manifest,
)
from se3_whole_body_control.config import resolve_model_path  # noqa: E402
from se3_whole_body_control.control.hybrid_recovery import HybridRecoveryController  # noqa: E402
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner  # noqa: E402
from se3_whole_body_control.control.tasks import com_jacobian  # noqa: E402


VARIANTS = (
    "fixed_foot",
    "current_hybrid",
    "one_step_current",
    "one_step_capture_point",
    "one_step_reach_edge",
    "one_step_landed_support",
)


class DiagnosticHybridController(HybridRecoveryController):
    """Hybrid controller with a controlled first-step target alternative."""

    def __init__(self, *args, diagnostic_variant: str, **kwargs):
        self.diagnostic_variant = str(diagnostic_variant)
        super().__init__(*args, **kwargs)
        if self.diagnostic_variant == "one_step_landed_support":
            # This is a contact-transition ablation.  A stable one-foot state
            # is a valid diagnostic outcome, even though the production arena
            # deliberately requires final double support.
            self.requires_final_double_support = False

    def _clamp_target_to_reach(self, target_xy: np.ndarray, support_foot: str) -> np.ndarray:
        support_center = self.model.body_pose(support_foot)[:2, 3]
        relative = np.asarray(target_xy, dtype=float) - support_center
        max_reach = float(self.hybrid_config.get("max_step_reach_m", 0.46))
        norm = float(np.linalg.norm(relative))
        if norm > max_reach:
            return support_center + relative / max(norm, 1e-12) * max_reach
        return np.asarray(target_xy, dtype=float)

    def _make_step_target(self, swing_foot: str, support_foot: str, direction_xy: np.ndarray) -> np.ndarray:
        target = super()._make_step_target(swing_foot, support_foot, direction_xy)
        # Only alter the first step.  The study is asking whether one landing
        # can be made viable; it is not a second-step tuning experiment.
        if self.step_count != 0:
            return target

        if self.diagnostic_variant == "one_step_capture_point":
            velocity = com_jacobian(self.model) @ self.model.data.qvel
            com = self.model.center_of_mass()
            # Standard LIPM capture-point upper-bound diagnostic.  The target
            # is still projected into the existing support-relative reach set.
            com_height = max(float(com[2]), 0.50)
            capture_horizon = math.sqrt(com_height / 9.81)
            capture_point = com[:2] + velocity[:2] * capture_horizon
            target[:2, 3] = self._clamp_target_to_reach(capture_point, support_foot)
        elif self.diagnostic_variant == "one_step_reach_edge":
            support_center = self.model.body_pose(support_foot)[:2, 3]
            max_reach = float(self.hybrid_config.get("max_step_reach_m", 0.46))
            target[:2, 3] = support_center + normalize_direction(direction_xy) * max_reach
        return target

    def _update_landing(self) -> None:
        super()._update_landing()
        if (
            self.diagnostic_variant == "one_step_landed_support"
            and self.step_phase == "recovered_step"
            and self.step_event is not None
            and self.step_count >= 1
            and self.contact_names != (self.step_event.swing_foot,)
        ):
            # Keep only the foot that actually passed the measured touchdown
            # gate.  Holding the measured CoM avoids hiding a contact-mode
            # failure behind an arbitrary post-landing target.
            self.set_active_contacts((self.step_event.swing_foot,))
            self.set_swing_target(None)
            self.com_des = self.model.center_of_mass().copy()


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-12)


def _custom_hybrid_trial(
    variant: str,
    configs: dict,
    push,
    duration_s: float,
    seed: int,
):
    model = make_model(configs)
    hybrid_config = copy.deepcopy(configs["experiments"].get("hybrid_recovery", {}))
    if variant.startswith("one_step_"):
        hybrid_config["max_steps"] = 1
    controller = DiagnosticHybridController(
        model,
        configs["controller"],
        configs["experiments"].get("recovery", {}),
        hybrid_config,
        diagnostic_variant=variant,
    )
    runner = SimulationRunner(
        model,
        controller,
        duration_s=duration_s,
        control_timestep_s=configs["robot"]["control_timestep"],
        warmup_duration_s=configs["robot"].get("warmup_duration_s", 0.4),
        warmup_reanchor=True,
    )
    return model, runner.run(
        push=push,
        recovery_config=recovery_config(configs),
        classify=True,
        seed=seed,
    ), hybrid_config


def _event_times(run) -> dict[str, list[float]]:
    events = run.metadata.get("controller_summary", {}).get("events", [])
    out: dict[str, list[float]] = {}
    for event in events:
        out.setdefault(str(event.get("label", "")), []).append(float(event.get("time_s", np.nan)))
    return out


def _at_time(values: np.ndarray, time_s: np.ndarray, event_time: float | None) -> float:
    if event_time is None or len(time_s) == 0:
        return float("nan")
    index = int(np.argmin(np.abs(time_s - float(event_time))))
    value = np.asarray(values)[index]
    if np.ndim(value) == 0:
        return float(value)
    return float(np.linalg.norm(value))


def _longest_true_duration(time_s: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not np.any(mask):
        return 0.0
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return float(max((time_s[end] - time_s[start] for start, end in zip(starts, ends)), default=0.0))


def _diagnostic_row(run, variant: str, push, trial_id: str, seed: int) -> dict:
    row = flatten_result(run, variant, push, trial_id, seed=seed)
    arrays = run.log.arrays()
    time_s = np.asarray(arrays["time_s"], dtype=float)
    com = np.asarray(arrays["com_world"], dtype=float)
    if len(time_s) >= 2:
        com_velocity = np.gradient(com, time_s, axis=0)
    else:
        com_velocity = np.zeros_like(com)
    com_speed = np.linalg.norm(com_velocity[:, :2], axis=1) if len(com_velocity) else np.zeros(0)
    events = _event_times(run)
    first_touchdown = events.get("TOUCHDOWN", [None])[0] if events.get("TOUCHDOWN") else None
    if first_touchdown is not None:
        post_mask = (time_s >= first_touchdown) & (time_s <= first_touchdown + 0.30)
        post_speed = com_speed[post_mask]
        post_speed_max = float(np.max(post_speed)) if len(post_speed) else float("nan")
        post_speed_end = float(post_speed[-1]) if len(post_speed) else float("nan")
    else:
        post_speed_max = float("nan")
        post_speed_end = float("nan")

    target = np.asarray(arrays.get("planned_foot_target_world", np.full((len(time_s), 3), np.nan)), dtype=float)
    foot_xy = np.asarray(arrays.get("foot_xy_world", np.full((len(time_s), 4), np.nan)), dtype=float).reshape(-1, 2, 2)
    swing = arrays.get("swing_foot", np.asarray([""] * len(time_s))).astype(str)
    target_error = np.full(len(time_s), np.nan, dtype=float)
    for index in range(len(time_s)):
        if not np.all(np.isfinite(target[index])):
            continue
        foot_index = 0 if swing[index] == "left_foot" else 1 if swing[index] == "right_foot" else -1
        if foot_index >= 0:
            target_error[index] = np.linalg.norm(foot_xy[index, foot_index] - target[index, :2])
    finite_target_error = target_error[np.isfinite(target_error)]
    modes = arrays.get("control_mode", np.asarray([""] * len(time_s))).astype(str)
    event_time = first_touchdown
    finite_support_margin = np.asarray(arrays["support_margin_m"], dtype=float)
    finite_support_margin = finite_support_margin[np.isfinite(finite_support_margin)]
    row.update({
        "first_liftoff_s": events.get("LIFTOFF", [None])[0] if events.get("LIFTOFF") else "",
        "first_swing_complete_s": events.get("SWING_COMPLETE", [None])[0] if events.get("SWING_COMPLETE") else "",
        "first_touchdown_s": first_touchdown if first_touchdown is not None else "",
        "second_touchdown_s": events.get("TOUCHDOWN", [None, None])[1] if len(events.get("TOUCHDOWN", [])) > 1 else "",
        "first_touchdown_com_speed_m_s": _at_time(com_speed, time_s, event_time),
        "first_touchdown_support_margin_m": _at_time(arrays["support_margin_m"], time_s, event_time),
        "first_touchdown_qp_slack_norm": _at_time(arrays["qp_slack_norm"], time_s, event_time),
        "first_touchdown_torque_utilization": _at_time(arrays["torque_utilization"], time_s, event_time),
        "first_touchdown_friction_utilization": _at_time(
            np.nanmax(arrays["actual_friction_utilization_post_step"], axis=1), time_s, event_time,
        ),
        "post_touchdown_com_speed_max_0p3s_m_s": post_speed_max,
        "post_touchdown_com_speed_end_0p3s_m_s": post_speed_end,
        "longest_single_support_s": _longest_true_duration(time_s, modes == "single_support"),
        "min_support_margin_m": float(np.min(finite_support_margin)) if len(finite_support_margin) else float("nan"),
        "max_target_error_m": float(np.max(finite_target_error)) if len(finite_target_error) else float("nan"),
        "max_torque_utilization": float(np.nanmax(arrays["torque_utilization"])),
        "max_qp_slack_norm": float(np.nanmax(arrays["qp_slack_norm"])),
        "qp_p95_ms": float(np.percentile(arrays["qp_solve_time_s"] * 1000.0, 95)),
        "qp_max_ms": float(np.max(arrays["qp_solve_time_s"] * 1000.0)),
        "events_json": json.dumps(run.metadata.get("controller_summary", {}).get("events", []), sort_keys=True),
        "step_history_json": json.dumps(run.metadata.get("controller_summary", {}).get("step_history", []), sort_keys=True),
    })
    return row


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _plot_diagnostic(run, variant: str, magnitude: float, direction: float, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    arrays = run.log.arrays()
    time_s = arrays["time_s"]
    com = arrays["com_world"]
    com_velocity = np.gradient(com, time_s, axis=0) if len(time_s) >= 2 else np.zeros_like(com)
    com_speed = np.linalg.norm(com_velocity[:, :2], axis=1)
    events = run.metadata.get("controller_summary", {}).get("events", [])
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_s, com_speed, color="#1769aa", label="finite-difference CoM speed")
    axes[0].set_ylabel("CoM speed [m/s]")
    axes[1].plot(time_s, arrays["support_margin_m"], color="#238b45", label="support margin")
    axes[1].axhline(0.0, color="#444", linewidth=0.8)
    axes[1].set_ylabel("support margin [m]")
    axes[2].plot(time_s, arrays["qp_slack_norm"], color="#d95f02", label="QP slack")
    axes[2].plot(time_s, arrays["torque_utilization"], color="#7b3294", label="torque utilization")
    axes[2].set_ylabel("slack / utilization")
    modes = arrays.get("control_mode", np.asarray([""] * len(time_s))).astype(str)
    mode_names = ["double_support", "transfer", "single_support", "landing", "failed_recovery"]
    mode_id = np.array([mode_names.index(value) if value in mode_names else -1 for value in modes])
    axes[3].step(time_s, mode_id, where="post", color="#222", label="controller mode")
    axes[3].set_yticks(range(len(mode_names)), mode_names, fontsize=8)
    axes[3].set_ylabel("mode")
    axes[3].set_xlabel("time [s]")
    for event in events:
        event_time = float(event.get("time_s", np.nan))
        if np.isfinite(event_time):
            for axis in axes:
                axis.axvline(event_time, color="#cc4c02", linewidth=0.7, alpha=0.45)
            axes[3].text(event_time, len(mode_names) - 0.2, str(event.get("label", "")), rotation=90, fontsize=7, va="top")
    axes[0].set_title(f"{variant}: {magnitude:g} N @ {direction:g} deg")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[60.0, 70.0, 75.0])
    parser.add_argument("--directions", type=float, nargs="+", default=[0.0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = load_configs(ROOT)
    output_root = args.output_root.resolve()
    data_root = output_root / "data" / "trials"
    logs_root = output_root / "logs"
    figures_root = output_root / "figures"
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output root: {output_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    write_execution_manifest(
        logs_root / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "diagnose_step_viability",
            "question": "Does a physically viable one-step recovery exist near the current stepping boundary?",
            "run_id": output_root.name,
            "model_path": str(resolve_model_path(configs)),
            "magnitudes_N": [float(value) for value in args.magnitudes],
            "directions_deg": [float(value) for value in args.directions],
            "variants": list(args.variants),
            "duration_s": float(args.duration),
            "oracle_definition": {
                "one_step_capture_point": "LIPM capture point at measured step trigger, projected into existing max reach",
                "one_step_reach_edge": "support-foot center plus existing max reach in measured motion direction",
            },
            "interpretation_rule": "Variants are upper-bound diagnostics; no variant is a recovery claim without the normal classifier.",
            "command": command,
        },
    )

    rows: list[dict] = []
    for magnitude in args.magnitudes:
        for direction in args.directions:
            push = make_push(configs, magnitude=magnitude, direction_deg=direction)
            for variant in args.variants:
                trial_id = f"{variant}_{magnitude:g}N_{direction:g}deg_seed{args.seed}"
                if variant == "fixed_foot":
                    model, run = run_trial(
                        "se3_wbc", configs, push=push, duration=args.duration, seed=args.seed, classify=True,
                    )
                    variant_config = {}
                else:
                    model, run, variant_config = _custom_hybrid_trial(
                        variant, configs, push, args.duration, args.seed,
                    )
                metadata = {
                    "run_id": output_root.name,
                    "trial_id": trial_id,
                    "experiment": "diagnose_step_viability",
                    "diagnostic_variant": variant,
                    "config": configs,
                    "variant_config": variant_config,
                    "push": push.__dict__,
                    "command": command,
                    "diagnostic_question": "one-step viability near stepping boundary",
                }
                save_run(run, data_root / f"{trial_id}.npz", metadata)
                row = _diagnostic_row(run, variant, push, trial_id, args.seed)
                rows.append(row)
                _plot_diagnostic(
                    run,
                    variant,
                    magnitude,
                    direction,
                    figures_root / f"{_safe_name(trial_id)}.png",
                )
                print(json.dumps({"trial_id": trial_id, "success": row["success"], "failure_reason": row["failure_reason"]}))

    write_csv(rows, output_root / "diagnostic_summary.csv")
    summary = {
        "run_id": output_root.name,
        "trial_count": len(rows),
        "variants": list(args.variants),
        "magnitudes_N": [float(value) for value in args.magnitudes],
        "directions_deg": [float(value) for value in args.directions],
        "success_counts": {
            variant: int(sum(bool(row["success"]) for row in rows if row["controller"] == variant))
            for variant in args.variants
        },
        "artifact_note": "Raw NPZ/JSON trial bundles and diagnostic plots are retained; no historical result root is modified.",
    }
    (logs_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
