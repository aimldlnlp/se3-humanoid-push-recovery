"""Run deterministic G1 Adaptive Recovery Arena scenarios.

Each scenario is a fresh, independently seeded MuJoCo trial.  The script
stores the full trajectory, event-level controller summary, telemetry figure,
and optional video/GIF under a new output root so historical benchmark and
revalidation artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from common import flatten_result, load_configs, make_push, run_trial, save_run, write_csv, write_execution_manifest
from se3_whole_body_control.config import resolve_model_path
from se3_whole_body_control.visualization.plots import plot_arena_telemetry
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video, make_gif


@dataclass(frozen=True)
class ArenaScenario:
    name: str
    magnitude_N: float
    direction_deg: float
    duration_s: float
    description: str


DEFAULT_SCENARIOS = (
    ArenaScenario("stabilize_40N", 40.0, 0.0, 0.15, "small forward push; remain in double support"),
    ArenaScenario("step_75N", 75.0, 0.0, 0.15, "near-boundary forward push; attempt measured foot recovery"),
    ArenaScenario("lateral_70N", 70.0, 90.0, 0.15, "lateral push; test direction-aware foot selection"),
    ArenaScenario("overload_100N", 100.0, 0.0, 0.15, "strong forward push; preserve a transparent failure case"),
)


def _scenario_lookup(names: list[str] | None) -> list[ArenaScenario]:
    scenarios = {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}
    if not names:
        return list(DEFAULT_SCENARIOS)
    missing = [name for name in names if name not in scenarios]
    if missing:
        raise ValueError(f"unknown arena scenario(s): {missing}; choose from {sorted(scenarios)}")
    return [scenarios[name] for name in names]


def _overlay_data(run, push, label: str) -> list[dict]:
    arrays = run.log.arrays()
    overlay = []
    for index in range(len(run.qpos_history)):
        target = arrays.get("planned_foot_target_world", np.full((len(run.qpos_history), 3), np.nan))[index]
        overlay.append({
            "time_s": float(arrays["time_s"][index]),
            "controller": label,
            "push_magnitude_N": push.magnitude_N,
            "push_direction_deg": push.direction_deg,
            "status": str(arrays["qp_status"][index]),
            "control_mode": str(arrays["control_mode"][index]),
            "step_phase": str(arrays["step_phase"][index]),
            "event_label": str(arrays.get("event_label", np.full(len(run.qpos_history), ""))[index]),
            "support_margin_m": float(arrays.get("support_margin_m", np.full(len(run.qpos_history), np.nan))[index]),
            "com_world": arrays["com_world"][index],
            "feet_xy": arrays["foot_xy_world"][index],
            "planned_foot_target_world": target,
            "push_point_world": arrays["torso_position"][index],
            "push_force": arrays["push_force"][index],
            "contact_left": bool(arrays["contact_left"][index]),
            "contact_right": bool(arrays["contact_right"][index]),
        })
    return overlay


def _render_trial(model, run, push, label: str, scenario_root: Path, render_fps: int = 30) -> list[Path]:
    frames = scenario_root / "frames"
    if frames.exists():
        raise FileExistsError(f"refusing to overwrite existing frame directory: {frames}")
    frames.mkdir(parents=True, exist_ok=False)
    configs = getattr(model, "_arena_configs", None)
    stride = max(1, int(round(1.0 / (0.004 * render_fps))))
    render_trial_frames(
        model,
        run.qpos_history,
        frames,
        width=1280,
        height=720,
        stride=stride,
        overlay_data=_overlay_data(run, push, label),
    )
    video_path = scenario_root / f"{scenario_root.name}.mp4"
    gif_path = scenario_root / f"{scenario_root.name}.gif"
    encode_video(frames, video_path, fps=render_fps)
    make_gif(frames, gif_path, fps=12)
    return [video_path, gif_path]


def _build_story(rendered: list[Path], output_root: Path, fps: int = 30) -> list[Path]:
    if not rendered:
        return []
    story_frames = output_root / "story_frames"
    story_frames.mkdir(parents=True, exist_ok=False)
    frame_index = 0
    for scenario_dir in rendered:
        for frame in sorted((scenario_dir / "frames").glob("frame_*.png")):
            shutil.copy2(frame, story_frames / f"frame_{frame_index:06d}.png")
            frame_index += 1
    video = output_root / "g1_adaptive_recovery_arena_story.mp4"
    gif = output_root / "g1_adaptive_recovery_arena_story.gif"
    encode_video(story_frames, video, fps=fps)
    make_gif(story_frames, gif, fps=12)
    return [video, gif]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "arena")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    configs = load_configs(ROOT)
    scenarios = _scenario_lookup(args.scenarios)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty arena output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    data_root = output_root / "data"
    trial_root = data_root / "trials"
    figures_root = output_root / "figures"
    logs_root.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    run_id = f"arena-{args.seed}-{output_root.name}"
    model_path = resolve_model_path(configs)
    model_sha256 = __import__("hashlib").sha256(model_path.read_bytes()).hexdigest()
    write_execution_manifest(
        logs_root / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "adaptive_recovery_arena",
            "run_id": run_id,
            "controller": "adaptive_hybrid_se3_wbc",
            "scenario_names": [scenario.name for scenario in scenarios],
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "duration_s": args.duration or configs["robot"]["duration_s"],
            "command": " ".join(sys.argv),
            "model_path": str(Path(configs["robot"]["model_path"])),
            "model_sha256": model_sha256,
            "rendered": not args.no_render,
            "artifact_root": str(output_root.name),
            "historical_results_policy": "new output root; results/data and results/revalidation are untouched",
        },
    )

    rows = []
    rendered_dirs = []
    artifact_paths = []
    for scenario in scenarios:
        scenario_root = output_root / scenario.name
        scenario_root.mkdir(parents=True, exist_ok=False)
        push = make_push(
            configs,
            magnitude=scenario.magnitude_N,
            direction_deg=scenario.direction_deg,
            duration=scenario.duration_s,
        )
        model, run = run_trial(
            "hybrid_wbc",
            configs,
            push=push,
            duration=args.duration,
            seed=args.seed,
            classify=True,
        )
        model._arena_configs = configs
        summary = run.metadata.get("controller_summary", {})
        row = flatten_result(
            run,
            "adaptive_hybrid_se3_wbc",
            push,
            scenario.name,
            seed=args.seed,
            extra={
                "scenario": scenario.name,
                "description": scenario.description,
                "step_triggered": bool(summary.get("step_triggered", False)),
                "attempted_step_count": int(summary.get("attempted_step_count", 0)),
                "landed_step_count": int(summary.get("step_count", 0)),
                "final_mode": summary.get("final_mode", ""),
                "final_phase": summary.get("final_phase", ""),
            },
        )
        rows.append(row)
        trial_path = trial_root / f"{scenario.name}.npz"
        save_run(
            run,
            trial_path,
            {
                "run_id": run_id,
                "trial_id": scenario.name,
                "controller": "adaptive_hybrid_se3_wbc",
                "scenario": asdict(scenario),
                "config": configs,
                "push": push.__dict__,
                "command": " ".join(sys.argv),
                "model_sha256": model_sha256,
            },
        )
        plot_arena_telemetry(run.log, figures_root, name=f"{scenario.name}_telemetry")
        if not args.no_render:
            _render_trial(model, run, push, "G1 Adaptive Recovery", scenario_root, render_fps=int(configs["robot"].get("render_fps", 30)))
            rendered_dirs.append(scenario_root)
        artifact_paths.extend(str(path.relative_to(output_root)) for path in scenario_root.rglob("*") if path.is_file())

    rows.sort(key=lambda row: str(row["trial_id"]))
    write_csv(rows, data_root / "arena_summary.csv")
    if rendered_dirs:
        artifact_paths.extend(str(path.relative_to(output_root)) for path in _build_story(rendered_dirs, output_root, fps=int(configs["robot"].get("render_fps", 30))))
    summary = {
        "run_id": run_id,
        "scenario_count": len(rows),
        "scenarios": rows,
        "artifact_paths": sorted(set(artifact_paths + ["logs/manifest.json", "data/arena_summary.csv"])),
    }
    (logs_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "scenario_count": len(rows), "summary": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
