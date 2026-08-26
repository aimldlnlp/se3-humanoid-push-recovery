"""Render presentation videos from saved trajectories without rerunning trials.

The renderer loads the frozen ``qpos_history`` stored in NPZ artifacts and
only performs MuJoCo forward kinematics plus image rendering. It never calls
the controller or creates new experiment logs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from common import load_configs, make_model
from se3_whole_body_control.visualization.fonts import pil_font
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video, make_gif


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _push_summary(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    force = np.asarray(arrays.get("push_force", np.zeros((0, 3))), dtype=float)
    active = np.linalg.norm(force[:, :2], axis=1) > 1e-9 if force.ndim == 2 and len(force) else np.zeros(0, dtype=bool)
    if not np.any(active):
        return 0.0, 0.0
    sample = force[np.flatnonzero(active)[0], :2]
    return float(np.linalg.norm(sample)), float(np.rad2deg(np.arctan2(sample[1], sample[0])) % 360.0)


def _overlay_data(arrays: dict[str, np.ndarray], controller: str, push: tuple[float, float]) -> list[dict[str, object]]:
    magnitude, direction = push
    count = len(arrays["qpos_history"])
    return [
        {
            "time_s": arrays["time_s"][i],
            "controller": controller,
            "push_magnitude_N": magnitude,
            "push_direction_deg": direction,
            "status": arrays["qp_status"][i],
            "com_world": arrays["com_world"][i],
            "feet_xy": arrays["foot_xy_world"][i],
            "push_point_world": arrays["torso_position"][i],
            "push_force": arrays["push_force"][i],
            "contact_left": arrays["contact_left"][i],
            "contact_right": arrays["contact_right"][i],
        }
        for i in range(count)
    ]


def _stride(configs: dict) -> int:
    return max(1, int(round(1.0 / (configs["robot"]["control_timestep"] * configs["robot"]["render_fps"]))))


def _render_trial_video(
    configs: dict,
    npz_path: Path,
    output_root: Path,
    output_name: str,
    controller_label: str,
    *,
    width: int,
    height: int,
    gif_width: int = 960,
) -> list[Path]:
    arrays = _load(npz_path)
    model = make_model(configs)
    frame_dir = output_root / "frames" / npz_path.stem
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = render_trial_frames(
        model,
        arrays["qpos_history"],
        frame_dir,
        width=width,
        height=height,
        stride=_stride(configs),
        overlay_data=_overlay_data(arrays, controller_label, _push_summary(arrays)),
    )
    encode_video(frame_dir, output_root / f"{output_name}.mp4", fps=configs["robot"]["render_fps"])
    make_gif(frame_dir, output_root / f"{output_name}.gif", fps=12, max_width=gif_width)
    return paths


def _render_comparison(configs: dict, data_root: Path, output_root: Path) -> None:
    rendered: list[list[Path]] = []
    arrays_by_controller: dict[str, dict[str, np.ndarray]] = {}
    for controller, label in (("pd", "PD"), ("se3_wbc", "SE(3) WBC")):
        arrays = _load(data_root / f"single_push_{controller}.npz")
        arrays_by_controller[controller] = arrays
        model = make_model(configs)
        frame_dir = output_root / "frames" / f"comparison_{controller}"
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        rendered.append(render_trial_frames(
            model,
            arrays["qpos_history"],
            frame_dir,
            width=960,
            height=540,
            stride=_stride(configs),
            overlay_data=_overlay_data(arrays, label, _push_summary(arrays)),
        ))

    output_dir = output_root / "frames" / "comparison_combined"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(paths) for paths in rendered)
    reference = arrays_by_controller["se3_wbc"]
    push = _push_summary(reference)
    title_font = pil_font(28, weight="bold")
    body_font = pil_font(20)
    stride = _stride(configs)
    for i in range(count):
        left = Image.open(rendered[0][i]).convert("RGB")
        right = Image.open(rendered[1][i]).convert("RGB")
        header_height = 52
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + header_height), "white")
        canvas.paste(left, (0, header_height)); canvas.paste(right, (left.width, header_height))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, header_height), fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.line((left.width, header_height, left.width, canvas.height), fill=(148, 163, 184), width=1)
        time_index = min(i * stride, len(reference["time_s"]) - 1)
        time_s = float(reference["time_s"][time_index])
        shared = f"same {push[0]:.0f} N push  |  t = {time_s:.2f} s"
        draw.text((16, 10), "PD", font=title_font, fill=(91, 103, 112))
        draw.text((left.width + 16, 10), "SE(3) WBC", font=title_font, fill=(0, 114, 178))
        bbox = draw.textbbox((0, 0), shared, font=body_font)
        shared_x = left.width - (bbox[2] - bbox[0]) - 18
        draw.text((shared_x, 13), shared, font=body_font, fill=(0, 0, 0))
        if np.linalg.norm(reference["push_force"][time_index, :2]) > 1e-9:
            draw.ellipse((shared_x - 14, 13, shared_x - 4, 23), fill=(214, 94, 0))
        canvas.save(output_dir / f"frame_{i:06d}.png")
    encode_video(output_dir, output_root / "pd_vs_se3_wbc_comparison.mp4", fps=configs["robot"]["render_fps"])
    make_gif(output_dir, output_root / "pd_vs_se3_wbc_comparison.gif", fps=12, max_width=960)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Writable staging/output directory for videos and frames.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "results" / "data")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs = load_configs(ROOT)

    _render_trial_video(
        configs,
        data_root / "demo_geometric_push.npz",
        output_root,
        "geometric_push_recovery",
        "SE(3) WBC",
        width=int(configs["robot"].get("render_width", 1920)),
        height=int(configs["robot"].get("render_height", 1080)),
    )
    _render_trial_video(
        configs,
        data_root / "perturbed_standing_se3_wbc.npz",
        output_root,
        "perturbed_standing_recovery",
        "SE(3) WBC — perturbed standing",
        width=960,
        height=540,
    )
    _render_comparison(configs, data_root, output_root)
    print(f"Rendered saved trajectories to {output_root}")


if __name__ == "__main__":
    main()
