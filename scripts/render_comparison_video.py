"""Render synchronized side-by-side PD and SE(3) WBC canonical trials."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import load_configs, make_push, output_dirs, run_trial
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video, make_gif


def overlay_data(run, controller, push):
    a = run.log.arrays()
    return [
        {
            "time_s": a["time_s"][i], "controller": controller, "push_magnitude_N": push.magnitude_N,
            "push_direction_deg": push.direction_deg, "status": a["qp_status"][i],
            "com_world": a["com_world"][i], "feet_xy": a["foot_xy_world"][i], "push_point_world": a["torso_position"][i], "push_force": a["push_force"][i],
            "contact_left": a["contact_left"][i], "contact_right": a["contact_right"][i],
        }
        for i in range(len(run.qpos_history))
    ]


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); push = make_push(configs)
    rendered = []
    reference_arrays = None
    stride = max(1, int(round(1.0 / (configs["robot"]["control_timestep"] * configs["robot"]["render_fps"]))))
    for controller, label in (("pd", "PD"), ("se3_wbc", "SE(3) WBC")):
        model, run = run_trial(controller, configs, push=push, classify=True)
        if reference_arrays is None:
            reference_arrays = run.log.arrays()
        frame_dir = dirs["frames"] / f"comparison_{controller}"
        if frame_dir.exists(): shutil.rmtree(frame_dir)
        paths = render_trial_frames(model, run.qpos_history, frame_dir, width=960, height=540, stride=stride, overlay_data=overlay_data(run, label, push))
        rendered.append(paths)
    output_dir = dirs["frames"] / "comparison_combined"
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(paths) for paths in rendered)
    font_path = "DejaVuSans.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 22)
        body_font = ImageFont.truetype(font_path, 16)
    except OSError:
        title_font = body_font = ImageFont.load_default()
    for i in range(count):
        left = Image.open(rendered[0][i]).convert("RGB"); right = Image.open(rendered[1][i]).convert("RGB")
        header_height = 42
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + header_height), "white")
        canvas.paste(left, (0, header_height)); canvas.paste(right, (left.width, header_height))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, header_height), fill=(255, 255, 255), outline=(31, 41, 51), width=1)
        draw.line((left.width, header_height, left.width, canvas.height), fill=(148, 163, 184), width=1)
        time_s = float(reference_arrays["time_s"][min(i * stride, len(reference_arrays["time_s"]) - 1)])
        force = float(np.linalg.norm(reference_arrays["push_force"][min(i * stride, len(reference_arrays["push_force"]) - 1), :2]))
        shared = f"same 120 N push  |  t = {time_s:.2f} s"
        draw.text((16, 10), "PD", font=title_font, fill=(91, 103, 112))
        draw.text((left.width + 16, 10), "SE(3) WBC", font=title_font, fill=(0, 114, 178))
        bbox = draw.textbbox((0, 0), shared, font=body_font)
        shared_width = bbox[2] - bbox[0]
        shared_x = left.width - shared_width - 18
        draw.text((shared_x, 13), shared, font=body_font, fill=(31, 41, 51))
        if force > 1e-9:
            draw.ellipse((shared_x - 14, 13, shared_x - 4, 23), fill=(214, 94, 0))
        canvas.save(output_dir / f"frame_{i:06d}.png")
    encode_video(output_dir, dirs["videos"] / "pd_vs_se3_wbc_comparison.mp4", fps=configs["robot"]["render_fps"])
    make_gif(output_dir, dirs["videos"] / "pd_vs_se3_wbc_comparison.gif", fps=12, max_width=960)
    print(f"frames={count}")


if __name__ == "__main__":
    main()
