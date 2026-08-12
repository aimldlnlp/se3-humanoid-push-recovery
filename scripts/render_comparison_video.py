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
            "com_world": a["com_world"][i], "feet_xy": a["foot_xy_world"][i], "push_force": a["push_force"][i],
            "contact_left": a["contact_left"][i], "contact_right": a["contact_right"][i],
        }
        for i in range(len(run.qpos_history))
    ]


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); push = make_push(configs)
    rendered = []
    stride = max(1, int(round(1.0 / (configs["robot"]["control_timestep"] * configs["robot"]["render_fps"]))))
    for controller, label in (("pd", "PD"), ("se3_wbc", "SE(3) WBC")):
        model, run = run_trial(controller, configs, push=push, classify=True)
        frame_dir = dirs["frames"] / f"comparison_{controller}"
        if frame_dir.exists(): shutil.rmtree(frame_dir)
        paths = render_trial_frames(model, run.qpos_history, frame_dir, width=960, height=540, stride=stride, overlay_data=overlay_data(run, label, push))
        rendered.append(paths)
    output_dir = dirs["frames"] / "comparison_combined"
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(paths) for paths in rendered)
    font = ImageFont.load_default()
    for i in range(count):
        left = Image.open(rendered[0][i]).convert("RGB"); right = Image.open(rendered[1][i]).convert("RGB")
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
        canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 26), fill=(0, 0, 0))
        draw.text((12, 7), "same initial state and same 120 N push  |  PD", font=font, fill="white")
        draw.text((left.width + 12, 7), "same initial state and same 120 N push  |  SE(3) WBC", font=font, fill="white")
        canvas.save(output_dir / f"frame_{i:06d}.png")
    encode_video(output_dir, dirs["videos"] / "pd_vs_se3_wbc_comparison.mp4", fps=configs["robot"]["render_fps"])
    make_gif(output_dir, dirs["videos"] / "pd_vs_se3_wbc_comparison.gif", fps=12)
    print(f"frames={count}")


if __name__ == "__main__":
    main()
