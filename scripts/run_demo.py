"""Run a compact canonical push and create a portfolio video."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# Select MuJoCo's headless EGL backend before importing the model wrapper.
# This keeps the demo command reproducible on the Linux SSH worker without a
# DISPLAY while still allowing an explicit MUJOCO_GL choice from the caller.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from common import ROOT, load_configs, make_push, output_dirs, run_trial, save_run
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video, make_gif


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); push = make_push(configs)
    model, run = run_trial("se3_wbc", configs, push=push, classify=True)
    save_run(run, dirs["data"] / "demo_geometric_push.npz", {"controller": "se3_wbc", "push": push.__dict__, "config": configs})
    frame_dir = dirs["frames"] / "demo"
    if frame_dir.exists(): shutil.rmtree(frame_dir)
    stride = max(1, int(round(1.0 / (configs["robot"]["control_timestep"] * configs["robot"]["render_fps"]))))
    arrays = run.log.arrays()
    overlay_data = [
        {
            "time_s": arrays["time_s"][i],
            "controller": "SE(3) WBC",
            "push_magnitude_N": push.magnitude_N,
            "push_direction_deg": push.direction_deg,
            "status": arrays["qp_status"][i],
            "com_world": arrays["com_world"][i],
            "push_force": arrays["push_force"][i],
            "contact_left": arrays["contact_left"][i],
            "contact_right": arrays["contact_right"][i],
        }
        for i in range(len(run.qpos_history))
    ]
    target_width = int(configs["robot"].get("render_width", 1920))
    target_height = int(configs["robot"].get("render_height", 1080))
    width, height = target_width, target_height
    fallback_reason = ""
    # MuJoCo's default GLFW backend can abort the process (before Python can
    # catch an exception) when a Linux worker has no DISPLAY. Do not probe a
    # large framebuffer in that known-unsafe configuration; EGL users can opt
    # in explicitly via MUJOCO_GL=egl.
    headless_without_backend = sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL")
    if headless_without_backend:
        fallback_reason = "Linux worker has no DISPLAY and MUJOCO_GL is unset; skipped unsafe GLFW target probe"
        width, height = 960, 540
    else:
        try:
            render_trial_frames(model, run.qpos_history, frame_dir, width=width, height=height, stride=stride, overlay_data=overlay_data)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if frame_dir.exists(): shutil.rmtree(frame_dir)
            width, height = 960, 540
    if not list(frame_dir.glob("frame_*.png")):
        render_trial_frames(model, run.qpos_history, frame_dir, width=width, height=height, stride=stride, overlay_data=overlay_data)
    encode_video(frame_dir, dirs["videos"] / "geometric_push_recovery.mp4", fps=configs["robot"]["render_fps"])
    make_gif(frame_dir, dirs["videos"] / "geometric_push_recovery.gif", fps=12)
    (dirs["logs"] / "video_render.txt").write_text(
        "timestamp_utc=" + datetime.now(timezone.utc).isoformat() + "\n"
        f"requested_resolution={target_width}x{target_height}\n"
        f"actual_resolution={width}x{height}\n"
        f"fallback_reason={fallback_reason or 'none'}\n"
        "overlay=time,controller,push,status,CoM-marker,contact-indicator\n",
        encoding="utf-8",
    )
    print(run.recovery)


if __name__ == "__main__":
    main()
