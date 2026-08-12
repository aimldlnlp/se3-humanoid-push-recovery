"""Render the corrected direct-perturbation standing response."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import ROOT, load_configs, output_dirs, randomized_initial_state, run_trial
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video, make_gif


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    cfg = configs["experiments"]["perturbed_standing"]
    initial_qpos, initial_qvel, _ = randomized_initial_state(configs, int(cfg["seed"]), randomization=cfg)
    model, run = run_trial(
        "se3_wbc", configs, duration=float(cfg["duration_s"]), seed=int(cfg["seed"]), classify=True,
        initial_qpos=initial_qpos, initial_qvel=initial_qvel,
    )
    frame_dir = dirs["frames"] / "perturbed_standing"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    arrays = run.log.arrays()
    overlay_data = [
        {
            "time_s": arrays["time_s"][i],
            "controller": "SE(3) WBC — perturbed standing",
            "push_magnitude_N": 0.0,
            "push_direction_deg": 0.0,
            "status": arrays["qp_status"][i],
            "com_world": arrays["com_world"][i],
            "feet_xy": arrays["foot_xy_world"][i],
            "push_force": arrays["push_force"][i],
            "contact_left": arrays["contact_left"][i],
            "contact_right": arrays["contact_right"][i],
        }
        for i in range(len(run.qpos_history))
    ]
    stride = max(1, int(round(1.0 / (configs["robot"]["control_timestep"] * configs["robot"]["render_fps"]))))
    paths = render_trial_frames(model, run.qpos_history, frame_dir, width=960, height=540, stride=stride, overlay_data=overlay_data)
    encode_video(frame_dir, dirs["videos"] / "perturbed_standing_recovery.mp4", fps=configs["robot"]["render_fps"])
    make_gif(frame_dir, dirs["videos"] / "perturbed_standing_recovery.gif", fps=12)
    (dirs["logs"] / "perturbed_video_render.txt").write_text(
        f"frames={len(paths)}\nrequested_fps={configs['robot']['render_fps']}\n"
        f"initial_torso_error_rad={arrays['torso_rotation_error_rad'][0]:.9f}\n"
        f"initial_torso_angular_velocity_rad_s={arrays['torso_angular_velocity_norm'][0]:.9f}\n"
        f"recovery_success={run.recovery.success if run.recovery else 'unclassified'}\n",
        encoding="utf-8",
    )
    print(run.recovery)


if __name__ == "__main__":
    main()
