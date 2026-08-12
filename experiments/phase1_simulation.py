"""Phase 1 gate: load, step, contact-check, push, and render the model."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from common import ROOT, load_configs, make_model, output_dirs
from se3_whole_body_control.visualization.renderer import render_trial_frames


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); model = make_model(configs)
    initial_contacts = model.contact_flags()
    initial_qpos = model.data.qpos.copy()
    frame_dir = dirs["frames"] / "phase1_initial"
    if frame_dir.exists(): shutil.rmtree(frame_dir)
    frames = render_trial_frames(model, [initial_qpos], frame_dir, width=960, height=540)
    shutil.copyfile(frames[0], dirs["png"] / "humanoid_initial_state.png")
    for _ in range(150):
        model.step(np.zeros(model.nu))
    torso_before = model.body_pose("torso")[0, 3]
    model.set_external_force("torso", [120.0, 0.0, 0.0])
    for _ in range(75):
        model.step(np.zeros(model.nu))
    torso_after = model.body_pose("torso")[0, 3]
    report = {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "body_ids": {k: int(v) for k, v in model.body_ids.items()},
        "initial_contacts": [bool(v) for v in initial_contacts],
        "steps": 225, "torso_x_before_push_m": torso_before,
        "torso_x_after_push_m": float(torso_after),
        "push_changed_motion": bool(abs(torso_after - torso_before) > 1e-6),
    }
    (dirs["data"] / "phase1_simulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import numpy as np
    main()
