"""Standing regulation from a reproducible tilted/offset initial state."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_configs, output_dirs, randomized_initial_state, run_trial, save_run
from se3_whole_body_control.visualization.plots import plot_trial


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    cfg = configs["experiments"]["perturbed_standing"]
    initial_qpos, initial_qvel, perturbation = randomized_initial_state(configs, int(cfg["seed"]), randomization=cfg)
    for controller in ("pd", "se3_wbc"):
        _, run = run_trial(
            controller, configs, duration=float(cfg["duration_s"]), seed=int(cfg["seed"]), classify=True,
            initial_qpos=initial_qpos, initial_qvel=initial_qvel,
        )
        stem = f"perturbed_standing_{controller}"
        save_run(run, dirs["data"] / f"{stem}.npz", {"controller": controller, "perturbation": perturbation, "config": configs})
        plot_trial(run.log, dirs["png"], stem)
        for pdf in dirs["png"].glob(f"{stem}_*.pdf"):
            pdf.replace(dirs["pdf"] / pdf.name)
        print(controller, run.recovery)


if __name__ == "__main__":
    main()
