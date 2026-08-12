"""Run the canonical 120 N push for both controllers."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_configs, make_push, output_dirs, run_trial, save_run
from se3_whole_body_control.visualization.plots import plot_trial


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    push = make_push(configs)
    for controller in ("pd", "se3_wbc"):
        model, run = run_trial(controller, configs, push=push, classify=True)
        stem = f"single_push_{controller}"
        save_run(run, dirs["data"] / f"{stem}.npz", {"controller": controller, "push": push.__dict__, "config": configs})
        plot_trial(run.log, dirs["png"], stem)
        for pdf in dirs["png"].glob(f"{stem}_*.pdf"):
            pdf.replace(dirs["pdf"] / pdf.name)
        if controller == "se3_wbc":
            shutil.copyfile(dirs["png"] / f"{stem}_se3_error.png", dirs["png"] / "single_push_se3_error.png")
            shutil.copyfile(dirs["pdf"] / f"{stem}_se3_error.pdf", dirs["pdf"] / "single_push_se3_error.pdf")
        print(controller, run.recovery)


if __name__ == "__main__":
    main()
