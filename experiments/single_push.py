"""Run the configured canonical push for both controllers."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_configs, make_push, output_dirs, run_trial, save_run
from se3_whole_body_control.visualization.plots import plot_actual_grf, plot_com_support_polygon, plot_flagship, plot_trial


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    push = make_push(configs)
    wbc_run = None
    for controller in ("pd", "se3_wbc"):
        model, run = run_trial(controller, configs, push=push, classify=True)
        stem = f"single_push_{controller}"
        save_run(run, dirs["data"] / f"{stem}.npz", {"controller": controller, "push": push.__dict__, "config": configs})
        plot_trial(run.log, dirs["png"], stem)
        for pdf in dirs["png"].glob(f"{stem}_*.pdf"):
            pdf.replace(dirs["pdf"] / pdf.name)
        if controller == "se3_wbc":
            wbc_run = run
            shutil.copyfile(dirs["png"] / f"{stem}_se3_error.png", dirs["png"] / "single_push_se3_error.png")
            shutil.copyfile(dirs["pdf"] / f"{stem}_se3_error.pdf", dirs["pdf"] / "single_push_se3_error.pdf")
        print(controller, run.recovery)
    if wbc_run is not None:
        for plotter, name in ((plot_flagship, "canonical_response"), (plot_actual_grf, "actual_ground_reaction_forces"), (plot_com_support_polygon, "com_support_polygon")):
            paths = plotter(wbc_run.log, dirs["png"], name)
            for path in paths:
                if path.suffix == ".pdf":
                    path.replace(dirs["pdf"] / path.name)


if __name__ == "__main__":
    main()
