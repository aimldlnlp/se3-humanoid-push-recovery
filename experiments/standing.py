"""Run baseline and geometric no-push standing experiments."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_configs, output_dirs, run_trial, save_run
from se3_whole_body_control.visualization.plots import plot_trial


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    duration = float(configs["experiments"]["standing_duration_s"])
    for controller in ("pd", "se3_wbc"):
        model, run = run_trial(controller, configs, duration=duration, classify=False)
        stem = "baseline_standing" if controller == "pd" else "geometric_standing"
        save_run(run, dirs["data"] / f"{stem}.npz", {"controller": controller, "duration_s": duration, "config": configs})
        plot_trial(run.log, dirs["png"], stem)
        # Plot helper saves PDFs alongside PNG output; keep canonical copies in pdf.
        for pdf in dirs["png"].glob(f"{stem}_*.pdf"):
            pdf.replace(dirs["pdf"] / pdf.name)
        # Stable, human-facing artifact names required by the README/plan.
        shutil.copyfile(dirs["png"] / f"{stem}_se3_error.png", dirs["png"] / f"{stem}.png")
        shutil.copyfile(dirs["pdf"] / f"{stem}_se3_error.pdf", dirs["pdf"] / f"{stem}.pdf")
        print(f"{controller}: {run.log.arrays()['time_s'][-1]:.3f}s, samples={len(run.log.time_s)}")


if __name__ == "__main__":
    main()
