"""Regenerate figures from saved trial/sweep data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common import ROOT, output_dirs, read_csv
from se3_whole_body_control.evaluation.metrics import TrialLog
from se3_whole_body_control.visualization.plots import plot_actual_grf, plot_com_support_polygon, plot_comparison, plot_flagship, plot_recovery_heatmap, plot_trial


def load_log(path: Path) -> TrialLog:
    import numpy as np
    z = np.load(path, allow_pickle=False)
    log = TrialLog.empty()
    for field in TrialLog.__dataclass_fields__:
        if field in z:
            setattr(log, field, z[field].tolist())
    return log


def main() -> None:
    dirs = output_dirs()
    for path in sorted(dirs["data"].glob("*.npz")):
        log = load_log(path)
        plot_trial(log, dirs["png"], path.stem)
        if "single_push_se3_wbc" in path.stem:
            for plotter, name in ((plot_flagship, "canonical_response"), (plot_actual_grf, "actual_ground_reaction_forces"), (plot_com_support_polygon, "com_support_polygon")):
                plotter(log, dirs["png"], name)
    sweep = dirs["data"] / "push_sweep.csv"
    if sweep.exists():
        rows = read_csv(sweep)
        for controller in ("pd", "se3_wbc"):
            plot_recovery_heatmap(rows, dirs["png"], controller, f"recovery_heatmap_{controller}")
        plot_comparison(rows, dirs["png"])
    for path in dirs["png"].glob("*.pdf"):
        path.replace(dirs["pdf"] / path.name)


if __name__ == "__main__":
    main()
