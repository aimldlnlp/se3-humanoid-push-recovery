"""Deterministic push-recovery sweep for PD and SE(3) WBC."""

from __future__ import annotations

import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, flatten_result, load_configs, make_push, output_dirs, run_trial, write_csv, write_execution_manifest
from se3_whole_body_control.visualization.plots import plot_comparison, plot_recovery_heatmap


def _run_one(task):
    """Process-isolated deterministic trial worker."""
    controller, configs, magnitude, direction, duration, start = task
    push = make_push(configs, magnitude=magnitude, direction_deg=direction, duration=duration, start=start)
    _, run = run_trial(controller, configs, push=push, duration=configs["experiments"]["sweep"]["trial_duration_s"], classify=True)
    return flatten_result(run, controller, push, f"{controller}_{magnitude:g}_{direction:g}")


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); sweep = configs["experiments"]["sweep"]
    tasks = [
        (controller, configs, magnitude, direction, sweep["duration_s"], sweep["start_time_s"])
        for controller in ("pd", "se3_wbc")
        for magnitude in sweep["magnitudes_N"]
        for direction in sweep["directions_deg"]
    ]
    rows = []
    workers = max(1, int(os.environ.get("SE3_SWEEP_WORKERS", "4")))
    if workers == 1:
        for row in map(_run_one, tasks):
            rows.append(row)
            print(row, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_run_one, tasks):
                rows.append(row)
                print(row, flush=True)
    rows.sort(key=lambda row: (row["controller"], float(row["push_magnitude_N"]), float(row["push_direction_deg"])))
    write_csv(rows, dirs["data"] / "push_sweep.csv")
    write_execution_manifest(dirs["logs"] / "push_sweep_manifest.json", configs, seed=int(configs["controller"].get("seed", 0)), extra={"experiment": "push_sweep", "trial_count": len(rows)})
    for controller in ("pd", "se3_wbc"):
        paths = plot_recovery_heatmap(rows, dirs["png"], controller=controller, name=f"recovery_heatmap_{controller}")
        for p in paths:
            if p.suffix == ".pdf": p.replace(dirs["pdf"] / p.name)
    shutil.copyfile(dirs["png"] / "recovery_heatmap_se3_wbc.png", dirs["png"] / "recovery_heatmap.png")
    shutil.copyfile(dirs["pdf"] / "recovery_heatmap_se3_wbc.pdf", dirs["pdf"] / "recovery_heatmap.pdf")
    paths = plot_comparison(rows, dirs["png"])
    for p in paths:
        if p.suffix == ".pdf": p.replace(dirs["pdf"] / p.name)


if __name__ == "__main__":
    main()
