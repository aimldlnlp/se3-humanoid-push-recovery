"""Deterministic pre-sweep push-range calibration for both controllers."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, flatten_result, load_configs, make_push, output_dirs, run_trial, write_csv, write_execution_manifest


CALIBRATION_MAGNITUDES_N = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
CALIBRATION_DIRECTIONS_DEG = tuple(float(direction) for direction in range(0, 360, 45))


def _run_one(task):
    controller, configs, magnitude, direction = task
    sweep = configs["experiments"]["sweep"]
    push = make_push(
        configs,
        magnitude=magnitude,
        direction_deg=direction,
        duration=float(sweep["duration_s"]),
        start=float(sweep["start_time_s"]),
    )
    _, run = run_trial(
        controller,
        configs,
        push=push,
        duration=float(sweep["trial_duration_s"]),
        classify=True,
    )
    return flatten_result(
        run,
        controller,
        push,
        f"calibration_{controller}_{magnitude:g}_{direction:g}",
    )


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    tasks = [
        (controller, configs, magnitude, direction)
        for controller in ("pd", "se3_wbc")
        for magnitude in CALIBRATION_MAGNITUDES_N
        for direction in CALIBRATION_DIRECTIONS_DEG
    ]
    workers = max(1, int(os.environ.get("SE3_CALIBRATION_WORKERS", "4")))
    if workers == 1:
        rows = list(map(_run_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, tasks))
    rows.sort(key=lambda row: (row["controller"], float(row["push_magnitude_N"]), float(row["push_direction_deg"])))
    write_csv(rows, dirs["data"] / "push_calibration.csv")
    write_execution_manifest(
        dirs["logs"] / "push_calibration_manifest.json",
        configs,
        seed=int(configs["experiments"].get("seed", 0)),
        extra={
            "experiment": "push_calibration",
            "trial_count": len(rows),
            "magnitudes_N": list(CALIBRATION_MAGNITUDES_N),
            "directions_deg": list(CALIBRATION_DIRECTIONS_DEG),
            "controllers": ["pd", "se3_wbc"],
        },
    )
    print(f"calibration trials: {len(rows)}")


if __name__ == "__main__":
    main()
