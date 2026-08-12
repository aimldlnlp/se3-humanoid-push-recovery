"""One-factor-at-a-time robustness experiment."""

from __future__ import annotations

import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, flatten_result, load_configs, make_push, output_dirs, randomized_initial_state, run_trial, write_csv, write_execution_manifest


def _run_one(task):
    factor, value, seed, configs = task
    local = {**configs, "robot": dict(configs["robot"]), "controller": dict(configs["controller"])}
    if factor == "friction":
        local["controller"]["friction_coefficient"] = value
    rng = np.random.default_rng(seed)
    base_push = local["experiments"]["push"]
    random_cfg = local["experiments"].get("robustness_randomization", {})
    magnitude = float(base_push["magnitude_N"]) * (1.0 + rng.uniform(-float(random_cfg.get("push_magnitude_jitter_fraction", 0.0)), float(random_cfg.get("push_magnitude_jitter_fraction", 0.0))))
    direction = float(base_push["direction_deg"]) + rng.normal(0.0, float(random_cfg.get("push_direction_jitter_deg", 0.0)))
    duration = value if factor == "push_duration_s" else float(base_push["duration_s"]) + rng.uniform(-float(random_cfg.get("push_duration_jitter_s", 0.0)), float(random_cfg.get("push_duration_jitter_s", 0.0)))
    duration = max(0.03, duration)
    push = make_push(local, magnitude=magnitude, direction_deg=direction, duration=duration)
    mass_scale = value if factor == "mass_scale" else 1.0
    friction = value if factor == "friction" else None
    initial_qpos, initial_qvel, perturbation = randomized_initial_state(local, seed, mass_scale=mass_scale, friction_coefficient=friction)
    _, run = run_trial(
        "se3_wbc", local, push=push,
        mass_scale=mass_scale, friction_coefficient=friction,
        seed=seed, classify=True, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
    )
    return flatten_result(run, "se3_wbc", push, f"{factor}_{value}_{seed}", seed, {"factor": factor, "factor_value": value, **{f"initial_{key}": val for key, val in perturbation.items()}})


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); rob = configs["experiments"]["robustness"]
    tasks = [
        (factor, value, seed, configs)
        for factor, values in (("friction", rob["friction"]), ("mass_scale", rob["mass_scale"]), ("push_duration_s", rob["push_duration_s"]))
        for value in values
        for seed in rob["seeds"]
    ]
    rows = []
    workers = max(1, int(os.environ.get("SE3_ROBUSTNESS_WORKERS", "4")))
    if workers == 1:
        iterator = map(_run_one, tasks)
        for row in iterator:
            rows.append(row); print(row, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_run_one, tasks):
                rows.append(row); print(row, flush=True)
    rows.sort(key=lambda row: (row["factor"], float(row["factor_value"]), int(row["seed"])))
    write_csv(rows, dirs["data"] / "robustness.csv")
    write_execution_manifest(dirs["logs"] / "robustness_manifest.json", configs, seed=int(configs["controller"].get("seed", 0)), extra={"experiment": "robustness", "trial_count": len(rows)})


if __name__ == "__main__":
    main()
