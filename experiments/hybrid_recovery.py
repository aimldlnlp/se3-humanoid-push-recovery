"""Compare fixed-foot and one-step hybrid recovery on a reproducible grid.

The fixed-foot WBC is rerun in the same invocation as the hybrid controller so
the comparison shares the exact source, configuration, and worker environment.
The old frozen sweep remains untouched as a separate baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import flatten_result, load_configs, make_push, run_trial, save_run, write_csv, write_execution_manifest
from se3_whole_body_control.visualization.plots import plot_hybrid_recovery_basin


def _summary_extra(run, controller_name: str) -> dict:
    arrays = run.log.arrays()
    modes = arrays.get("control_mode", np.asarray([], dtype=str)).astype(str)
    support_margin = np.asarray(arrays.get("support_margin_m", np.asarray([], dtype=float)), dtype=float)
    finite_margin = support_margin[np.isfinite(support_margin)]
    return {
        "controller_variant": controller_name,
        "step_triggered": bool(run.metadata.get("controller_summary", {}).get("step_triggered", False)),
        "step_count": int(run.metadata.get("controller_summary", {}).get("step_count", 0)),
        "swing_foot": run.metadata.get("controller_summary", {}).get("swing_foot", "") or "",
        "final_mode": run.metadata.get("controller_summary", {}).get("final_mode", "fixed"),
        "single_support_fraction": float(np.mean(modes == "single_support")) if len(modes) else 0.0,
        "min_double_support_margin_m": float(np.min(finite_margin)) if len(finite_margin) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "hybrid_recovery")
    parser.add_argument("--magnitudes", type=float, nargs="+", default=None)
    parser.add_argument("--directions", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--save-all-trials", action="store_true", help="Save every trajectory; default saves canonical examples only.")
    args = parser.parse_args()

    configs = load_configs(ROOT)
    sweep = configs["experiments"]["sweep"]
    magnitudes = [float(value) for value in (args.magnitudes or sweep["magnitudes_N"])]
    directions = [float(value) for value in (args.directions or sweep["directions_deg"])]
    output_root = args.output_root.resolve()
    data_root = output_root / "data"
    trial_root = data_root / "trials"
    logs_root = output_root / "logs"
    data_root.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    run_id = f"hybrid-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    write_execution_manifest(
        logs_root / "manifest.json",
        configs,
        seed=args.seed,
        extra={
            "experiment": "hybrid_recovery",
            "run_id": run_id,
            "controllers": ["fixed_foot_wbc", "hybrid_se3_wbc"],
            "magnitudes_N": magnitudes,
            "directions_deg": directions,
            "duration_s": args.duration or configs["robot"]["duration_s"],
            "hybrid_config": configs["experiments"].get("hybrid_recovery", {}),
            "baseline_source_note": "The prior frozen G1 sweep remains in results/data/push_sweep.csv and is not overwritten.",
        },
    )

    rows: list[dict] = []
    canonical_saved = set()
    for magnitude in magnitudes:
        for direction in directions:
            push = make_push(configs, magnitude=magnitude, direction_deg=direction)
            for controller_name, label in (("se3_wbc", "fixed_foot_wbc"), ("hybrid_wbc", "hybrid_se3_wbc")):
                trial_id = f"{label}_{magnitude:.0f}N_{direction:.0f}deg_seed{args.seed}"
                model, run = run_trial(
                    controller_name,
                    configs,
                    push=push,
                    duration=args.duration,
                    seed=args.seed,
                    classify=True,
                )
                extra = _summary_extra(run, label)
                row = flatten_result(run, label, push, trial_id, seed=args.seed, extra=extra)
                rows.append(row)
                is_canonical = abs(magnitude - float(configs["experiments"]["push"]["magnitude_N"])) < 1e-9 and abs(direction - float(configs["experiments"]["push"]["direction_deg"])) < 1e-9
                if args.save_all_trials or is_canonical:
                    save_run(
                        run,
                        trial_root / f"{trial_id}.npz",
                        {
                            "run_id": run_id,
                            "trial_id": trial_id,
                            "controller": label,
                            "config": configs,
                            "push": push.__dict__,
                            "hybrid_summary": run.metadata.get("controller_summary", {}),
                        },
                    )
                    canonical_saved.add(trial_id)

    rows.sort(key=lambda row: (str(row["controller"]), float(row["push_magnitude_N"]), float(row["push_direction_deg"])))
    write_csv(rows, data_root / "hybrid_recovery_sweep.csv")
    plot_hybrid_recovery_basin(rows, output_root / "figures" / "png")
    summary = {
        "run_id": run_id,
        "trial_count": len(rows),
        "saved_trajectory_count": len(canonical_saved),
        "controllers": sorted({str(row["controller"]) for row in rows}),
        "success_counts": {
            controller: int(sum(bool(row["success"]) for row in rows if str(row["controller"]) == controller))
            for controller in sorted({str(row["controller"]) for row in rows})
        },
    }
    (logs_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
