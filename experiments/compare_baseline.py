"""Compare baseline PD and geometric WBC on a small deterministic set."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, flatten_result, load_configs, make_push, output_dirs, run_trial, write_csv
from se3_whole_body_control.visualization.plots import plot_comparison


def main() -> None:
    configs = load_configs(ROOT); dirs = output_dirs(); push = make_push(configs)
    rows = []
    for controller in ("pd", "se3_wbc"):
        _, run = run_trial(controller, configs, push=push, classify=True)
        rows.append(flatten_result(run, controller, push, f"canonical_{controller}"))
    write_csv(rows, dirs["data"] / "controller_comparison.csv")
    paths = plot_comparison(rows, dirs["png"])
    for p in paths:
        if p.suffix == ".pdf": p.replace(dirs["pdf"] / p.name)
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()
