"""Write provenance manifests for already-computed aggregate experiment data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common import load_configs, output_dirs, write_execution_manifest


def main() -> None:
    configs = load_configs(ROOT)
    dirs = output_dirs()
    aggregate_runs = (
        ("push_sweep.csv", "push_sweep", "push_sweep_manifest.json"),
        ("robustness.csv", "robustness", "robustness_manifest.json"),
    )
    for data_name, experiment, manifest_name in aggregate_runs:
        data_path = dirs["data"] / data_name
        if not data_path.exists():
            continue
        count = max(0, sum(1 for _ in data_path.open(encoding="utf-8")) - 1)
        write_execution_manifest(
            dirs["logs"] / manifest_name,
            configs,
            seed=int(configs["controller"].get("seed", 0)),
            extra={"experiment": experiment, "trial_count": count, "data_file": data_name},
        )
        print(dirs["logs"] / manifest_name)


if __name__ == "__main__":
    main()
