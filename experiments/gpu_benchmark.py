"""Probe optional JAX/MJX acceleration without weakening the CPU reference."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, output_dirs
from se3_whole_body_control.visualization.plots import plot_gpu_benchmark


def main() -> None:
    dirs = output_dirs(); rows = []
    try:
        import jax
        devices = jax.devices()
        gpu = [d for d in devices if d.platform == "gpu"]
        if not gpu:
            raise RuntimeError(f"no GPU visible: {devices}")
        import mujoco.mjx as mjx  # noqa: F401
        for batch in (64, 128, 256, 512, 1024):
            t0 = time.perf_counter()
            # MJX support is model/controller dependent; record the probe and
            # keep a place for a validated batched kernel rather than fake data.
            elapsed = time.perf_counter() - t0
            rows.append({"batch_size": batch, "status": "unavailable", "reason": "validated batched WBC kernel not yet enabled", "elapsed_s": elapsed, "simulations_per_second": 0.0})
    except Exception as exc:
        rows.append({"batch_size": 0, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "elapsed_s": 0.0, "simulations_per_second": 0.0})
    with (dirs["data"] / "gpu_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "status", "reason", "elapsed_s", "simulations_per_second"]); writer.writeheader(); writer.writerows(rows)
    paths = plot_gpu_benchmark(rows, dirs["png"])
    for p in paths:
        if p.suffix == ".pdf": p.replace(dirs["pdf"] / p.name)
    print(rows)


if __name__ == "__main__":
    main()
