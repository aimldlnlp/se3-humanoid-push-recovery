"""Scientific plots saved as both PNG and PDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import apply_style


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _save(fig, output_dir: str | Path, name: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_trial(log, output_dir: str | Path, prefix: str = "trial") -> list[Path]:
    apply_style()
    a = log.arrays()
    t = a["time_s"]
    paths: list[Path] = []
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, a["torso_error"][:, 3:], label=["x", "y", "z"])
    ax[0].set_ylabel("rotation error [rad]")
    ax[0].legend(ncol=3)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, a["torso_error"][:, :3], label=["x", "y", "z"])
    ax[1].set_ylabel("translation error [m]")
    ax[1].set_xlabel("time [s]")
    ax[1].legend(ncol=3)
    ax[1].grid(alpha=0.3)
    paths.extend(_save(fig, output_dir, f"{prefix}_se3_error"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, a["com_world"], label=["x", "y", "z"])
    ax.set_xlabel("time [s]"); ax.set_ylabel("CoM position [m]"); ax.grid(alpha=0.3); ax.legend(ncol=3)
    paths.extend(_save(fig, output_dir, f"{prefix}_com"))

    fig, ax = plt.subplots(figsize=(8, 4))
    wrench = a["contact_wrench"]
    if wrench.ndim == 2 and wrench.shape[1] >= 12:
        ax.plot(t, wrench[:, 2], label="left Fz")
        ax.plot(t, wrench[:, 8], label="right Fz")
    ax.set_xlabel("time [s]"); ax.set_ylabel("normal contact force [N]"); ax.grid(alpha=0.3); ax.legend()
    paths.extend(_save(fig, output_dir, f"{prefix}_contact_forces"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, a["torque_abs_max_Nm"], color="tab:red")
    ax.set_xlabel("time [s]"); ax.set_ylabel("max actuator command [N or N m]"); ax.grid(alpha=0.3)
    paths.extend(_save(fig, output_dir, f"{prefix}_torques"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, a["push_force"][:, 0], label="Fx")
    ax.plot(t, a["push_force"][:, 1], label="Fy")
    ax.set_xlabel("time [s]"); ax.set_ylabel("push force [N]"); ax.grid(alpha=0.3); ax.legend()
    paths.extend(_save(fig, output_dir, f"{prefix}_push_force"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, a["qp_solve_time_s"] * 1000.0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("QP solve time [ms]"); ax.grid(alpha=0.3)
    paths.extend(_save(fig, output_dir, f"{prefix}_qp_solve_time"))
    return paths


def plot_comparison(rows: list[dict], output_dir: str | Path, name: str = "controller_comparison") -> list[Path]:
    apply_style()
    controllers = sorted({str(r["controller"]) for r in rows})
    rates = [np.mean([_as_bool(r["success"]) for r in rows if r["controller"] == c]) for c in controllers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(controllers, rates, color=["tab:gray", "tab:blue"][:len(controllers)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("recovery rate"); ax.set_title("Push-recovery comparison"); ax.grid(axis="y", alpha=0.3)
    return list(_save(fig, output_dir, name))


def plot_recovery_heatmap(rows: list[dict], output_dir: str | Path, controller: str = "se3_wbc", name: str = "recovery_heatmap") -> list[Path]:
    apply_style()
    mags = sorted({float(r["push_magnitude_N"]) for r in rows})
    dirs = sorted({float(r["push_direction_deg"]) for r in rows})
    grid = np.full((len(mags), len(dirs)), np.nan)
    for r in rows:
        if r["controller"] == controller:
            grid[mags.index(float(r["push_magnitude_N"])), dirs.index(float(r["push_direction_deg"]))] = float(_as_bool(r["success"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(len(dirs)), [f"{d:.0f}" for d in dirs], rotation=90)
    ax.set_yticks(range(len(mags)), [f"{m:.0f}" for m in mags])
    ax.set_xlabel("push direction [deg]"); ax.set_ylabel("push magnitude [N]"); ax.set_title(f"Recovery map: {controller}")
    fig.colorbar(im, ax=ax, label="success")
    return list(_save(fig, output_dir, name))


def plot_gpu_benchmark(rows: list[dict], output_dir: str | Path, name: str = "gpu_benchmark") -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = [r for r in rows if str(r.get("status", "")) == "ok"]
    if valid:
        ax.plot([r["batch_size"] for r in valid], [r["simulations_per_second"] for r in valid], marker="o", label="throughput")
    ax.set_xlabel("batch size"); ax.set_ylabel("simulations / second"); ax.grid(alpha=0.3)
    if valid:
        ax.legend()
    return list(_save(fig, output_dir, name))
