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
    wrench = a["actual_contact_wrench"]
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


def _shade_push(ax, time_s: np.ndarray, push_force: np.ndarray) -> None:
    magnitude = np.linalg.norm(push_force[:, :2], axis=1) if len(push_force) else np.zeros(0)
    active = magnitude > 1e-9
    if not np.any(active):
        return
    indices = np.flatnonzero(active)
    start, end = float(time_s[indices[0]]), float(time_s[indices[-1]])
    ax.axvspan(start, end, color="tab:orange", alpha=0.18, label="push interval")


def plot_flagship(log, output_dir: str | Path, name: str = "canonical_response") -> list[Path]:
    """Coherent multi-panel canonical response figure for the README."""
    apply_style()
    a = log.arrays()
    t = a["time_s"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    push = a["push_force"]
    axes[0, 0].plot(t, np.linalg.norm(push[:, :2], axis=1), color="tab:orange")
    axes[0, 0].set_ylabel("push [N]")
    axes[0, 1].plot(t, a["torso_rotation_error_rad"], color="tab:blue")
    axes[0, 1].set_ylabel("torso rotation [rad]")
    com = a["com_world"]
    axes[1, 0].plot(t, np.linalg.norm(com[:, :2] - com[0, :2], axis=1), color="tab:green")
    axes[1, 0].set_ylabel("CoM displacement [m]")
    actual = a["actual_contact_wrench"]
    axes[1, 1].plot(t, actual[:, 2], label="left Fz")
    axes[1, 1].plot(t, actual[:, 8], label="right Fz")
    axes[1, 1].set_ylabel("actual GRF Fz [N]")
    axes[1, 1].legend(loc="best")
    torque_limit = 180.0
    axes[2, 0].plot(t, np.max(np.abs(a["control"]), axis=1) / torque_limit, color="tab:red")
    axes[2, 0].set_ylabel("torque utilization")
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 1].plot(t, np.max(a["actual_friction_utilization"], axis=1), color="tab:purple", label="friction utilization")
    axes[2, 1].plot(t, np.max(a["foot_tangent_velocity"], axis=1), color="tab:brown", label="foot tangential speed [m/s]")
    axes[2, 1].set_ylabel("contact activity")
    axes[2, 1].legend(loc="best")
    axes[3, 0].plot(t, a["qp_solve_time_s"] * 1000.0, color="tab:gray")
    axes[3, 0].set_ylabel("QP solve [ms]")
    axes[3, 1].plot(t, a["dynamics_residual_norm"], label="dynamics")
    axes[3, 1].plot(t, a["contact_acceleration_residual_norm"], label="contact")
    axes[3, 1].set_ylabel("constraint residual")
    axes[3, 1].set_yscale("symlog", linthresh=1e-8)
    axes[3, 1].legend(loc="best")
    for row in axes:
        for ax in row:
            _shade_push(ax, t, push)
            ax.grid(alpha=0.25)
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 1].set_xlabel("time [s]")
    fig.suptitle("Canonical SE(3) push recovery: measured response and constraints", y=0.995)
    fig.tight_layout()
    return list(_save(fig, output_dir, name))


def plot_actual_grf(log, output_dir: str | Path, name: str = "actual_ground_reaction_forces") -> list[Path]:
    apply_style()
    a = log.arrays(); t = a["time_s"]; wrench = a["actual_contact_wrench"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, wrench[:, 0], label="left Fx")
    axes[0].plot(t, wrench[:, 1], label="left Fy")
    axes[0].plot(t, wrench[:, 2], label="left Fz")
    axes[0].set_ylabel("left GRF [N]"); axes[0].legend(ncol=3); axes[0].grid(alpha=0.25)
    axes[1].plot(t, wrench[:, 6], label="right Fx")
    axes[1].plot(t, wrench[:, 7], label="right Fy")
    axes[1].plot(t, wrench[:, 8], label="right Fz")
    axes[1].set_ylabel("right GRF [N]"); axes[1].set_xlabel("time [s]"); axes[1].legend(ncol=3); axes[1].grid(alpha=0.25)
    for ax in axes:
        _shade_push(ax, t, a["push_force"])
    fig.suptitle("Actual MuJoCo ground-reaction forces")
    fig.tight_layout()
    return list(_save(fig, output_dir, name))


def plot_com_support_polygon(log, output_dir: str | Path, name: str = "com_support_polygon") -> list[Path]:
    apply_style()
    a = log.arrays(); com = a["com_world"][:, :2]; feet = a["foot_xy_world"].reshape(-1, 2, 2)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(com[:, 0], com[:, 1], color="tab:blue", linewidth=2, label="CoM trajectory")
    ax.scatter(com[0, 0], com[0, 1], color="tab:green", label="nominal CoM", zorder=3)
    half_x, half_y, center_x = 0.17, 0.12, 0.055
    for index, label, color in ((0, "left foot support", "tab:orange"), (1, "right foot support", "tab:red")):
        center = feet[0, index] + np.array([center_x, 0.0])
        polygon = np.array([[center[0] - half_x, center[1] - half_y], [center[0] + half_x, center[1] - half_y], [center[0] + half_x, center[1] + half_y], [center[0] - half_x, center[1] + half_y], [center[0] - half_x, center[1] - half_y]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.18, label=label)
    push_mask = np.linalg.norm(a["push_force"][:, :2], axis=1) > 1e-9
    if np.any(push_mask):
        ax.scatter(com[push_mask, 0], com[push_mask, 1], s=5, color="tab:purple", label="push interval")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]"); ax.set_title("CoM trajectory and double-support polygon"); ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return list(_save(fig, output_dir, name))
