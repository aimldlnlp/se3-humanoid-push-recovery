"""Publication-oriented plots generated from measured trial data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from .style import COLORS, apply_style, panel_label, style_axes


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _controller_label(name: str) -> str:
    return {"pd": "PD", "se3_wbc": "SE(3) WBC"}.get(str(name), str(name))


def _controller_color(name: str) -> str:
    return COLORS["wbc"] if str(name) == "se3_wbc" else COLORS["pd"]


def _foot_support_vertices(a: dict[str, np.ndarray]) -> np.ndarray:
    """Return logged foot support vertices, with a legacy-data fallback."""
    raw = np.asarray(a.get("foot_support_vertices_world", np.zeros(0)), dtype=float)
    if raw.size >= 16 and raw.size % 16 == 0:
        return raw.reshape(-1, 2, 4, 2)
    feet = np.asarray(a["foot_xy_world"], dtype=float).reshape(-1, 2, 2)
    center = np.array([0.055, 0.0])
    half = np.array([0.17, 0.12])
    local = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
    return feet[:, :, None, :] + center + local[None, None, :, :]


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotonic-chain convex hull for finite 2-D support vertices."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= 1:
        return points
    points = np.unique(points, axis=0)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _active_support_hull(vertices: np.ndarray, active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool).reshape(2)
    selected = vertices[np.asarray(active)]
    return convex_hull_2d(selected.reshape(-1, 2)) if selected.size else np.zeros((0, 2))


def _torque_utilization(a: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(a.get("torque_utilization", np.zeros(0)), dtype=float)
    if values.size:
        return values
    return np.asarray(a["torque_abs_max_Nm"], dtype=float) / 180.0


def _qp_timing(ax, t: np.ndarray, solve_time_s: np.ndarray, deadline_ms: float = 4.0) -> dict[str, float]:
    values_ms = np.asarray(solve_time_s, dtype=float) * 1000.0
    finite = values_ms[np.isfinite(values_ms)]
    stats = {
        "mean_ms": float(np.mean(finite)) if finite.size else float("nan"),
        "p95_ms": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "p99_ms": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        "max_ms": float(np.max(finite)) if finite.size else float("nan"),
        "deadline_miss_pct": float(np.mean(finite > deadline_ms) * 100.0) if finite.size else float("nan"),
    }
    ax.plot(t, values_ms, color=COLORS["muted"], linewidth=1.1, label="solve time")
    ax.axhline(deadline_ms, color=COLORS["push"], linestyle=(0, (4, 3)), linewidth=1.0, label=f"deadline {deadline_ms:.1f} ms")
    ax.set_ylabel("QP solve time [ms]")
    ax.text(
        0.02, 0.96,
        "mean {:.2f} | p95 {:.2f} | p99 {:.2f}\nmax {:.2f} | misses {:.1f}%".format(
            stats["mean_ms"], stats["p95_ms"], stats["p99_ms"], stats["max_ms"], stats["deadline_miss_pct"]
        ),
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["paper"], "alpha": 0.86, "edgecolor": COLORS["grid"], "pad": 3},
    )
    return stats


def _save(fig, output_dir: str | Path, name: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=COLORS["paper"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)
    return png, pdf


def _shade_push(ax, time_s: np.ndarray, push_force: np.ndarray) -> None:
    magnitude = np.linalg.norm(np.asarray(push_force)[:, :2], axis=1) if len(push_force) else np.zeros(0)
    active = magnitude > 1e-9
    if np.any(active):
        indices = np.flatnonzero(active)
        ax.axvspan(float(time_s[indices[0]]), float(time_s[indices[-1]]), color=COLORS["push"], alpha=0.11, linewidth=0)


def _push_summary(push_force: np.ndarray) -> tuple[float, float]:
    """Return the measured force magnitude and direction in a logged trial."""
    force = np.asarray(push_force, dtype=float)
    active = np.linalg.norm(force[:, :2], axis=1) > 1e-9 if force.ndim == 2 and len(force) else np.zeros(0, dtype=bool)
    if not np.any(active):
        return 0.0, 0.0
    sample = force[np.flatnonzero(active)[0], :2]
    return float(np.linalg.norm(sample)), float(np.rad2deg(np.arctan2(sample[1], sample[0])) % 360.0)


def _finish_shared_time_axes(axes: np.ndarray, t: np.ndarray, push: np.ndarray) -> None:
    for ax in np.asarray(axes).reshape(-1):
        _shade_push(ax, t, push)
        style_axes(ax)


def plot_trial(log, output_dir: str | Path, prefix: str = "trial") -> list[Path]:
    apply_style()
    a = log.arrays()
    t = a["time_s"]
    paths: list[Path] = []
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 4.8), sharex=True)
    for i, color in enumerate((COLORS["axis_x"], COLORS["axis_y"], COLORS["axis_z"])):
        ax[0].plot(t, a["torso_error"][:, 3 + i], color=color, label=("x", "y", "z")[i])
        ax[1].plot(t, a["torso_error"][:, i], color=color, label=("x", "y", "z")[i])
    ax[0].set_ylabel("rotation error [rad]")
    ax[1].set_ylabel("translation error [m]")
    ax[1].set_xlabel("time [s]")
    for axis in ax:
        axis.legend(ncol=3, loc="upper right")
        style_axes(axis)
    paths.extend(_save(fig, output_dir, f"{prefix}_se3_error"))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, color in enumerate((COLORS["axis_x"], COLORS["axis_y"], COLORS["axis_z"])):
        ax.plot(t, a["com_world"][:, i], color=color, label=("x", "y", "z")[i])
    ax.set_xlabel("time [s]"); ax.set_ylabel("CoM position [m]"); ax.legend(ncol=3, loc="upper right"); style_axes(ax)
    paths.extend(_save(fig, output_dir, f"{prefix}_com"))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    wrench = a["actual_contact_wrench"]
    if wrench.ndim == 2 and wrench.shape[1] >= 12:
        ax.plot(t, wrench[:, 2], color=COLORS["left_foot"], label="left $F_z$")
        ax.plot(t, wrench[:, 8], color=COLORS["right_foot"], label="right $F_z$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("vertical GRF [N]"); ax.legend(loc="upper right"); style_axes(ax)
    paths.extend(_save(fig, output_dir, f"{prefix}_contact_forces"))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot(t, _torque_utilization(a), color=COLORS["actual"], label=r"max $|\tau_i|/\tau_{i,\max}$")
    ax.axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label="limit")
    ax.set_xlabel("time [s]"); ax.set_ylabel("torque utilization"); ax.set_ylim(bottom=0); ax.legend(loc="upper right"); style_axes(ax)
    paths.extend(_save(fig, output_dir, f"{prefix}_torques"))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot(t, a["push_force"][:, 0], color=COLORS["push"], label="$F_x$")
    ax.plot(t, a["push_force"][:, 1], color=COLORS["axis_y"], label="$F_y$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("push force [N]"); ax.legend(loc="upper right"); style_axes(ax)
    paths.extend(_save(fig, output_dir, f"{prefix}_push_force"))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    _qp_timing(ax, t, a["qp_solve_time_s"])
    ax.set_xlabel("time [s]"); style_axes(ax)
    paths.extend(_save(fig, output_dir, f"{prefix}_qp_solve_time"))
    return paths


def plot_comparison(rows: list[dict], output_dir: str | Path, name: str = "controller_comparison") -> list[Path]:
    apply_style()
    controllers = [c for c in ("pd", "se3_wbc") if any(str(r["controller"]) == c for r in rows)]
    labels = [_controller_label(c) for c in controllers]
    rates = []
    largest = []
    median_latency = []
    for controller in controllers:
        subset = [r for r in rows if str(r["controller"]) == controller]
        successes = [r for r in subset if _as_bool(r.get("success", False))]
        rates.append(float(np.mean([_as_bool(r.get("success", False)) for r in subset])))
        largest.append(max((float(r["push_magnitude_N"]) for r in successes), default=float("nan")))
        latency = [float(r["recovery_latency_s"]) for r in successes if str(r.get("recovery_latency_s", "")) not in {"", "nan"}]
        median_latency.append(float(np.median(latency)) if latency else float("nan"))

    fig = plt.figure(figsize=(9.2, 3.9))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.45), wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    bars = ax.bar(labels, rates, color=[_controller_color(c) for c in controllers], width=0.58)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.025, f"{100 * rate:.1f}%", ha="center", va="bottom", color=COLORS["ink"], fontsize=9)
    per_controller = max((sum(str(r["controller"]) == c for r in rows) for c in controllers), default=0)
    ax.set_ylim(0, 1.08); ax.set_ylabel("recovery rate"); ax.set_title(f"Success over {per_controller} tested pushes / controller", loc="left", pad=8); style_axes(ax)

    ax_table = fig.add_subplot(grid[0, 1]); ax_table.axis("off")
    ax_table.set_title("Measured basin summary", loc="left", pad=8)
    ax_table.add_patch(plt.Rectangle((0.02, 0.22), 0.96, 0.58, facecolor=COLORS["paper"], edgecolor=COLORS["grid"], linewidth=0.8, transform=ax_table.transAxes))
    x_values = (0.63, 0.85) if len(labels) == 2 else np.linspace(0.63, 0.88, len(labels))
    for x, label, controller in zip(x_values, labels, controllers):
        ax_table.text(x, 0.70, label, ha="center", va="center", color=_controller_color(controller), fontsize=9.5, transform=ax_table.transAxes)
    ax_table.text(0.05, 0.51, "largest recovered push [N]", ha="left", va="center", color=COLORS["ink"], fontsize=8.4, transform=ax_table.transAxes)
    ax_table.text(0.05, 0.31, "median latency [s]", ha="left", va="center", color=COLORS["ink"], fontsize=8.4, transform=ax_table.transAxes)
    for x, value, latency in zip(x_values, largest, median_latency):
        ax_table.text(x, 0.51, f"{value:.0f}" if np.isfinite(value) else "—", ha="center", va="center", color=COLORS["ink"], fontsize=9.2, transform=ax_table.transAxes)
        ax_table.text(x, 0.31, f"{latency:.3f}" if np.isfinite(latency) else "—", ha="center", va="center", color=COLORS["ink"], fontsize=9.2, transform=ax_table.transAxes)
    for x in x_values:
        ax_table.plot([x - 0.09, x - 0.09], [0.22, 0.80], color=COLORS["grid"], linewidth=0.6, transform=ax_table.transAxes, clip_on=False)
    ax_table.plot([0.02, 0.98], [0.61, 0.61], color=COLORS["grid"], linewidth=0.6, transform=ax_table.transAxes)
    fig.suptitle("PD versus SE(3) whole-body control", fontsize=12, y=1.02)
    return list(_save(fig, output_dir, name))


def _recovery_grid(rows: list[dict], controller: str):
    mags = sorted({float(r["push_magnitude_N"]) for r in rows})
    dirs = sorted({float(r["push_direction_deg"]) for r in rows})
    grid = np.full((len(mags), len(dirs)), np.nan)
    for r in rows:
        if str(r["controller"]) == controller:
            grid[mags.index(float(r["push_magnitude_N"])), dirs.index(float(r["push_direction_deg"]))] = float(_as_bool(r.get("success", False)))
    return np.asarray(mags), np.asarray(dirs), grid


def _recovery_envelope(rows: list[dict], controller: str, dirs: np.ndarray) -> np.ndarray:
    envelope = np.full(len(dirs), np.nan)
    for i, direction in enumerate(dirs):
        successful = [float(r["push_magnitude_N"]) for r in rows if str(r["controller"]) == controller and float(r["push_direction_deg"]) == float(direction) and _as_bool(r.get("success", False))]
        if successful:
            envelope[i] = max(successful)
    return envelope


def plot_recovery_heatmap(rows: list[dict], output_dir: str | Path, controller: str = "se3_wbc", name: str = "recovery_heatmap") -> list[Path]:
    apply_style()
    mags, dirs, grid = _recovery_grid(rows, controller)
    cmap = ListedColormap([COLORS["failure"], COLORS["success"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels([f"{int(d)}" if int(d) % 45 == 0 else "" for d in dirs])
    ax.set_yticks(range(len(mags)), [f"{m:.0f}" for m in mags])
    ax.set_xticks(np.arange(-0.5, len(dirs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mags), 1), minor=True)
    ax.grid(which="minor", color=COLORS["paper"], linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("push direction [deg]"); ax.set_ylabel("push magnitude [N]")
    ax.set_title(f"{_controller_label(controller)} recovery basin", loc="left", pad=8)
    envelope = _recovery_envelope(rows, controller, dirs)
    boundary_y = np.array([mags.tolist().index(value) if np.isfinite(value) else np.nan for value in envelope], dtype=float)
    ax.plot(np.arange(len(dirs)), boundary_y, color=COLORS["boundary"], linewidth=1.3, marker=".", markersize=3, label="measured boundary")
    ax.legend(handles=[Patch(facecolor=COLORS["success"], edgecolor="none", label="Recovered"), Patch(facecolor=COLORS["failure"], edgecolor="none", label="Failed")], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    style_axes(ax, grid=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.91)
    return list(_save(fig, output_dir, name))


def plot_recovery_envelope(rows: list[dict], output_dir: str | Path, name: str = "recovery_envelope") -> list[Path]:
    """Plot maximum measured recovered push magnitude versus direction."""
    apply_style()
    dirs = np.asarray(sorted({float(r["push_direction_deg"]) for r in rows}))
    theta = np.deg2rad(dirs)
    fig, ax = plt.subplots(figsize=(6.4, 5.8), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(top=0.79, bottom=0.15, left=0.07, right=0.93)
    for controller in ("pd", "se3_wbc"):
        envelope = _recovery_envelope(rows, controller, dirs)
        finite = np.isfinite(envelope)
        radial = np.where(finite, envelope, 0.0)
        theta_closed = np.r_[theta, theta[0]]
        radial_closed = np.r_[radial, radial[0]]
        ax.plot(theta_closed, radial_closed, color=_controller_color(controller), linewidth=2.0, marker="o", markersize=3.8, label=_controller_label(controller))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.set_rlabel_position(105)
    max_magnitude = max((float(r["push_magnitude_N"]) for r in rows), default=1.0)
    ax.set_ylim(0, max(max_magnitude, 1.0) + 10.0)
    fig.suptitle("Sampled measured recovery envelope", fontsize=12, y=0.97)
    fig.text(0.5, 0.915, "largest recovered tested push by direction — not a continuous boundary", ha="center", va="center", fontsize=8.2, color=COLORS["muted"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(color=COLORS["grid"], alpha=0.5, linewidth=0.6)
    return list(_save(fig, output_dir, name))


def plot_gpu_benchmark(rows: list[dict], output_dir: str | Path, name: str = "gpu_benchmark") -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = [r for r in rows if str(r.get("status", "")) == "ok"]
    if valid:
        ax.plot([r["batch_size"] for r in valid], [r["simulations_per_second"] for r in valid], marker="o", color=COLORS["wbc"], label="throughput")
    ax.set_xlabel("batch size"); ax.set_ylabel("simulations / second"); style_axes(ax)
    if valid:
        ax.legend()
    return list(_save(fig, output_dir, name))


def plot_flagship(log, output_dir: str | Path, name: str = "canonical_response") -> list[Path]:
    """Landscape six-panel canonical response figure for the README."""
    apply_style()
    a = log.arrays()
    t = a["time_s"]
    push = a["push_force"]
    push_magnitude, push_direction = _push_summary(push)
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.1), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.94, hspace=0.48, wspace=0.24)

    axes[0, 0].plot(t, np.linalg.norm(push[:, :2], axis=1), color=COLORS["push"])
    axes[0, 0].set_ylabel("Push force [N]")
    axes[0, 0].set_title("(a) Applied disturbance", loc="left", pad=7)
    axes[0, 0].text(0.98, 0.88, f"{push_magnitude:.0f} N @ {push_direction:.0f}°", transform=axes[0, 0].transAxes, ha="right", color=COLORS["push"], fontsize=8.5)

    axes[0, 1].plot(t, a["torso_rotation_error_rad"], color=COLORS["actual"])
    axes[0, 1].set_ylabel("Torso orientation error [rad]")
    axes[0, 1].set_title("(b) Geometric pose response", loc="left", pad=7)

    com = a["com_world"]
    axes[1, 0].plot(t, np.linalg.norm(com[:, :2] - com[0, :2], axis=1), color=COLORS["com"])
    axes[1, 0].set_ylabel("Horizontal CoM displacement [m]")
    axes[1, 0].set_title("(c) Whole-body translation", loc="left", pad=7)

    actual = a["actual_contact_wrench"]
    axes[1, 1].plot(t, actual[:, 2], color=COLORS["left_foot"], label="left foot")
    axes[1, 1].plot(t, actual[:, 8], color=COLORS["right_foot"], label="right foot")
    axes[1, 1].set_ylabel("Vertical GRF [N]")
    axes[1, 1].set_title("(d) Measured ground reaction", loc="left", pad=7)
    axes[1, 1].legend(loc="upper right", ncol=2)

    axes[2, 0].plot(t, _torque_utilization(a), color=COLORS["actual"], label=r"max $|\tau_i|/\tau_{i,\max}$")
    axes[2, 0].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label="limit = 1")
    axes[2, 0].set_ylabel("Torque utilization")
    axes[2, 0].set_title("(e) Actuator constraint activity", loc="left", pad=7)
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].legend(loc="upper right")

    eta = np.max(a["actual_friction_utilization"], axis=1)
    axes[2, 1].plot(t, eta, color=COLORS["cop"], label=r"$\eta=\|F_t\|/(\mu F_z)$")
    axes[2, 1].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label=r"boundary $\eta=1$")
    axes[2, 1].set_ylabel(r"Friction utilization $\eta$")
    axes[2, 1].set_title("(f) Contact friction activity", loc="left", pad=7)
    axes[2, 1].set_ylim(bottom=0)
    axes[2, 1].legend(loc="upper right")

    _finish_shared_time_axes(axes, t, push)
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    fig.suptitle(f"Canonical {push_magnitude:.0f} N push: measured SE(3) WBC response", fontsize=12, y=0.99)
    return list(_save(fig, output_dir, name))


def plot_contact_diagnostics(log, output_dir: str | Path, name: str = "contact_slip_diagnostics") -> list[Path]:
    """Supplementary plot keeping friction utilization and slip speed separate."""
    apply_style()
    a = log.arrays(); t = a["time_s"]; push = a["push_force"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=True)
    axes[0].plot(t, np.max(a["actual_friction_utilization"], axis=1), color=COLORS["cop"], label=r"$\eta$ friction utilization")
    axes[0].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label=r"$\eta=1$ boundary")
    axes[0].set_ylabel(r"Friction utilization $\eta$"); axes[0].set_title("(a) Dimensionless friction margin", loc="left", pad=7); axes[0].legend(loc="upper right")
    axes[1].plot(t, np.max(a["foot_tangent_velocity"], axis=1), color=COLORS["push"], label="max foot tangential speed")
    axes[1].set_ylabel("Tangential speed [m/s]"); axes[1].set_xlabel("Time [s]"); axes[1].set_title("(b) Physical slip indicator", loc="left", pad=7); axes[1].legend(loc="upper right")
    _finish_shared_time_axes(axes, t, push)
    return list(_save(fig, output_dir, name))


def plot_qp_diagnostics(log, output_dir: str | Path, name: str = "qp_timing_diagnostics") -> list[Path]:
    apply_style()
    a = log.arrays(); t = a["time_s"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    _qp_timing(ax, t, a["qp_solve_time_s"])
    ax.set_xlabel("Time [s]"); ax.set_title("QP timing against the 4 ms control deadline", loc="left", pad=8); style_axes(ax)
    return list(_save(fig, output_dir, name))


def plot_actual_grf(log, output_dir: str | Path, name: str = "actual_ground_reaction_forces") -> list[Path]:
    apply_style()
    a = log.arrays(); t = a["time_s"]; wrench = a["actual_contact_wrench"]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.4), sharex=True)
    axes[0].plot(t, wrench[:, 0], color=COLORS["axis_x"], label="$F_x$")
    axes[0].plot(t, wrench[:, 1], color=COLORS["axis_y"], label="$F_y$")
    axes[0].plot(t, wrench[:, 2], color=COLORS["left_foot"], label="$F_z$")
    axes[0].set_ylabel("Left foot GRF [N]"); axes[0].legend(ncol=3, loc="upper right"); axes[0].set_title("Left foot", loc="left", pad=7)
    axes[1].plot(t, wrench[:, 6], color=COLORS["axis_x"], label="$F_x$")
    axes[1].plot(t, wrench[:, 7], color=COLORS["axis_y"], label="$F_y$")
    axes[1].plot(t, wrench[:, 8], color=COLORS["right_foot"], label="$F_z$")
    axes[1].set_ylabel("Right foot GRF [N]"); axes[1].set_xlabel("Time [s]"); axes[1].legend(ncol=3, loc="upper right"); axes[1].set_title("Right foot", loc="left", pad=7)
    _finish_shared_time_axes(axes, t, a["push_force"])
    fig.suptitle("Actual MuJoCo ground-reaction forces", fontsize=12, y=0.99)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.91, hspace=0.32)
    return list(_save(fig, output_dir, name))


def _finite_cop(a: dict[str, np.ndarray], foot: int) -> np.ndarray:
    cop = np.asarray(a.get("foot_cop_world", np.zeros(0)), dtype=float)
    if cop.size < 4:
        return np.empty((0, 2))
    values = cop.reshape(-1, 2, 2)[:, foot]
    return values[np.all(np.isfinite(values), axis=1)]


def plot_com_support_polygon(log, output_dir: str | Path, name: str = "com_support_polygon") -> list[Path]:
    apply_style()
    a = log.arrays(); com = a["com_world"][:, :2]; vertices = _foot_support_vertices(a)
    active = np.column_stack([a["contact_left"], a["contact_right"]]).astype(bool)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for index, label, color in ((0, "left foot", COLORS["left_foot"]), (1, "right foot", COLORS["right_foot"])):
        polygon = np.vstack([vertices[0, index], vertices[0, index, 0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.16, label=label, zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.0, zorder=2)
    hull = _active_support_hull(vertices[0], active[0])
    if len(hull) >= 3:
        closed = np.vstack([hull, hull[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=COLORS["wbc"], alpha=0.09, label="double-support hull", zorder=0)
        ax.plot(closed[:, 0], closed[:, 1], color=COLORS["wbc"], linewidth=1.7, zorder=3)
    ax.plot(com[:, 0], com[:, 1], color=COLORS["com"], linewidth=2.0, label="CoM trajectory", zorder=4)
    push_mask = np.linalg.norm(a["push_force"][:, :2], axis=1) > 1e-9
    if np.any(push_mask):
        ax.plot(com[push_mask, 0], com[push_mask, 1], color=COLORS["push"], linewidth=3.0, label="push interval", zorder=5)
    peak = int(np.argmax(np.linalg.norm(com - com[0], axis=1)))
    ax.scatter(com[0, 0], com[0, 1], color=COLORS["desired"], marker="o", s=42, label="initial CoM", zorder=7)
    ax.scatter(com[peak, 0], com[peak, 1], color=COLORS["push"], marker="^", s=48, label="peak CoM", zorder=7)
    ax.scatter(com[-1, 0], com[-1, 1], color=COLORS["actual"], marker="s", s=40, label="final CoM", zorder=7)
    for foot, color in ((0, COLORS["left_foot"]), (1, COLORS["right_foot"])):
        cop = _finite_cop(a, foot)
        if len(cop):
            stride = max(1, len(cop) // 18)
            ax.scatter(cop[::stride, 0], cop[::stride, 1], color=COLORS["cop"], alpha=0.60, s=14, marker=".", zorder=6)
    if any(len(_finite_cop(a, foot)) for foot in (0, 1)):
        ax.plot([], [], color=COLORS["cop"], linewidth=1.1, marker=".", label="measured CoP")
    finite_points = np.concatenate([com, vertices.reshape(-1, 2)], axis=0)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    lo = finite_points.min(axis=0); hi = finite_points.max(axis=0)
    pad = max(0.025, 0.08 * float(np.max(hi - lo)))
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x [m]"); ax.set_ylabel("World y [m]")
    ax.set_title("CoM motion inside the measured double-support region", loc="left", pad=9)
    style_axes(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
        columnspacing=0.8, handlelength=1.8, handletextpad=0.5,
        labelspacing=0.35, fontsize=8.0,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.92)
    return list(_save(fig, output_dir, name))
