"""Publication-oriented plots generated from measured trial data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .style import COLORS, apply_style, panel_label, style_axes


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _controller_label(name: str) -> str:
    return {"pd": "PD", "se3_wbc": "SE(3) WBC", "fixed_foot_wbc": "fixed-foot WBC", "hybrid_se3_wbc": "hybrid one-step WBC"}.get(str(name), str(name))


def _controller_color(name: str) -> str:
    return COLORS["wbc"] if str(name) in {"se3_wbc", "fixed_foot_wbc", "hybrid_se3_wbc"} else COLORS["pd"]


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


def _support_margin(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Return signed distance to a convex support hull (positive inside)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    hull = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(hull) < 3:
        return np.full(len(points), np.nan)
    signed_area = 0.5 * np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - hull[:, 1] * np.roll(hull[:, 0], -1))
    if signed_area < 0.0:
        hull = hull[::-1]
    edge = np.roll(hull, -1, axis=0) - hull
    length = np.linalg.norm(edge, axis=1)
    delta = points[:, None, :] - hull[None, :, :]
    cross = edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0]
    return np.min(cross / np.maximum(length[None, :], 1e-12), axis=1)


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
        transform=ax.transAxes, ha="left", va="top", fontsize=10.0,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["paper"], "alpha": 0.86, "edgecolor": COLORS["grid"], "pad": 3},
    )
    return stats


def _save(fig, output_dir: str | Path, name: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    svg_dir = output_dir.parent / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg = svg_dir / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=COLORS["paper"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=COLORS["paper"])
    fig.savefig(svg, bbox_inches="tight", facecolor=COLORS["paper"])
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
    tested, recovered, rates, largest, median_latency = [], [], [], [], []
    for controller in controllers:
        subset = [r for r in rows if str(r["controller"]) == controller]
        successes = [r for r in subset if _as_bool(r.get("success", False))]
        tested.append(len(subset))
        recovered.append(len(successes))
        rates.append(float(np.mean([_as_bool(r.get("success", False)) for r in subset])))
        largest.append(max((float(r["push_magnitude_N"]) for r in successes), default=float("nan")))
        latency = [float(r["recovery_latency_s"]) for r in successes if str(r.get("recovery_latency_s", "")) not in {"", "nan"}]
        median_latency.append(float(np.median(latency)) if latency else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), gridspec_kw={"width_ratios": (1.05, 1.35)})
    ax = axes[0]
    y = np.arange(len(labels))
    bars = ax.barh(y, np.asarray(rates) * 100.0, color=[_controller_color(c) for c in controllers], height=0.34)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("recovery rate [%]")
    ax.set_title("(a) Measured recovery", loc="left", pad=5)
    for bar, ok, total in zip(bars, recovered, tested):
        ax.text(min(float(bar.get_width()) + 2.0, 96.0), bar.get_y() + bar.get_height() / 2, f"{ok}/{total}", ha="left", va="center", color=COLORS["ink"], fontsize=10.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    style_axes(ax)

    ax_table = axes[1]
    ax_table.axis("off")
    ax_table.set_title("(b) Basin summary", loc="left", pad=5)
    x_values = np.linspace(0.61, 0.88, max(len(labels), 2))[:len(labels)]
    ax_table.text(0.04, 0.78, "metric", ha="left", va="center", color=COLORS["muted"], fontsize=10.0, transform=ax_table.transAxes)
    for x, label, controller in zip(x_values, labels, controllers):
        ax_table.text(x, 0.78, label, ha="center", va="center", color=_controller_color(controller), fontsize=10.0, transform=ax_table.transAxes)
    metrics = (
        ("largest recovered [N]", largest, lambda value: f"{value:.0f}" if np.isfinite(value) else "—"),
        ("median latency [s]", median_latency, lambda value: f"{value:.3f}" if np.isfinite(value) else "—"),
    )
    for row_index, (label, values, formatter) in enumerate(metrics):
        y_pos = 0.56 - 0.22 * row_index
        ax_table.text(0.04, y_pos, label, ha="left", va="center", color=COLORS["ink"], fontsize=10.0, transform=ax_table.transAxes)
        for x, value in zip(x_values, values):
            ax_table.text(x, y_pos, formatter(value), ha="center", va="center", color=COLORS["ink"], fontsize=10.5, transform=ax_table.transAxes)
        ax_table.plot([0.03, 0.97], [y_pos - 0.105, y_pos - 0.105], color=COLORS["grid"], linewidth=0.6, transform=ax_table.transAxes)
    ax_table.plot([0.03, 0.97], [0.69, 0.69], color=COLORS["grid"], linewidth=0.7, transform=ax_table.transAxes)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.17, top=0.87, wspace=0.35)
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


def _draw_recovery_heatmap(ax, rows: list[dict], controller: str, *, show_ylabel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mags, dirs, grid = _recovery_grid(rows, controller)
    cmap = ListedColormap([COLORS["failure"], COLORS["success"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels([f"{int(d)}" if int(d) % 45 == 0 else "" for d in dirs])
    ax.set_yticks(range(len(mags)), [f"{m:.0f}" for m in mags])
    ax.set_xticks(np.arange(-0.5, len(dirs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mags), 1), minor=True)
    ax.grid(which="minor", color=COLORS["paper"], linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("push direction [deg]")
    if show_ylabel:
        ax.set_ylabel("push magnitude [N]")
    else:
        ax.set_ylabel("")
    recovered = int(np.nansum(grid == 1.0))
    total = int(np.sum(np.isfinite(grid)))
    ax.set_title(f"{_controller_label(controller)}", loc="left", pad=5)
    ax.text(0.99, 1.02, f"{recovered}/{total} recovered", transform=ax.transAxes, ha="right", va="bottom", fontsize=10.0, color=COLORS["muted"])
    envelope = _recovery_envelope(rows, controller, dirs)
    boundary_y = np.array([mags.tolist().index(value) if np.isfinite(value) else np.nan for value in envelope], dtype=float)
    ax.plot(np.arange(len(dirs)), boundary_y, color=COLORS["boundary"], linewidth=0.9, linestyle=(0, (2, 2)), marker="o", markersize=2.2, label="last recovered tested cell")
    style_axes(ax, grid=False)
    return mags, dirs


def _recovery_legend_handles() -> list:
    return [
        Patch(facecolor=COLORS["success"], edgecolor="none", label="recovered"),
        Patch(facecolor=COLORS["failure"], edgecolor="none", label="failed"),
        Line2D([0], [0], color=COLORS["boundary"], linewidth=0.9, linestyle=(0, (2, 2)), marker="o", markersize=2.2, label="last recovered tested cell"),
    ]


def plot_recovery_heatmap(rows: list[dict], output_dir: str | Path, controller: str = "se3_wbc", name: str = "recovery_heatmap") -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    _draw_recovery_heatmap(ax, rows, controller)
    fig.legend(handles=_recovery_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.86)
    return list(_save(fig, output_dir, name))


def plot_recovery_basin(rows: list[dict], output_dir: str | Path, name: str = "recovery_basin") -> list[Path]:
    """Compact side-by-side PD/WBC basin with a shared discrete color key."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), sharex=True, sharey=True)
    _draw_recovery_heatmap(axes[0], rows, "pd", show_ylabel=True)
    _draw_recovery_heatmap(axes[1], rows, "se3_wbc", show_ylabel=False)
    fig.text(0.5, 0.96, "Sampled recovery basin", ha="center", va="center", fontsize=13.0)
    fig.text(0.5, 0.915, "each cell is one tested push; colors show the common physical classifier", ha="center", va="center", fontsize=10.0, color=COLORS["muted"])
    fig.legend(handles=_recovery_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.84, wspace=0.10)
    return list(_save(fig, output_dir, name))


def plot_hybrid_recovery_basin(rows: list[dict], output_dir: str | Path, name: str = "hybrid_recovery_basin") -> list[Path]:
    """Compare fixed-foot and one-step recovery without hiding failures."""
    apply_style()
    fixed_mags, fixed_dirs, fixed = _recovery_grid(rows, "fixed_foot_wbc")
    hybrid_mags, hybrid_dirs, hybrid = _recovery_grid(rows, "hybrid_se3_wbc")
    if not (np.array_equal(fixed_mags, hybrid_mags) and np.array_equal(fixed_dirs, hybrid_dirs)):
        raise ValueError("fixed-foot and hybrid grids must have identical tested cells")
    step_grid = np.full_like(hybrid, False, dtype=bool)
    for row in rows:
        if str(row.get("controller")) == "hybrid_se3_wbc":
            i = list(hybrid_mags).index(float(row["push_magnitude_N"]))
            j = list(hybrid_dirs).index(float(row["push_direction_deg"]))
            step_grid[i, j] = _as_bool(row.get("step_triggered", False))

    cmap = ListedColormap([COLORS["failure"], COLORS["pd"], COLORS["push"], COLORS["wbc"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    comparable = np.isfinite(fixed) & np.isfinite(hybrid)
    delta = np.full_like(fixed, np.nan, dtype=float)
    delta[comparable] = np.where((fixed[comparable] == 1.0) & (hybrid[comparable] == 1.0), 3.0,
                                 np.where((fixed[comparable] == 1.0) & (hybrid[comparable] == 0.0), 1.0,
                                          np.where((fixed[comparable] == 0.0) & (hybrid[comparable] == 1.0), 2.0, 0.0)))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.45), sharex=True, sharey=True,
                             gridspec_kw={"width_ratios": (1.0, 1.0, 1.10)})
    grids = ((fixed, "fixed-foot WBC"), (hybrid, "hybrid one-step WBC"), (delta, "hybrid effect"))
    for index, (ax, (grid, title)) in enumerate(zip(axes, grids)):
        ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(fixed_dirs)))
        ax.set_xticklabels([f"{int(d)}" if int(d) % 45 == 0 else "" for d in fixed_dirs])
        ax.set_yticks(range(len(fixed_mags)), [f"{m:.0f}" for m in fixed_mags])
        ax.set_xticks(np.arange(-0.5, len(fixed_dirs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(fixed_mags), 1), minor=True)
        ax.grid(which="minor", color=COLORS["paper"], linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("push direction [deg]")
        ax.set_title(title, loc="left", pad=5)
        if index == 0:
            ax.set_ylabel("push magnitude [N]")
        else:
            ax.set_ylabel("")
        recovered = int(np.nansum(grid == 1.0)) if index < 2 else int(np.nansum(grid >= 2.0))
        total = int(np.sum(np.isfinite(grid)))
        if index < 2:
            recovered = int(np.nansum(grid == 1.0))
        ax.text(0.99, 1.02, f"{recovered}/{total} recovered", transform=ax.transAxes, ha="right", va="bottom", fontsize=10.0, color=COLORS["muted"])
        style_axes(ax, grid=False)
    # A star marks cells in which the hybrid controller actually entered the
    # single-support mode; recovery alone does not imply a step occurred.
    for i, j in zip(*np.where(step_grid)):
        axes[1].plot(j, i, marker="*", markersize=5.0, markerfacecolor=COLORS["paper"], markeredgecolor=COLORS["ink"], markeredgewidth=0.55)
    handles = [
        Patch(facecolor=COLORS["failure"], edgecolor="none", label="both failed"),
        Patch(facecolor=COLORS["pd"], edgecolor="none", label="fixed-foot only"),
        Patch(facecolor=COLORS["push"], edgecolor="none", label="hybrid only"),
        Patch(facecolor=COLORS["wbc"], edgecolor="none", label="both recovered"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["paper"], markeredgecolor=COLORS["ink"], markersize=5.0, label="hybrid entered swing mode"),
    ]
    fig.text(0.5, 0.965, "Fixed-foot versus one-step recovery", ha="center", va="center", fontsize=13.0)
    fig.text(0.5, 0.915, "same G1 model, push grid, classifier, and source; cells remain measured trials", ha="center", va="center", fontsize=10.0, color=COLORS["muted"])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, columnspacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.84, wspace=0.11)
    return list(_save(fig, output_dir, name))


def plot_recovery_envelope(rows: list[dict], output_dir: str | Path, name: str = "recovery_envelope") -> list[Path]:
    """Plot the sampled recovery limit against push direction in Cartesian form."""
    apply_style()
    dirs = np.asarray(sorted({float(r["push_direction_deg"]) for r in rows}))
    fig, ax = plt.subplots(figsize=(7.55, 4.10))
    curves: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for controller in ("pd", "se3_wbc"):
        envelope = _recovery_envelope(rows, controller, dirs)
        curves[controller] = envelope
        subset = [r for r in rows if str(r["controller"]) == controller]
        recovered = sum(_as_bool(r.get("success", False)) for r in subset)
        labels[controller] = f"{_controller_label(controller)}  {recovered}/{len(subset)} recovered"

    # Close the 0/360-degree seam without suggesting an unsampled boundary.
    closes_seam = len(dirs) > 1 and np.isclose(dirs[0], 0.0) and dirs[-1] < 360.0
    xline = np.r_[dirs, 360.0] if closes_seam else dirs
    for controller in ("pd", "se3_wbc"):
        envelope = curves[controller]
        yline = np.r_[envelope, envelope[0]] if closes_seam else envelope
        finite = np.isfinite(envelope)
        ax.plot(xline, yline, color=_controller_color(controller), linewidth=1.65, label=labels[controller], zorder=3)
        ax.scatter(dirs[finite], envelope[finite], color=_controller_color(controller), s=23, zorder=4)

    pd_curve = curves["pd"]
    wbc_curve = curves["se3_wbc"]
    if closes_seam:
        pd_curve = np.r_[pd_curve, pd_curve[0]]
        wbc_curve = np.r_[wbc_curve, wbc_curve[0]]
    finite_gap = np.isfinite(pd_curve) & np.isfinite(wbc_curve)
    if np.any(finite_gap):
        ax.fill_between(xline, pd_curve, wbc_curve, where=finite_gap, color=COLORS["wbc"], alpha=0.08, zorder=1)

    magnitudes = sorted({float(r["push_magnitude_N"]) for r in rows})
    max_magnitude = max(magnitudes, default=1.0)
    y_limit = max(20.0, float(np.ceil(max_magnitude / 20.0) * 20.0))
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, y_limit + 4.0)
    ax.set_xticks(np.arange(0.0, 361.0, 45.0))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "360°"])
    ax.set_yticks(np.arange(0.0, y_limit + 0.1, 20.0))
    ax.set_xlabel("push direction [deg]")
    ax.set_ylabel("largest recovered tested push [N]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, handlelength=1.8, columnspacing=1.8)
    style_axes(ax)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.42, linewidth=0.6)
    for direction in (0.0, 90.0, 180.0, 270.0):
        ax.axvline(direction, color=COLORS["grid"], alpha=0.22, linewidth=0.55, zorder=0)
    fig.text(0.5, 0.965, "Measured recovery by push direction", ha="center", va="center", fontsize=13.0)
    fig.text(0.5, 0.925, "24 sampled directions at 15° spacing · connecting lines are visual guides, not a continuous boundary", ha="center", va="center", fontsize=10.0, color=COLORS["muted"])
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.80)
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
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 6.0), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.93, hspace=0.58, wspace=0.27)

    axes[0, 0].plot(t, np.linalg.norm(push[:, :2], axis=1), color=COLORS["push"])
    axes[0, 0].set_ylabel("force [N]")
    axes[0, 0].set_title("(a) push", loc="left", pad=5)
    axes[0, 0].text(0.98, 0.84, f"{push_magnitude:.0f} N @ {push_direction:.0f}°", transform=axes[0, 0].transAxes, ha="right", color=COLORS["push"], fontsize=10.0)

    axes[0, 1].plot(t, a["torso_rotation_error_rad"], color=COLORS["actual"])
    axes[0, 1].set_ylabel("orientation error [rad]")
    axes[0, 1].set_title("(b) torso orientation", loc="left", pad=5)

    com = a["com_world"]
    axes[1, 0].plot(t, np.linalg.norm(com[:, :2] - com[0, :2], axis=1), color=COLORS["com"])
    axes[1, 0].set_ylabel("horizontal CoM [m]")
    axes[1, 0].set_title("(c) translation", loc="left", pad=5)

    actual = a["actual_contact_wrench"]
    axes[1, 1].plot(t, actual[:, 2], color=COLORS["left_foot"], label="left foot")
    axes[1, 1].plot(t, actual[:, 8], color=COLORS["right_foot"], label="right foot")
    axes[1, 1].set_ylabel("vertical GRF [N]")
    axes[1, 1].set_title("(d) measured GRF", loc="left", pad=5)
    axes[1, 1].legend(loc="upper right", ncol=2, handlelength=1.4)

    axes[2, 0].plot(t, _torque_utilization(a), color=COLORS["actual"], label=r"max $|\tau_i|/\tau_{i,\max}$")
    axes[2, 0].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label="limit = 1")
    axes[2, 0].set_ylabel("torque utilization")
    axes[2, 0].set_title("(e) actuator limit", loc="left", pad=5)
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].legend(loc="upper right", handlelength=1.4)

    eta = np.max(a["actual_friction_utilization"], axis=1)
    axes[2, 1].plot(t, eta, color=COLORS["cop"], label=r"$\eta=\|F_t\|/(\mu F_z)$")
    axes[2, 1].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label=r"boundary $\eta=1$")
    axes[2, 1].set_ylabel(r"friction utilization $\eta$")
    axes[2, 1].set_title("(f) friction margin", loc="left", pad=5)
    axes[2, 1].set_ylim(bottom=0)
    axes[2, 1].legend(loc="upper right", handlelength=1.4)

    _finish_shared_time_axes(axes, t, push)
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    fig.text(0.5, 0.972, f"Canonical response — SE(3) WBC, {push_magnitude:.0f} N push at {push_direction:.0f}°", ha="center", va="center", fontsize=13.0)
    return list(_save(fig, output_dir, name))


def plot_contact_diagnostics(log, output_dir: str | Path, name: str = "contact_slip_diagnostics") -> list[Path]:
    """Supplementary plot keeping friction utilization and slip speed separate."""
    apply_style()
    a = log.arrays(); t = a["time_s"]; push = a["push_force"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.35), sharex=True)
    axes[0].plot(t, np.max(a["actual_friction_utilization"], axis=1), color=COLORS["cop"], label=r"$\eta$ friction utilization")
    axes[0].axhline(1.0, color=COLORS["boundary"], linestyle=(0, (4, 3)), linewidth=1.0, label=r"$\eta=1$ boundary")
    axes[0].set_ylabel(r"friction $\eta$"); axes[0].set_title("(a) friction utilization", loc="left", pad=5); axes[0].legend(loc="upper right", handlelength=1.4)
    axes[1].plot(t, np.max(a["foot_tangent_velocity"], axis=1), color=COLORS["push"], label="max foot tangential speed")
    axes[1].set_ylabel("foot speed [m/s]"); axes[1].set_xlabel("time [s]"); axes[1].set_title("(b) physical slip indicator", loc="left", pad=5); axes[1].legend(loc="upper right", handlelength=1.4)
    _finish_shared_time_axes(axes, t, push)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.91, hspace=0.34)
    return list(_save(fig, output_dir, name))


def plot_qp_diagnostics(log, output_dir: str | Path, name: str = "qp_timing_diagnostics") -> list[Path]:
    apply_style()
    a = log.arrays(); t = a["time_s"]
    fig, ax = plt.subplots(figsize=(8.0, 3.35))
    _qp_timing(ax, t, a["qp_solve_time_s"])
    ax.set_xlabel("time [s]"); ax.set_title("QP timing diagnostic", loc="left", pad=5); ax.legend(loc="upper right", handlelength=1.4); style_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.88)
    return list(_save(fig, output_dir, name))


def plot_actual_grf(log, output_dir: str | Path, name: str = "actual_ground_reaction_forces") -> list[Path]:
    apply_style()
    a = log.arrays(); t = a["time_s"]; wrench = a["actual_contact_wrench"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.85), sharex=True)
    axes[0].plot(t, wrench[:, 0], color=COLORS["axis_x"], label="$F_x$")
    axes[0].plot(t, wrench[:, 1], color=COLORS["axis_y"], label="$F_y$")
    axes[0].plot(t, wrench[:, 2], color=COLORS["left_foot"], label="$F_z$")
    axes[0].set_ylabel("left GRF [N]"); axes[0].legend(ncol=3, loc="upper right", handlelength=1.4); axes[0].set_title("(a) left contact", loc="left", pad=5)
    axes[1].plot(t, wrench[:, 6], color=COLORS["axis_x"], label="$F_x$")
    axes[1].plot(t, wrench[:, 7], color=COLORS["axis_y"], label="$F_y$")
    axes[1].plot(t, wrench[:, 8], color=COLORS["right_foot"], label="$F_z$")
    axes[1].set_ylabel("right GRF [N]"); axes[1].set_xlabel("time [s]"); axes[1].legend(ncol=3, loc="upper right", handlelength=1.4); axes[1].set_title("(b) right contact", loc="left", pad=5)
    _finish_shared_time_axes(axes, t, a["push_force"])
    fig.text(0.5, 0.972, "Measured MuJoCo ground-reaction forces", ha="center", va="center", fontsize=13.0)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.90, hspace=0.38)
    return list(_save(fig, output_dir, name))


def plot_contact_wrench_consistency(log, output_dir: str | Path, name: str = "contact_wrench_consistency") -> list[Path]:
    """Compare QP-predicted contact quantities with MuJoCo measurements.

    The comparison is diagnostic: the QP wrench and the measured ground
    reaction are not assumed to be identical. Both are expressed at the foot
    body-frame origin, so force and CoP differences are physically interpretable.
    """
    apply_style()
    a = log.arrays(); t = a["time_s"]
    predicted = np.asarray(a["predicted_contact_wrench"], dtype=float)
    actual = np.asarray(a["actual_contact_wrench"], dtype=float)
    if predicted.ndim != 2 or actual.ndim != 2 or predicted.shape[1] < 12 or actual.shape[1] < 12:
        raise ValueError("contact-wrench consistency requires 12-component predicted and measured wrenches")
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.0), sharex=True)
    foot_specs = ((0, "left foot", COLORS["left_foot"]), (1, "right foot", COLORS["right_foot"]))
    for foot, label, color in foot_specs:
        off = 6 * foot
        axes[0, 0].plot(t, actual[:, off + 2], color=color, linewidth=1.3, label=f"{label} measured")
        axes[0, 0].plot(t, predicted[:, off + 2], color=color, linestyle=(0, (4, 2)), linewidth=1.0, label=f"{label} QP")
        actual_tangent = np.linalg.norm(actual[:, off:off + 2], axis=1)
        predicted_tangent = np.linalg.norm(predicted[:, off:off + 2], axis=1)
        axes[0, 1].plot(t, actual_tangent, color=color, linewidth=1.3, label=f"{label} measured")
        axes[0, 1].plot(t, predicted_tangent, color=color, linestyle=(0, (4, 2)), linewidth=1.0, label=f"{label} QP")
    axes[0, 0].set_ylabel("vertical force [N]")
    axes[0, 0].set_title("(a) vertical GRF", loc="left", pad=5)
    axes[0, 0].legend(ncol=2, loc="upper right", handlelength=1.4)
    axes[0, 1].set_ylabel("tangential force [N]")
    axes[0, 1].set_title("(b) tangential force", loc="left", pad=5)
    axes[0, 1].legend(ncol=2, loc="upper right", handlelength=1.4)

    total_force_delta = np.linalg.norm(
        predicted[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3).sum(axis=1)
        - actual[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3).sum(axis=1), axis=1,
    )
    axes[1, 0].plot(t, total_force_delta, color=COLORS["actual"], linewidth=1.2)
    axes[1, 0].set_ylabel("force discrepancy [N]")
    axes[1, 0].set_title("(c) total-force discrepancy", loc="left", pad=5)

    foot_xy = np.asarray(a["foot_xy_world"], dtype=float).reshape(-1, 2, 2)
    measured_cop = np.asarray(a["foot_cop_world"], dtype=float).reshape(-1, 2, 2)
    predicted_cop = np.full_like(measured_cop, np.nan)
    for foot in range(2):
        off = 6 * foot
        fz = predicted[:, off + 2]
        valid = fz > 1e-8
        predicted_cop[valid, foot, 0] = foot_xy[valid, foot, 0] - predicted[valid, off + 4] / fz[valid]
        predicted_cop[valid, foot, 1] = foot_xy[valid, foot, 1] + predicted[valid, off + 3] / fz[valid]
    cop_error = np.linalg.norm(predicted_cop - measured_cop, axis=2)
    for foot, label, color in foot_specs:
        axes[1, 1].plot(t, cop_error[:, foot], color=color, linewidth=1.2, label=label)
    axes[1, 1].set_ylabel("CoP discrepancy [m]")
    axes[1, 1].set_title("(d) contact-point discrepancy", loc="left", pad=5)
    axes[1, 1].legend(loc="upper right", handlelength=1.4)
    for ax in axes.flat:
        style_axes(ax)
        _shade_push(ax, t, a["push_force"])
    axes[1, 0].set_xlabel("time [s]"); axes[1, 1].set_xlabel("time [s]")
    fig.text(0.5, 0.972, "QP prediction versus physical MuJoCo contact measurement", ha="center", va="center", fontsize=13.0)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.90, wspace=0.25, hspace=0.38)
    return list(_save(fig, output_dir, name))


def _finite_cop(a: dict[str, np.ndarray], foot: int) -> np.ndarray:
    cop = np.asarray(a.get("foot_cop_world", np.zeros(0)), dtype=float)
    if cop.size < 4:
        return np.empty((0, 2))
    values = cop.reshape(-1, 2, 2)[:, foot]
    return values[np.all(np.isfinite(values), axis=1)]


def plot_com_support_polygon(log, output_dir: str | Path, name: str = "com_support_polygon") -> list[Path]:
    """Show support geometry, a derived support margin, and CoM response."""
    apply_style()
    a = log.arrays()
    com = np.asarray(a["com_world"], dtype=float)[:, :2]
    torso_xy = np.asarray(a["torso_position"], dtype=float)[:, :2]
    vertices = _foot_support_vertices(a)
    active = np.column_stack([a["contact_left"], a["contact_right"]]).astype(bool)
    t = np.asarray(a["time_s"], dtype=float)
    push = np.asarray(a["push_force"], dtype=float)
    push_mask = np.linalg.norm(push[:, :2], axis=1) > 1e-9
    push_indices = np.flatnonzero(push_mask)
    push_magnitude, push_direction = _push_summary(push)
    push_unit = np.array([np.cos(np.deg2rad(push_direction)), np.sin(np.deg2rad(push_direction))])
    lateral_unit = np.array([-push_unit[1], push_unit[0]])
    displacement = com - com[0]
    along_push = displacement @ push_unit
    lateral = displacement @ lateral_unit
    peak = int(np.argmax(np.linalg.norm(displacement, axis=1)))
    double_support_frames = np.flatnonzero(np.all(active, axis=1))
    support_frame = int(double_support_frames[0]) if len(double_support_frames) else 0
    support_vertices = vertices[support_frame]
    hull = _active_support_hull(support_vertices, np.array([True, True]))
    support_margin = np.full(len(com), np.nan)
    for frame in range(len(com)):
        frame_hull = _active_support_hull(vertices[frame], active[frame])
        if len(frame_hull) >= 3:
            support_margin[frame] = _support_margin(com[frame:frame + 1], frame_hull)[0]

    push_origin = torso_xy[push_indices[0]] if len(push_indices) else torso_xy[0]
    push_anchor = push_origin + 0.027 * lateral_unit
    push_end = push_anchor + 0.038 * push_unit if push_magnitude > 0.0 else push_origin + 0.045 * push_unit

    # Rotate the plan view so the two feet read left/right, as in a standard
    # humanoid top-down schematic; retain world labels on the axes.
    com_map = np.column_stack((-com[:, 1], com[:, 0]))
    vertices_map = np.stack((-vertices[..., 1], vertices[..., 0]), axis=-1)
    support_vertices_map = np.stack((-support_vertices[..., 1], support_vertices[..., 0]), axis=-1)
    hull_map = np.column_stack((-hull[:, 1], hull[:, 0])) if len(hull) else hull
    push_origin_map = np.array([-push_origin[1], push_origin[0]])
    push_unit_map = np.array([-push_unit[1], push_unit[0]])
    lateral_unit_map = np.array([-lateral_unit[1], lateral_unit[0]])
    push_anchor_map = push_origin_map + 0.027 * lateral_unit_map
    push_end_map = push_anchor_map + 0.038 * push_unit_map if push_magnitude > 0.0 else push_origin_map + 0.045 * push_unit_map

    fig = plt.figure(figsize=(9.15, 4.15))
    outer = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.30), wspace=0.0)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=(3.80, 0.75), hspace=0.34)
    ax = fig.add_subplot(left[0, 0])
    margin_ax = fig.add_subplot(left[1, 0])
    response_ax = fig.add_subplot(outer[0, 1])

    # Main panel: an intentionally restrained top-down schematic.  Structural
    # contact geometry is neutral; only measured/estimated motion gets color.
    if len(hull_map) >= 3:
        closed_hull = np.vstack([hull_map, hull_map[0]])
        ax.fill(closed_hull[:, 0], closed_hull[:, 1], color=COLORS["wbc"], alpha=0.035, zorder=0)

    for index, label in ((0, "L foot"), (1, "R foot")):
        polygon = np.vstack([support_vertices_map[index], support_vertices_map[index, 0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=COLORS["grid"], alpha=0.20, zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], color=COLORS["ink"], linewidth=0.85, zorder=3)
        if index == 0:
            label_position = np.array([np.min(polygon[:, 0]) + 0.012, np.max(polygon[:, 1]) - 0.012])
            label_align = {"ha": "left", "va": "top"}
        else:
            label_position = np.array([np.max(polygon[:, 0]) - 0.012, np.max(polygon[:, 1]) - 0.012])
            label_align = {"ha": "right", "va": "top"}
        ax.text(label_position[0], label_position[1], label, fontsize=9.0, color=COLORS["ink"], zorder=8, **label_align)

    if len(hull_map) >= 3:
        ax.plot(
            closed_hull[:, 0], closed_hull[:, 1], color=COLORS["wbc"],
            linewidth=1.0, linestyle=(0, (3, 2)), zorder=4,
        )
        ax.text(
            0.50, 0.06, "double support", transform=ax.transAxes,
            fontsize=9.0, color=COLORS["boundary"], ha="left", va="center", zorder=8,
        )

    for foot, marker in ((0, "o"), (1, "x")):
        cop = _finite_cop(a, foot)
        if len(cop):
            sample_count = min(5, len(cop))
            sample_indices = np.linspace(0, len(cop) - 1, sample_count, dtype=int)
            samples = cop[sample_indices]
            samples_map = np.column_stack((-samples[:, 1], samples[:, 0]))
            ax.plot(
                samples_map[:, 0], samples_map[:, 1], color=COLORS["cop"], linewidth=0.65,
                linestyle=(0, (2, 2)), marker=marker, markersize=2.8,
                markeredgewidth=0.75, alpha=0.72, zorder=4,
            )
            if foot == 0:
                label_point = samples_map[0] + np.array([-0.030, 0.018])
                ax.annotate(
                    "measured CoP", xy=samples_map[0], xytext=label_point,
                    fontsize=9.0, color=COLORS["cop"], ha="right", va="bottom",
                    arrowprops={"arrowstyle": "-", "color": COLORS["cop"], "lw": 0.45},
                    bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "pad": 1.0}, zorder=8,
                )

    ax.plot(com_map[:, 0], com_map[:, 1], color=COLORS["com"], linewidth=2.15, zorder=5)
    if len(push_indices) > 1:
        ax.plot(com_map[push_indices, 0], com_map[push_indices, 1], color=COLORS["push"], linewidth=3.0, solid_capstyle="round", zorder=6)
        middle = int(push_indices[len(push_indices) // 2])
        arrow_step = max(2, int(round(0.06 / max(float(np.median(np.diff(t))), 1e-9))))
        arrow_target = min(int(push_indices[-1]), middle + arrow_step)
        if arrow_target > middle:
            ax.annotate(
                "", xy=com_map[arrow_target], xytext=com_map[middle],
                arrowprops={"arrowstyle": "-|>", "color": COLORS["com"], "lw": 1.05, "mutation_scale": 8},
            )

    ax.scatter(com_map[0, 0], com_map[0, 1], facecolor=COLORS["paper"], edgecolor=COLORS["ink"], linewidth=1.05, marker="o", s=30, zorder=9)
    ax.scatter(com_map[peak, 0], com_map[peak, 1], facecolor=COLORS["push"], edgecolor=COLORS["paper"], linewidth=0.7, marker="^", s=42, zorder=9)
    ax.scatter(com_map[-1, 0], com_map[-1, 1], facecolor=COLORS["actual"], edgecolor=COLORS["paper"], linewidth=0.7, marker="s", s=34, zorder=9)

    if push_magnitude > 0.0:
        ax.plot(
            [push_origin_map[0], push_anchor_map[0]], [push_origin_map[1], push_anchor_map[1]],
            color=COLORS["push"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=7,
        )
        ax.scatter(push_origin_map[0], push_origin_map[1], facecolor=COLORS["paper"], edgecolor=COLORS["push"], linewidth=0.95, marker="o", s=22, zorder=8)
        ax.annotate(
            "", xy=push_end_map, xytext=push_anchor_map,
            arrowprops={"arrowstyle": "-|>", "color": COLORS["push"], "lw": 1.35, "mutation_scale": 9},
        )
        ax.text(
            push_anchor_map[0] + 0.004 * push_unit_map[0], push_anchor_map[1] + 0.006 * lateral_unit_map[1],
            rf"$F_{{\mathrm{{push}}}}$  {push_magnitude:.0f} N @ {push_direction:.0f}°",
            color=COLORS["push"], fontsize=9.0, ha="left", va="bottom",
            bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "pad": 1.0}, zorder=10,
        )

    finite_points = np.concatenate([com_map, vertices_map.reshape(-1, 2), push_origin_map[None, :], push_end_map[None, :]], axis=0)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    lo = finite_points.min(axis=0); hi = finite_points.max(axis=0)
    pad = max(0.018, 0.075 * float(np.max(hi - lo)))
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral [m]  (left +)"); ax.set_ylabel("forward [m]  (+x)")
    ax.set_title("(a) support-plane geometry", loc="left", pad=5)
    style_axes(ax, grid=False)

    # The strip makes the main scientific claim quantitative: the CoM stays
    # inside the measured double-support hull when the signed margin is > 0.
    finite_margin = np.isfinite(support_margin)
    if np.any(finite_margin):
        _shade_push(margin_ax, t, push)
        margin_ax.axhline(0.0, color=COLORS["boundary"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
        margin_ax.fill_between(t, 0.0, support_margin, where=support_margin >= 0.0, color=COLORS["wbc"], alpha=0.10, zorder=1)
        margin_ax.plot(t, support_margin, color=COLORS["com"], linewidth=1.35, zorder=3)
        margin_ax.scatter(t[0], support_margin[0], facecolor=COLORS["paper"], edgecolor=COLORS["ink"], linewidth=0.8, marker="o", s=20, zorder=4)
        margin_ax.scatter(t[peak], support_margin[peak], facecolor=COLORS["push"], edgecolor=COLORS["paper"], linewidth=0.5, marker="^", s=24, zorder=4)
        margin_ax.scatter(t[-1], support_margin[-1], facecolor=COLORS["actual"], edgecolor=COLORS["paper"], linewidth=0.5, marker="s", s=20, zorder=4)
        margin_ax.annotate("initial", xy=(t[0], support_margin[0]), xytext=(4, 4), textcoords="offset points", fontsize=9.0, color=COLORS["ink"])
        margin_ax.annotate("peak", xy=(t[peak], support_margin[peak]), xytext=(4, 4), textcoords="offset points", fontsize=9.0, color=COLORS["ink"])
        margin_ax.annotate("final", xy=(t[-1], support_margin[-1]), xytext=(-24, 4), textcoords="offset points", fontsize=9.0, color=COLORS["ink"])
        margin_ax.set_ylabel("margin [m]")
        margin_ax.set_xlabel("time [s]")
        margin_ax.set_title("support margin  (>0 = inside hull)", loc="left", pad=2, fontsize=10.0)
        margin_min = float(np.nanmin(support_margin)); margin_max = float(np.nanmax(support_margin))
        margin_span = max(margin_max - margin_min, 1e-4)
        margin_ax.set_ylim(min(0.0, margin_min) - 0.12 * margin_span, max(0.0, margin_max) + 0.22 * margin_span)
        style_axes(margin_ax)
    else:
        margin_ax.set_visible(False)

    _shade_push(response_ax, t, push)
    response_ax.axhline(0.0, color=COLORS["grid"], linewidth=0.75, zorder=0)
    response_ax.plot(t, along_push, color=COLORS["com"], linewidth=1.65, label="along push")
    response_ax.plot(t, lateral, color=COLORS["muted"], linewidth=1.05, linestyle=(0, (3, 2)), label="lateral")
    response_ax.scatter(t[peak], along_push[peak], color=COLORS["push"], marker="^", s=34, zorder=5)
    response_ax.scatter(t[-1], along_push[-1], color=COLORS["actual"], marker="s", s=28, zorder=5)
    response_ax.annotate("peak", xy=(t[peak], along_push[peak]), xytext=(5, 7), textcoords="offset points", fontsize=9.0, color=COLORS["ink"])
    response_ax.set_xlabel("time [s]")
    response_ax.set_ylabel("CoM displacement [m]")
    response_ax.set_title("(b) CoM response", loc="left", pad=6)
    response_ax.text(0.98, 0.90, f"{push_magnitude:.0f} N @ {push_direction:.0f}°", transform=response_ax.transAxes, ha="right", va="top", fontsize=10.0, color=COLORS["push"])
    response_ax.legend(loc="upper left", ncol=2, handlelength=1.5, columnspacing=0.8)
    style_axes(response_ax)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.90, wspace=0.30)
    return list(_save(fig, output_dir, name))


def plot_arena_telemetry(log, output_dir: str | Path, name: str = "arena_telemetry") -> list[Path]:
    """Create a compact mode/contact/telemetry view for arena scenarios."""
    apply_style()
    a = log.arrays()
    t = np.asarray(a["time_s"], dtype=float)
    if len(t) == 0:
        return []
    com = np.asarray(a["com_world"], dtype=float)
    com_delta = com[:, :2] - com[0, :2]
    com_norm = np.linalg.norm(com_delta, axis=1)
    margin = np.asarray(a.get("support_margin_m", np.full(len(t), np.nan)), dtype=float)
    post_wrench = np.asarray(a.get("actual_contact_wrench_post_step", a["actual_contact_wrench"]), dtype=float)
    force = post_wrench[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
    force_norm = np.linalg.norm(force, axis=2)
    contacts = np.column_stack([
        np.asarray(a.get("contact_left_post_step", a["contact_left"]), dtype=bool),
        np.asarray(a.get("contact_right_post_step", a["contact_right"]), dtype=bool),
    ])
    modes = np.asarray(a.get("control_mode", np.full(len(t), "double_support"))).astype(str)
    events = np.asarray(a.get("event_label", np.full(len(t), ""))).astype(str)
    fig, axes = plt.subplots(4, 1, figsize=(9.4, 7.2), sharex=True, gridspec_kw={"height_ratios": (1.25, 1.0, 1.0, 0.95)})
    _shade_push(axes[0], t, a.get("push_force", np.zeros((len(t), 3))))
    axes[0].plot(t, com_norm, color=COLORS["com"], linewidth=1.55, label="horizontal CoM displacement")
    axes[0].axhline(0.10, color=COLORS["boundary"], linewidth=0.8, linestyle=(0, (3, 2)), label="0.10 m criterion")
    axes[0].set_ylabel("CoM [m]")
    axes[0].set_title("Adaptive Recovery Arena | measured response", loc="left", pad=5)
    axes[0].legend(loc="upper left", ncol=2, handlelength=1.6)

    finite_margin = np.isfinite(margin)
    if np.any(finite_margin):
        axes[1].axhline(0.0, color=COLORS["boundary"], linewidth=0.8, linestyle=(0, (3, 2)))
        axes[1].plot(t, margin, color=COLORS["com"], linewidth=1.4, label="active support margin")
        axes[1].fill_between(t, 0.0, margin, where=margin >= 0.0, color=COLORS["success"], alpha=0.10)
    axes[1].set_ylabel("margin [m]")
    axes[1].set_title("support geometry  (>0 = CoM inside active hull)", loc="left", pad=3)

    axes[2].step(t, contacts[:, 0].astype(float), where="post", color=COLORS["left_foot"], linewidth=1.2, label="left contact")
    axes[2].step(t, contacts[:, 1].astype(float) + 1.05, where="post", color=COLORS["right_foot"], linewidth=1.2, label="right contact")
    axes[2].plot(t, force_norm[:, 0] / 100.0, color=COLORS["left_foot"], alpha=0.50, linewidth=0.85, linestyle=(0, (2, 2)), label="left force / 100 N")
    axes[2].plot(t, force_norm[:, 1] / 100.0 + 1.05, color=COLORS["right_foot"], alpha=0.50, linewidth=0.85, linestyle=(0, (2, 2)), label="right force / 100 N")
    axes[2].set_ylim(-0.15, 2.2)
    axes[2].set_yticks([0.0, 1.05, 2.0], ["L", "R", ""])
    axes[2].set_ylabel("contact")
    axes[2].set_title("contact state and measured ground-reaction magnitude", loc="left", pad=3)
    axes[2].legend(loc="upper left", ncol=2, handlelength=1.6)

    mode_spans = []
    start = 0
    for index in range(1, len(t) + 1):
        if index == len(t) or modes[index] != modes[start]:
            mode_spans.append((float(t[start]), float(t[index - 1]), modes[start]))
            start = index
    mode_colors = {
        "double_support": COLORS["success"],
        "transfer": COLORS["push"],
        "single_support": COLORS["right_foot"],
        "landing": COLORS["left_foot"],
        "failed_recovery": COLORS["failure"],
    }
    for ax in axes[:3]:
        for left, right, mode in mode_spans:
            ax.axvspan(left, right, color=mode_colors.get(mode, COLORS["grid"]), alpha=0.045, linewidth=0)

    axes[3].plot(t, np.asarray(a["qp_solve_time_s"], dtype=float) * 1000.0, color=COLORS["muted"], linewidth=1.0, label="QP solve")
    axes[3].axhline(4.0, color=COLORS["push"], linewidth=0.8, linestyle=(0, (3, 2)), label="4 ms control budget")
    axes[3].set_ylabel("QP [ms]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("solver timing; event labels mark supervisor decisions", loc="left", pad=3)
    axes[3].legend(loc="upper left", ncol=2, handlelength=1.6)
    event_indices = np.flatnonzero(events != "")
    for index in event_indices:
        axes[3].axvline(t[index], color=COLORS["boundary"], linewidth=0.6, alpha=0.55)
        axes[3].text(t[index], axes[3].get_ylim()[1] * 0.82, events[index].replace("_", " "), rotation=90, fontsize=9.0, color=COLORS["ink"], ha="right", va="top")
    for ax in axes:
        style_axes(ax)
    fig.tight_layout(h_pad=0.55)
    return list(_save(fig, output_dir, name))
