"""Animate the production right-invariant spatial SE(3) error."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("VISUAL_OUTPUT_ROOT", ROOT / "results" / "videos"))
sys.path.insert(0, str(ROOT / "src"))
from se3_whole_body_control.geometry.se3 import exp_se3, inverse_se3, log_se3
from se3_whole_body_control.visualization.style import COLORS, apply_style

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def frame(ax, transform, label, label_color):
    origin = transform[:3, 3]
    axis_colors = (COLORS["axis_x"], COLORS["axis_y"], COLORS["axis_z"])
    for index, color in enumerate(axis_colors):
        direction = transform[:3, index]
        ax.quiver(*origin, *direction, color=color, length=0.22, normalize=True, linewidth=2.0, arrow_length_ratio=0.18)
    ax.scatter(*origin, color=label_color, s=18, depthshade=False)
    ax.text(*origin, label, color=label_color, fontsize=12, zorder=10)


def clean_3d_axis(ax):
    ax.set_xlim(-0.48, 0.60); ax.set_ylim(-0.45, 0.48); ax.set_zlim(-0.25, 0.55)
    ax.set_box_aspect((1.15, 1.0, 0.85))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.set_ticks([])
    ax.set_xlabel("x", labelpad=-4); ax.set_ylabel("y", labelpad=-4); ax.set_zlabel("z", labelpad=-4)
    ax.view_init(elev=24, azim=-58)


def main() -> None:
    apply_style()
    desired = exp_se3(np.array([0.0, 0.0, 0.0, 0.12, -0.08, 0.05]))
    error_twist = np.array([0.35, -0.18, 0.12, 0.35, -0.22, 0.28])
    relative = exp_se3(error_twist)
    current = relative @ desired
    error = log_se3(current @ inverse_se3(desired))
    times = np.linspace(0.0, 1.0, 45)
    fig = plt.figure(figsize=(10.4, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.38, 1.0), hspace=0.34, wspace=0.14)
    ax_pose = fig.add_subplot(grid[0, 0], projection="3d")
    ax_error = fig.add_subplot(grid[0, 1], projection="3d")
    ax_twist = fig.add_subplot(grid[1, :])

    def update(index):
        alpha = times[index]
        ax_pose.clear(); ax_error.clear(); ax_twist.clear()
        intermediate = exp_se3(alpha * error) @ desired
        current_error = log_se3(intermediate @ inverse_se3(desired))

        ax_pose.set_title(r"Desired and current body frames", loc="left", pad=7)
        frame(ax_pose, desired, r"$T_d$", COLORS["desired"])
        frame(ax_pose, intermediate, r"$T$", COLORS["push"])
        clean_3d_axis(ax_pose)

        ax_error.set_title(r"Relative error frame $E_s$", loc="left", pad=7)
        frame(ax_error, np.eye(4), r"$I$", COLORS["desired"])
        frame(ax_error, exp_se3(current_error), r"$E_s$", COLORS["wbc"])
        clean_3d_axis(ax_error)

        values = current_error
        positions = np.arange(6)
        bar_colors = [COLORS["axis_x"], COLORS["axis_y"], COLORS["axis_z"]] * 2
        ax_twist.axvspan(-0.5, 2.5, color=COLORS["linear"], alpha=0.10, linewidth=0)
        ax_twist.axvspan(2.5, 5.5, color=COLORS["angular"], alpha=0.08, linewidth=0)
        ax_twist.bar(positions, values, color=bar_colors, width=0.62, edgecolor="white", linewidth=0.5)
        ax_twist.axhline(0.0, color=COLORS["boundary"], linewidth=0.8)
        ax_twist.set_xticks(positions, [r"$v_x$", r"$v_y$", r"$v_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])
        ax_twist.set_xlim(-0.6, 5.6); ax_twist.set_ylim(-0.45, 0.45)
        ax_twist.set_ylabel(r"$\xi_e = \mathrm{Log}(E_s)^\vee$", labelpad=8)
        ax_twist.text(0.25, 1.04, "linear", transform=ax_twist.transAxes, ha="center", color=COLORS["linear"], fontsize=11.0)
        ax_twist.text(0.75, 1.04, "angular", transform=ax_twist.transAxes, ha="center", color=COLORS["angular"], fontsize=11.0)
        ax_twist.text(0.5, 1.04, "world / spatial tangent coordinates  •  [linear | angular]", transform=ax_twist.transAxes, ha="center", fontsize=11.0, color=COLORS["ink"])
        ax_twist.spines["top"].set_visible(False); ax_twist.spines["right"].set_visible(False)
        ax_twist.grid(axis="y", color=COLORS["grid"], alpha=0.42, linewidth=0.6)
        ax_twist.set_axisbelow(True)
        fig.suptitle(f"SE(3) geometric error  •  progress = {alpha:.2f}", fontsize=14.0, y=0.98)
        fig.text(0.5, 0.485, r"$E_s = T\,T_d^{-1}$    $\longrightarrow$    $\xi_e = \mathrm{Log}(E_s)^\vee$", ha="center", va="center", fontsize=13.0, color=COLORS["ink"])
        return []

    animation = FuncAnimation(fig, update, frames=len(times), interval=70, blit=False)
    out = OUTPUT_ROOT; out.mkdir(parents=True, exist_ok=True)
    animation.save(out / "se3_geometry.gif", writer=PillowWriter(fps=15))
    try:
        animation.save(out / "se3_geometry.mp4", writer="ffmpeg", fps=15, dpi=130)
    except Exception as exc:
        (out.parent / "logs").mkdir(parents=True, exist_ok=True)
        (out.parent / "logs" / "se3_geometry_animation.txt").write_text(f"mp4_unavailable={type(exc).__name__}: {exc}\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
