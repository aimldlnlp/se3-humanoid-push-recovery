"""Animate the SE(3) error construction without a simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from se3_whole_body_control.geometry.se3 import exp_se3, inverse_se3, log_se3


def frame(ax, T, label, color):
    origin = T[:3, 3]
    for i, axis in enumerate(("x", "y", "z")):
        direction = T[:3, i]
        ax.quiver(*origin, *direction, color=("tab:red", "tab:green", "tab:blue")[i], length=0.22, normalize=True)
    ax.text(*origin, label, color=color)


def main() -> None:
    desired = exp_se3(np.array([0.0, 0.0, 0.0, 0.12, -0.08, 0.05]))
    error_twist = np.array([0.35, -0.18, 0.12, 0.35, -0.22, 0.28])
    relative = exp_se3(error_twist)
    current = relative @ desired
    error = log_se3(current @ inverse_se3(desired))
    times = np.linspace(0.0, 1.0, 45)
    fig = plt.figure(figsize=(12, 7))
    ax_pose = fig.add_subplot(221, projection="3d")
    ax_error = fig.add_subplot(222, projection="3d")
    ax_twist = fig.add_subplot(212)

    def update(index):
        alpha = times[index]
        ax_pose.clear(); ax_error.clear(); ax_twist.clear()
        intermediate = exp_se3(alpha * error) @ desired
        ax_pose.set_title(r"$T_d$ and $T$")
        frame(ax_pose, desired, r"$T_d$", "black")
        frame(ax_pose, intermediate, r"$T$", "tab:orange")
        ax_pose.set_xlim(-0.5, 0.6); ax_pose.set_ylim(-0.5, 0.5); ax_pose.set_zlim(-0.3, 0.6)
        ax_pose.set_xlabel("x"); ax_pose.set_ylabel("y"); ax_pose.set_zlabel("z")
        current_error = log_se3(intermediate @ inverse_se3(desired))
        ax_error.set_title(r"$E=T_d^{-1}T$ in the chosen left/spatial convention")
        frame(ax_error, np.eye(4), "I", "black")
        frame(ax_error, exp_se3(current_error), "E", "tab:purple")
        ax_error.set_xlim(-0.5, 0.6); ax_error.set_ylim(-0.5, 0.5); ax_error.set_zlim(-0.3, 0.6)
        ax_error.set_xlabel("linear tangent x"); ax_error.set_ylabel("linear tangent y"); ax_error.set_zlabel("linear tangent z")
        ax_twist.bar(np.arange(6), current_error, color=["tab:blue"] * 3 + ["tab:orange"] * 3)
        ax_twist.axhline(0.0, color="black", linewidth=0.8)
        ax_twist.set_xticks(np.arange(6), ["v_x", "v_y", "v_z", "ω_x", "ω_y", "ω_z"])
        ax_twist.set_ylim(-0.45, 0.45)
        ax_twist.set_ylabel(r"$\mathrm{Log}(E)^\vee$")
        ax_twist.set_title(r"$\xi_e = \mathrm{Log}(E)^\vee$; translation first, rotation second")
        fig.suptitle(f"SE(3) geometry animation  |  progress={alpha:.2f}")
        fig.tight_layout()

    animation = FuncAnimation(fig, update, frames=len(times), interval=70, blit=False)
    out = ROOT / "results" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    animation.save(out / "se3_geometry_animation.gif", writer=PillowWriter(fps=15))
    try:
        animation.save(out / "se3_geometry_animation.mp4", writer="ffmpeg", fps=15, dpi=130)
    except Exception as exc:
        (ROOT / "results" / "logs" / "se3_geometry_animation.txt").write_text(f"mp4_unavailable={type(exc).__name__}: {exc}\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
