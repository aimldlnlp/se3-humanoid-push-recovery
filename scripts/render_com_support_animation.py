"""Animate measured CoM motion and the double-support polygon."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import ROOT
from se3_whole_body_control.evaluation.metrics import TrialLog


def load_log(path: Path) -> TrialLog:
    log = TrialLog.empty()
    with np.load(path, allow_pickle=False) as data:
        for field in TrialLog.__dataclass_fields__:
            if field in data:
                setattr(log, field, data[field].tolist())
    return log


def main() -> None:
    log = load_log(ROOT / "results" / "data" / "single_push_se3_wbc.npz")
    a = log.arrays(); com = a["com_world"][:, :2]; feet = a["foot_xy_world"].reshape(-1, 2, 2); t = a["time_s"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(com[:, 0].min() - 0.4, com[:, 0].max() + 0.4); ax.set_ylim(com[:, 1].min() - 0.4, com[:, 1].max() + 0.4)
    trajectory, = ax.plot([], [], color="tab:blue", linewidth=2, label="CoM")
    marker, = ax.plot([], [], "o", color="tab:green", label="actual CoM")
    push_marker, = ax.plot([], [], ".", color="tab:purple", label="push interval")
    polygons = [plt.Polygon(np.zeros((4, 2)), alpha=0.2, color=color, label=label) for color, label in (("tab:orange", "left support"), ("tab:red", "right support"))]
    for polygon in polygons: ax.add_patch(polygon)
    ax.legend(loc="best"); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")

    def update(i):
        trajectory.set_data(com[: i + 1, 0], com[: i + 1, 1]); marker.set_data([com[i, 0]], [com[i, 1]])
        if np.linalg.norm(a["push_force"][i, :2]) > 1e-9: push_marker.set_data(com[: i + 1, 0], com[: i + 1, 1])
        else: push_marker.set_data([], [])
        for foot_id, polygon in enumerate(polygons):
            center = feet[i, foot_id] + np.array([0.055, 0.0]); hx, hy = 0.17, 0.12
            polygon.set_xy(np.array([[center[0]-hx, center[1]-hy], [center[0]+hx, center[1]-hy], [center[0]+hx, center[1]+hy], [center[0]-hx, center[1]+hy]]))
        ax.set_title(f"CoM + support polygon  |  t={t[i]:.2f} s")
        return [trajectory, marker, push_marker, *polygons]

    animation = FuncAnimation(fig, update, frames=len(t), interval=35, blit=False)
    out = ROOT / "results" / "videos"; out.mkdir(parents=True, exist_ok=True)
    animation.save(out / "com_support_polygon_animation.gif", writer=PillowWriter(fps=24))
    try:
        animation.save(out / "com_support_polygon_animation.mp4", writer="ffmpeg", fps=24, dpi=130)
    except Exception as exc:
        (ROOT / "results" / "logs" / "com_support_polygon_animation.txt").write_text(f"mp4_unavailable={type(exc).__name__}: {exc}\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
