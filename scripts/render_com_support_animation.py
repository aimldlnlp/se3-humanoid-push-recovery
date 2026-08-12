"""Animate measured CoM motion and the active double-support polygon."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from common import ROOT
from se3_whole_body_control.evaluation.metrics import TrialLog
from se3_whole_body_control.visualization.plots import _active_support_hull, _foot_support_vertices


def load_log(path: Path) -> TrialLog:
    log = TrialLog.empty()
    with np.load(path, allow_pickle=False) as data:
        for field in TrialLog.__dataclass_fields__:
            if field in data:
                setattr(log, field, data[field].tolist())
    return log


def _nearest_indices(time_s: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(time_s, frame_times, side="left")
    right = np.clip(right, 0, len(time_s) - 1)
    left = np.clip(right - 1, 0, len(time_s) - 1)
    choose_left = np.abs(time_s[left] - frame_times) <= np.abs(time_s[right] - frame_times)
    return np.where(choose_left, left, right)


def main() -> None:
    log = load_log(ROOT / "results" / "data" / "single_push_se3_wbc.npz")
    a = log.arrays()
    com = a["com_world"][:, :2]
    time_s = a["time_s"]
    vertices = _foot_support_vertices(a)
    active = np.column_stack([a["contact_left"], a["contact_right"]]).astype(bool)
    cop = np.asarray(a.get("foot_cop_world", np.zeros(0)), dtype=float)
    cop = cop.reshape(-1, 2, 2) if cop.size >= 4 else np.full((len(time_s), 2, 2), np.nan)

    fps = 30
    sample_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0 / fps
    duration_s = float(time_s[-1] + sample_dt) if len(time_s) else 0.0
    frame_times = np.linspace(0.0, max(float(time_s[-1]), 0.0), max(1, int(round(duration_s * fps))))
    frame_indices = _nearest_indices(time_s, frame_times)

    finite_points = np.concatenate([com.reshape(-1, 2), vertices.reshape(-1, 2)], axis=0)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    lo = finite_points.min(axis=0) - 0.12
    hi = finite_points.max(axis=0) + 0.12
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    trail, = ax.plot([], [], color="tab:blue", linewidth=2, label="CoM trail")
    current, = ax.plot([], [], "o", color="tab:green", markersize=8, label="actual CoM")
    initial = ax.scatter([com[0, 0]], [com[0, 1]], color="tab:green", marker="o", label="initial / nominal CoM", zorder=5)
    peak_index = int(np.argmax(np.linalg.norm(com - com[0], axis=1)))
    peak = ax.scatter([com[peak_index, 0]], [com[peak_index, 1]], color="tab:red", marker="^", label="peak CoM", zorder=5)
    final = ax.scatter([com[-1, 0]], [com[-1, 1]], color="black", marker="s", label="final CoM", zorder=5)
    push_trail, = ax.plot([], [], ".", color="tab:purple", markersize=3, label="push interval")
    cop_scatter = ax.scatter([], [], color="black", marker="x", s=24, label="measured CoP", zorder=6)
    foot_polygons = [
        plt.Polygon(np.zeros((4, 2)), closed=True, alpha=0.20, color=color, label=label)
        for color, label in (("tab:orange", "left foot"), ("tab:red", "right foot"))
    ]
    for polygon in foot_polygons:
        ax.add_patch(polygon)
    support_polygon = plt.Polygon(np.zeros((3, 2)), closed=True, alpha=0.22, color="tab:cyan", label="double-support convex hull")
    ax.add_patch(support_polygon)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")

    def update(frame_number):
        sample = int(frame_indices[frame_number])
        trail.set_data(com[: sample + 1, 0], com[: sample + 1, 1])
        current.set_data([com[sample, 0]], [com[sample, 1]])
        push_mask = np.linalg.norm(a["push_force"][: sample + 1, :2], axis=1) > 1e-9
        if np.any(push_mask):
            push_trail.set_data(com[: sample + 1][push_mask, 0], com[: sample + 1][push_mask, 1])
        else:
            push_trail.set_data([], [])
        for foot_id, polygon in enumerate(foot_polygons):
            foot = vertices[sample, foot_id]
            polygon.set_xy(np.vstack([foot, foot[0]]))
            polygon.set_alpha(0.20 if active[sample, foot_id] else 0.06)
        hull = _active_support_hull(vertices[sample], active[sample])
        if len(hull) >= 3:
            support_polygon.set_xy(np.vstack([hull, hull[0]]))
            support_polygon.set_visible(True)
        else:
            support_polygon.set_visible(False)
        valid_cop = np.all(np.isfinite(cop[: sample + 1]), axis=2)
        cop_points = cop[: sample + 1][valid_cop]
        cop_scatter.set_offsets(cop_points if len(cop_points) else np.empty((0, 2)))
        push_active = bool(np.linalg.norm(a["push_force"][sample, :2]) > 1e-9)
        ax.set_title(f"CoM + active double-support polygon  |  t={time_s[sample]:.2f} s  |  push={'on' if push_active else 'off'}")
        return [trail, current, push_trail, cop_scatter, *foot_polygons, support_polygon]

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000.0 / fps, blit=False)
    out = ROOT / "results" / "videos"; out.mkdir(parents=True, exist_ok=True)
    animation.save(out / "com_support_polygon.gif", writer=PillowWriter(fps=fps))
    try:
        animation.save(out / "com_support_polygon.mp4", writer="ffmpeg", fps=fps, dpi=130)
    except Exception as exc:
        (ROOT / "results" / "logs" / "com_support_polygon.txt").write_text(f"mp4_unavailable={type(exc).__name__}: {exc}\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
