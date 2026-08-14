"""Animate measured CoM motion with the active double-support region."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from common import ROOT
from se3_whole_body_control.evaluation.metrics import TrialLog
from se3_whole_body_control.visualization.plots import _active_support_hull, _finite_cop, _foot_support_vertices
from se3_whole_body_control.visualization.style import COLORS, apply_style, style_axes

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


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
    apply_style()
    log = load_log(ROOT / "results" / "data" / "single_push_se3_wbc.npz")
    a = log.arrays()
    com = a["com_world"][:, :2]
    time_s = a["time_s"]
    vertices = _foot_support_vertices(a)
    active = np.column_stack([a["contact_left"], a["contact_right"]]).astype(bool)

    fps = 30
    sample_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0 / fps
    duration_s = float(time_s[-1] + sample_dt) if len(time_s) else 0.0
    frame_times = np.linspace(0.0, max(float(time_s[-1]), 0.0), max(1, int(round(duration_s * fps))))
    frame_indices = _nearest_indices(time_s, frame_times)

    finite_points = np.concatenate([com.reshape(-1, 2), vertices.reshape(-1, 2)], axis=0)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    lo = finite_points.min(axis=0); hi = finite_points.max(axis=0)
    pad = max(0.025, 0.08 * float(np.max(hi - lo)))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    style_axes(ax)
    trail, = ax.plot([], [], color=COLORS["com"], linewidth=2.0, label="CoM trail")
    current, = ax.plot([], [], "o", color=COLORS["com"], markersize=7, label="actual CoM")
    push_trail, = ax.plot([], [], color=COLORS["push"], linewidth=3.0, label="push interval")
    cop_history = [
        ax.scatter([], [], color=COLORS["cop"], alpha=0.60, s=14, marker=".", label="measured CoP" if foot == 0 else "_")
        for foot in (0, 1)
    ]
    cop_current = ax.scatter([], [], color=COLORS["cop"], marker=".", s=45, zorder=7)
    foot_polygons = [
        plt.Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.16, label=label)
        for color, label in ((COLORS["left_foot"], "left foot"), (COLORS["right_foot"], "right foot"))
    ]
    for polygon in foot_polygons:
        ax.add_patch(polygon)
    support_polygon = plt.Polygon(np.zeros((3, 2)), closed=True, facecolor=COLORS["wbc"], edgecolor=COLORS["wbc"], linewidth=1.5, alpha=0.09, label="double-support hull")
    ax.add_patch(support_polygon)
    peak_index = int(np.argmax(np.linalg.norm(com - com[0], axis=1)))
    initial = ax.scatter([com[0, 0]], [com[0, 1]], color=COLORS["desired"], marker="o", s=34, label="initial / nominal")
    peak = ax.scatter([com[peak_index, 0]], [com[peak_index, 1]], color=COLORS["push"], marker="^", s=42, label="peak")
    final = ax.scatter([com[-1, 0]], [com[-1, 1]], color=COLORS["actual"], marker="s", s=34, label="final")
    ax.set_xlabel("World x [m]"); ax.set_ylabel("World y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, columnspacing=0.9)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.91)

    def update(frame_number):
        sample = int(frame_indices[frame_number])
        trail.set_data(com[: sample + 1, 0], com[: sample + 1, 1])
        current.set_data([com[sample, 0]], [com[sample, 1]])
        push_mask = np.linalg.norm(a["push_force"][: sample + 1, :2], axis=1) > 1e-9
        push_trail.set_data(com[: sample + 1][push_mask, 0], com[: sample + 1][push_mask, 1])
        for foot_id, polygon in enumerate(foot_polygons):
            foot = vertices[sample, foot_id]
            polygon.set_xy(np.vstack([foot, foot[0]]))
            polygon.set_alpha(0.16 if active[sample, foot_id] else 0.04)
        hull = _active_support_hull(vertices[sample], active[sample])
        if len(hull) >= 3:
            support_polygon.set_xy(np.vstack([hull, hull[0]])); support_polygon.set_visible(True)
        else:
            support_polygon.set_visible(False)
        current_cop = []
        for foot_id, history in enumerate(cop_history):
            cop = _finite_cop(a, foot_id)
            if len(cop):
                # The logged CoP arrays contain only contact-valid samples;
                # use the nearest timestamp prefix for a restrained trail.
                raw = np.asarray(a["foot_cop_world"], dtype=float).reshape(-1, 2, 2)[: sample + 1, foot_id]
                valid = raw[np.all(np.isfinite(raw), axis=1)]
                stride = max(1, len(valid) // 18)
                history.set_offsets(valid[::stride] if len(valid) else np.empty((0, 2)))
                if len(valid):
                    current_cop.append(valid[-1])
            else:
                history.set_offsets(np.empty((0, 2)))
        cop_current.set_offsets(np.asarray(current_cop) if current_cop else np.empty((0, 2)))
        push_active = bool(np.linalg.norm(a["push_force"][sample, :2]) > 1e-9)
        ax.set_title(f"CoM + support  |  t = {time_s[sample]:.2f} s  |  {'PUSH' if push_active else 'idle'}", loc="left", fontsize=10.5, pad=9)
        return [trail, current, push_trail, cop_current, *cop_history, *foot_polygons, support_polygon]

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000.0 / fps, blit=False)
    out = ROOT / "results" / "videos"; out.mkdir(parents=True, exist_ok=True)
    animation.save(out / "com_support_polygon.gif", writer=PillowWriter(fps=fps))
    try:
        animation.save(out / "com_support_polygon.mp4", writer="ffmpeg", fps=fps, dpi=120)
    except Exception as exc:
        (ROOT / "results" / "logs" / "com_support_polygon.txt").write_text(f"mp4_unavailable={type(exc).__name__}: {exc}\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
