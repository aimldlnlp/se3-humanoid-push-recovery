"""Render the reproducible control-and-evaluation architecture diagram."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from se3_whole_body_control.visualization.style import COLORS, apply_style

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, center, width, height, title, subtitle, color, *, dashed=False):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=COLORS["paper"], edgecolor=color, linewidth=1.6,
        linestyle=(0, (4, 3)) if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x, y + 0.02, title, ha="center", va="center", fontsize=10.5, color=COLORS["ink"])
    ax.text(x, y - 0.055, subtitle, ha="center", va="center", fontsize=7.8, color=COLORS["muted"], linespacing=1.25)


def _arrow(ax, start, end, label, color=COLORS["muted"], *, dashed=False, label_offset=(0.0, 0.035)):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.25, color=color,
        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle="arc3,rad=0.0",
    ))
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=7.8, color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, (0.13, 0.72), 0.17, 0.15, "Desired torso / CoM", "pose and regulation targets", COLORS["desired"])
    _box(ax, (0.35, 0.72), 0.17, 0.15, "SE(3) tasks", r"$\xi_e = \mathrm{Log}(E_s)^\vee$", COLORS["wbc"])
    _box(ax, (0.58, 0.72), 0.21, 0.17, "Whole-body QP", r"dynamics + contact + friction", COLORS["wbc"])
    _box(ax, (0.82, 0.72), 0.15, 0.15, "Torques", "actuator limits", COLORS["angular"])
    _box(ax, (0.82, 0.37), 0.22, 0.18, "MuJoCo plant", "floating-base humanoid\nfixed-foot contact", COLORS["actual"])
    _box(ax, (0.53, 0.19), 0.22, 0.16, "Evaluation", "recovery, slip, GRF, basin", COLORS["cop"])

    _arrow(ax, (0.215, 0.72), (0.265, 0.72), "targets", COLORS["desired"])
    _arrow(ax, (0.435, 0.72), (0.475, 0.72), "task acceleration", COLORS["wbc"])
    _arrow(ax, (0.685, 0.72), (0.745, 0.72), r"$\tau$", COLORS["angular"])
    _arrow(ax, (0.82, 0.635), (0.82, 0.47), "control", COLORS["angular"], label_offset=(0.06, 0.0))
    _arrow(ax, (0.70, 0.37), (0.69, 0.63), r"$q,\dot q$, contacts", COLORS["actual"], label_offset=(-0.08, 0.0))
    _arrow(ax, (0.76, 0.32), (0.64, 0.23), "actual GRF", COLORS["cop"], label_offset=(0.0, -0.035))
    _arrow(ax, (0.79, 0.27), (0.90, 0.34), "push disturbance", COLORS["push"], dashed=True, label_offset=(0.0, -0.04))

    ax.text(0.58, 0.58, r"$M(q)\ddot q+h=B\tau+J_c^T\lambda$", ha="center", va="center", fontsize=8.2, color=COLORS["muted"])
    ax.text(0.5, 0.965, "Fixed-foot humanoid push-recovery architecture", ha="center", va="top", fontsize=13, color=COLORS["ink"])
    ax.text(0.5, 0.915, "The primary controller does not receive an oracle copy of the external push", ha="center", va="top", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
    out_png = ROOT / "results" / "figures" / "png" / "system_architecture.png"
    out_pdf = ROOT / "results" / "figures" / "pdf" / "system_architecture.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=COLORS["paper"])
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
