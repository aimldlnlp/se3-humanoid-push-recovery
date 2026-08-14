"""Central research visual language for every static figure and animation."""

from __future__ import annotations

import matplotlib

# Figures and animations are generated on the headless SSH worker and should
# not depend on a local Tk/GUI installation.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# Okabe--Ito-inspired colors.  Semantic names are used throughout the
# project so a controller or physical quantity never changes color by plot.
COLORS = {
    "ink": "#1F2933",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "paper": "#FFFFFF",
    "pd": "#5B6770",
    "wbc": "#0072B2",
    "desired": "#374151",
    "actual": "#0072B2",
    "left_foot": "#E69F00",
    "right_foot": "#CC79A7",
    "com": "#0072B2",
    "cop": "#009E73",
    "push": "#D55E00",
    "boundary": "#4B5563",
    "success": "#0072B2",
    "failure": "#D9DEE5",
    "linear": "#56B4E9",
    "angular": "#D55E00",
    "axis_x": "#D55E00",
    "axis_y": "#009E73",
    "axis_z": "#0072B2",
}


def apply_style() -> None:
    """Apply the shared, portable, publication-oriented Matplotlib style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.42,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "figure.facecolor": COLORS["paper"],
        "axes.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_axes(ax, *, grid: bool = True) -> None:
    """Apply small axes-level details consistently after plotting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color=COLORS["grid"], alpha=0.42, linewidth=0.6)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    """Place a restrained publication-style panel label."""
    ax.text(
        0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10.5, color=COLORS["ink"], fontweight="normal",
    )
