"""Portable, shared typography for every project visualization.

The architecture figure is typeset with the Latin Modern Roman / Computer
Modern family.  This module makes the same family available to Matplotlib
and Pillow without relying on a machine-wide font installation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

FONT_FAMILY = "Latin Modern Roman"
_FONT_ROOT = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "latin-modern"

FontStyle = Literal["normal", "italic"]
FontWeight = Literal["normal", "bold"]

_FONT_FILES = {
    ("normal", "normal"): "lmroman10-regular.otf",
    ("italic", "normal"): "lmroman10-italic.otf",
    ("normal", "bold"): "lmroman10-bold.otf",
    ("italic", "bold"): "lmroman10-bolditalic.otf",
}


def _normalise_style(style: str | None) -> FontStyle:
    return "italic" if str(style or "normal").lower() in {"italic", "oblique"} else "normal"


def _normalise_weight(weight: str | int | None) -> FontWeight:
    if isinstance(weight, int):
        return "bold" if weight >= 600 else "normal"
    return "bold" if str(weight or "normal").lower() in {"bold", "semibold", "heavy", "black"} else "normal"


@lru_cache(maxsize=None)
def font_path(*, style: str = "normal", weight: str | int = "normal") -> Path:
    """Return a bundled Latin Modern OTF path and fail loudly if absent."""
    key = (_normalise_style(style), _normalise_weight(weight))
    path = _FONT_ROOT / _FONT_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled visualization font is missing: {path}. "
            "Restore assets/fonts/latin-modern before rendering artifacts."
        )
    return path


@lru_cache(maxsize=1)
def register_matplotlib_fonts() -> str:
    """Register bundled faces with Matplotlib for the current process."""
    from matplotlib import font_manager

    # Matplotlib may already know a machine-wide Latin Modern installation.
    # Remove that family before registering the repository-owned faces so
    # ``findfont`` cannot silently select a different version on the worker.
    bundled_paths = {
        font_path(style=style, weight=weight).resolve()
        for style, weight in (
            ("normal", "normal"),
            ("italic", "normal"),
            ("normal", "bold"),
            ("italic", "bold"),
        )
    }
    font_manager.fontManager.ttflist = [
        entry
        for entry in font_manager.fontManager.ttflist
        if entry.name != FONT_FAMILY or Path(entry.fname).resolve() in bundled_paths
    ]
    for style, weight in (("normal", "normal"), ("italic", "normal"), ("normal", "bold"), ("italic", "bold")):
        font_manager.fontManager.addfont(str(font_path(style=style, weight=weight)))
    return FONT_FAMILY


def pil_font(size: int, *, style: str = "normal", weight: str | int = "normal"):
    """Load a bundled face for Pillow overlays."""
    from PIL import ImageFont

    return ImageFont.truetype(str(font_path(style=style, weight=weight)), size=max(1, int(size)))
