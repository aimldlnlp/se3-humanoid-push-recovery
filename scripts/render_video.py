"""Render the saved demo trajectory into MP4 and GIF."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_demo


def main() -> None:
    run_demo.main()


if __name__ == "__main__":
    main()
