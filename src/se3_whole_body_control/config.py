"""Configuration and repository path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else repo_root()
    return {
        "root": base,
        "robot": load_yaml(base / "configs" / "robot.yaml"),
        "controller": load_yaml(base / "configs" / "controller.yaml"),
        "experiments": load_yaml(base / "configs" / "experiments.yaml"),
    }


def resolve_model_path(config: dict[str, Any]) -> Path:
    root = Path(config["root"])
    return root / config["robot"]["model_path"]
