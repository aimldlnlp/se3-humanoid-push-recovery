"""Configuration and repository path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(root: str | Path | None = None, robot_name: str | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else repo_root()
    catalog = load_yaml(base / "configs" / "robot.yaml")
    selected = robot_name or os.environ.get("SE3_ROBOT") or catalog.get("active_robot", "unitree_g1")
    profiles_dir = base / str(catalog.get("profiles_dir", "configs/robots"))
    profile_path = profiles_dir / f"{selected}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"robot profile not found for {selected!r}: {profile_path}")
    robot = load_yaml(profile_path)
    robot["profile_name"] = str(selected)
    robot["profile_path"] = profile_path.relative_to(base).as_posix()
    controller_path = base / str(robot.get("controller_profile", "configs/controller.yaml"))
    return {
        "root": base,
        "robot_catalog": catalog,
        "robot": robot,
        "controller": load_yaml(controller_path),
        "experiments": load_yaml(base / "configs" / "experiments.yaml"),
    }


def resolve_model_path(config: dict[str, Any]) -> Path:
    root = Path(config["root"])
    return root / config["robot"]["model_path"]
