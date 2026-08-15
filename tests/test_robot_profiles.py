from pathlib import Path

from se3_whole_body_control.config import load_configs, resolve_model_path


def test_primary_profile_is_g1_and_legacy_profile_remains_selectable():
    root = Path(__file__).resolve().parents[1]
    primary = load_configs(root)
    legacy = load_configs(root, robot_name="mini_humanoid")
    assert primary["robot"]["profile_name"] == "unitree_g1"
    assert primary["robot"]["model_variant"] == "g1_29dof_no_hands"
    assert resolve_model_path(primary).name == "scene_push_recovery.xml"
    assert legacy["robot"]["profile_name"] == "mini_humanoid"
    assert resolve_model_path(legacy).name == "mini_humanoid.xml"
