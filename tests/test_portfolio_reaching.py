import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
import reaching_workspace


def test_portfolio_initial_state_is_forwarded_without_enabling_hierarchy(tmp_path, monkeypatch):
    commands = []
    (tmp_path/'logs').mkdir()
    summary = dict(reach_success=True, balance_success=True, final_goal_error_m=.004,
                   hold_max_goal_error_m=.005, tracking_rmse_m=.006,
                   recovery=dict(max_joint_torque_Nm=22), max_foot_displacement_m=.007,
                   max_foot_tangent_velocity_m_s=.02, continuous_double_support=True,
                   qp_failures=0, qp_deadline_miss_percent=100,
                   provenance=dict(initial_condition=dict(sha256='saved-state'), source_version='test'))
    monkeypatch.setattr(reaching_workspace.subprocess, 'run',
                        lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(reaching_workspace, 'assess_trial', lambda path: (summary, True, 'PASS'))
    state = tmp_path/'initial.npz'
    row = reaching_workspace.run_trial(
        tmp_path, 'corrected_hold', [.125, 0, 0], 70, 3,
        reach_weight=3000, contact_velocity_damping=20, contact_mode=True,
        pose_origin=True, initial_condition_path=state)
    command = commands[0]
    assert command[command.index('--initial-condition')+1] == str(state)
    assert '--pose-origin' in command and '--contact-mode' in command
    assert '--hierarchical' not in command
    assert row['combined_success'] and row['initial_condition_sha256']=='saved-state'
