import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT,load_configs,make_model,prepare_paired_initial_condition
from reaching_task_audit import TaskAuditController,objective_records,active_constraints,com_feasibility
from se3_whole_body_control.config import load_yaml


def test_task_audit_reconstructs_exact_objective_and_feasible_com_controls():
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    model=make_model(cfg)
    model.reset(initial.qpos,initial.qvel)
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    settings['contact_velocity_damping_s_inv']=20
    controller=TaskAuditController(model,cfg['controller'],settings)
    result=controller.solve()
    assert result.success,result.message
    records,cost,regularization=objective_records(controller)
    assert cost>=0 and regularization>=0
    assert {r['task'] for r in records} >= {'torso','pelvis','com','posture','reach','nominal_torque','point_slack_0'}
    np.testing.assert_allclose(sum(r['weighted_squared_residual'] for r in records)+regularization,cost)
    constraints=active_constraints(controller)
    assert len(constraints)==controller.problem[2].shape[0]
    diagnostic=com_feasibility(controller,())
    assert diagnostic['success'],diagnostic['status']
    assert diagnostic['constraint_budget_ratio']<=1
