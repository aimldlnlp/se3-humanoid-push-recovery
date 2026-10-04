import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT,load_configs,make_model,prepare_paired_initial_condition
from reaching_constraint_split import selected_rows,ConstraintSplitController
from se3_whole_body_control.config import load_yaml


def test_rows_split_world_normal_and_tangent_only_on_selected_foot():
    labels=['dynamics']+['point_acceleration_0']*6+['ray_nonnegative_1']+['point_acceleration_1']*3
    assert selected_rows(labels,0,'normal')==[3,6]
    assert selected_rows(labels,0,'tangent')==[1,2,4,5]
    assert selected_rows(labels,1,'all')==[8,9,10]
    assert selected_rows(labels,1,'none')==[]
    assert selected_rows(labels,2,'normal')==[]
    with pytest.raises(ValueError):
        selected_rows(labels,0,'invalid')


def test_counterfactual_preserves_objective_matrix_and_all_other_constraints():
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    model=make_model(cfg)
    model.reset(initial.qpos,initial.qvel)
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    settings['contact_velocity_damping_s_inv']=20
    controller=ConstraintSplitController(model,cfg['controller'],settings,1,'normal')
    result=controller.solve()
    assert result.success,result.message
    before,after=controller.original_problem,controller.problem
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1],after[1])
    np.testing.assert_array_equal(before[2].toarray(),after[2].toarray())
    assert controller.relaxed_point_rows
    keep=np.ones(len(before[3]),dtype=bool)
    keep[controller.relaxed_point_rows]=False
    np.testing.assert_array_equal(before[3][keep],after[3][keep])
    np.testing.assert_array_equal(before[4][keep],after[4][keep])
    assert np.all(np.isneginf(after[3][~keep])) and np.all(np.isposinf(after[4][~keep]))
