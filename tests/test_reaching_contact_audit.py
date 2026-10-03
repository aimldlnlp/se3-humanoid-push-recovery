import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_contact_audit import first_event


def test_first_event_ignores_pre_push_and_separates_transient_from_sustained():
    times=np.arange(8)*.01
    mask=np.array([True,True,False,True,False,True,True,True])
    assert first_event(times,mask,.02)==.03
    assert first_event(times,mask,.02,.02)==.05
    assert first_event(times,mask,.02,.03) is None


def test_first_event_no_crossing():
    assert first_event(np.arange(5),np.zeros(5,dtype=bool),0) is None


def test_contact_damping_changes_only_contact_rhs_and_consistent_bias():
    from common import load_configs, make_model, ROOT
    from se3_whole_body_control.config import load_yaml
    from reach_and_balance import ReachingController
    model=make_model(load_configs(ROOT,robot_name='unitree_g1'))
    model.data.qvel[0]=.02
    cfg=load_configs(ROOT,robot_name='unitree_g1')['controller']
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    controller=ReachingController(model,cfg,settings)
    controller._sync_internal_model()
    original=controller._build_problem()
    controller.settings={**settings,'contact_velocity_damping_s_inv':20}
    changed=controller._build_problem()
    for i in (0,1,2):
        a,b=original[i],changed[i]
        np.testing.assert_array_equal(a.toarray() if hasattr(a,'toarray') else a,
                                      b.toarray() if hasattr(b,'toarray') else b)
    correction=20*(original[10]@model.data.qvel)
    for i in (3,4):
        expected=original[i].copy()
        expected[model.nv:model.nv+controller.nw]-=correction
        np.testing.assert_allclose(changed[i],expected,rtol=0,atol=1e-10)
    np.testing.assert_allclose(changed[11],original[11]+correction)
    controller.settings={**settings,'contact_velocity_damping_s_inv':0}
    restored=controller._build_problem()
    np.testing.assert_array_equal(restored[3],original[3])
