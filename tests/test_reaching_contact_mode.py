import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT,load_configs,make_model,prepare_paired_initial_condition
from reaching_contact_mode import ContactModeController,contact_spec
from se3_whole_body_control.config import load_yaml


def test_contact_mode_flat_patch_rank_and_force_moment_geometry():
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    model=make_model(cfg)
    model.reset(initial.qpos,initial.qvel)
    spec=contact_spec(model,0,'left_foot')
    assert spec['rank']==6
    assert spec['J'].shape[0]==3*len(spec['points'])
    assert spec['G'].shape[1]==4*len(spec['points'])
    origin=model.body_pose('left_foot')[:3,3]
    for i,point in enumerate(spec['points']):
        G=spec['G'][:,4*i:4*i+4]
        np.testing.assert_allclose(G[3:].T,np.cross(point-origin,G[:3].T))
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    settings['contact_velocity_damping_s_inv']=20
    controller=ContactModeController(model,cfg['controller'],settings)
    result=controller.solve()
    assert result.success,result.message
    assert result.diagnostics['constraint_budget_ratio']<=1
    assert result.diagnostics['contact_motion_ranks']==[6,6]


def test_material_point_constraints_allow_rotation_about_point_and_edge():
    def mapping(r):
        skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        return np.c_[np.eye(3),-skew]
    point=np.array([.1,0,-.05])
    omega=np.array([0,1,0])
    twist=np.r_[-np.cross(omega,point),omega]
    np.testing.assert_allclose(mapping(point)@twist,0,atol=1e-12)
    edge=np.vstack([mapping(point),mapping(point+np.array([0,.1,0]))])
    assert np.linalg.matrix_rank(edge)==5
    np.testing.assert_allclose(edge@twist,0,atol=1e-12)


def test_unsupported_mode_records_fallback_without_crashing(monkeypatch):
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    model=make_model(cfg)
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    controller=ContactModeController(model,cfg['controller'],settings)
    def unsupported(*args):
        raise ValueError('contact-mode prototype supports only condim=3')
    monkeypatch.setattr('reaching_contact_mode.contact_spec',unsupported)
    result=controller.solve()
    assert not result.success
    assert 'condim=3' in result.message
    assert controller.mode_ranks==[[-1,-1]]
