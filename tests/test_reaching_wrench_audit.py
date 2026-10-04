import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_wrench_audit import cone_residual, wrench_generators


def test_point_contact_rejects_unrealizable_pitch_but_allows_origin_moment():
    point=np.array([.1, 0, 0])
    origin=np.array([0, 0, .05])
    generators=wrench_generators([point], [np.eye(3)[[2,0,1]]], [np.array([.7,.7,0,0,0])], [3], origin)
    force=np.array([10, 0, 100])
    wrench=np.r_[force, np.cross(point-origin, force)]
    assert cone_residual(generators, wrench)<1e-10
    wrench[4]+=5
    assert cone_residual(generators, wrench)>.01


def test_contact_spin_and_roll_dimensions_expand_cone():
    frame=np.eye(3)[[2,0,1]]
    mu=np.array([.7,.7,.02,.01,.01])
    wrench=np.array([0,0,100,1,0,2])
    for dimension, possible in ((3,False),(6,True)):
        generators=wrench_generators([np.zeros(3)], [frame], [mu], [dimension], np.zeros(3))
        assert (cone_residual(generators,wrench)<1e-10)==possible


def test_patch_moment_rows_include_height_and_preserve_disabled_behavior():
    from common import ROOT, load_configs, make_model, prepare_paired_initial_condition
    from reach_and_balance import ReachingController
    from se3_whole_body_control.config import load_yaml
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    model=make_model(cfg)
    model.reset(initial.qpos,initial.qvel)
    settings=load_yaml(ROOT/'configs/reaching.yaml')
    controller=ReachingController(model,cfg['controller'],settings)
    def constraints():
        rows,low,high=[],[],[]
        controller._friction_rows(rows,low,high,0)
        return np.array(rows),np.array(low),np.array(high)
    original=constraints()
    controller.settings={**settings,'contact_patch_bounds':True}
    changed=constraints()
    np.testing.assert_array_equal(changed[1],original[1])
    np.testing.assert_array_equal(changed[2],original[2])
    for foot in range(2):
        generators,_,_=__import__('reaching_wrench_audit').contact_cone(model,foot)
        assert generators.shape[1]>0
        for column in generators.T:
            wrench=np.zeros(controller.nx)
            wrench[6*foot:6*foot+6]=column
            values=changed[0]@wrench
            for i in range(11*foot+5,11*foot+9):
                assert changed[1][i]-1e-10<=values[i]<=changed[2][i]+1e-10
        assert changed[0][11*foot+5,6*foot+1]!=0
    controller.settings=settings
    restored=constraints()
    np.testing.assert_array_equal(restored[0],original[0])
