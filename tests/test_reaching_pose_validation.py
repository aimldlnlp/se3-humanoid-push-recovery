import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from common import ROOT,load_configs,make_push
from reaching_pose_validation import CASES


def test_validation_plan_has_one_positive_control_and_eight_single_factor_changes():
    assert len(CASES)==9 and len({case[0] for case in CASES})==9
    assert CASES[0]==('control',70,0,3.0)
    baseline=CASES[0][1:]
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    for _,force,direction,start in CASES[1:]:
        assert sum(a!=b for a,b in zip((force,direction,start),baseline))==1
        push=make_push(cfg,magnitude=force,direction_deg=direction,start=start)
        assert push.magnitude_N==force and push.start_time_s==start and push.duration_s==.15
        np.testing.assert_allclose(push.direction_rad,np.deg2rad(direction))
