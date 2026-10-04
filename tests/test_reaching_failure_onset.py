import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_failure_onset import event_start, events
from se3_whole_body_control.evaluation.recovery import RecoveryConfig


def test_onset_uses_run_start_and_does_not_join_interrupted_events():
    t=np.arange(8)*.01
    mask=np.array([False,True,True,False,True,True,True,True])
    assert event_start(t,mask)==1
    assert event_start(t,mask,.025)==4
    assert event_start(t,mask,.04)==-1
    assert event_start(t,np.zeros(8,dtype=bool))==-1


def test_post_step_events_use_interval_end_and_do_not_claim_sustained_loss():
    a=dict(time_s=np.array([2.996,3.0,3.004,3.008]),
           foot_tangent_velocity_post_step=np.zeros((4,2)),
           foot_xy_displacement_post_step=np.zeros((4,2)),
           actual_friction_utilization_post_step=np.zeros((4,2)),
           contact_left_post_step=np.array([True,False,True,True]),
           contact_right_post_step=np.ones(4,dtype=bool),
           com_world=np.zeros((4,3)),torso_height_m=np.ones(4))
    a['foot_tangent_velocity_post_step'][1,0]=.1
    summary=dict(provenance=dict(push_start_s=3.0,com_reference_world=[0,0,0]))
    result=events(a,summary,RecoveryConfig())
    assert result['contact_loss_first']['index']==1
    assert result['contact_loss_first']['time_s']==3.004
    assert result['slip_speed_first']['time_s']==3.004
    assert result['contact_loss_sustained'] is None
    assert result['slip_sustained'] is None
