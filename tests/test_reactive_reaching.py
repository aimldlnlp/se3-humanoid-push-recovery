"""Continuity and endpoint checks for a moving-target redirection."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from reactive_reaching import segment, sample


def test_retarget_preserves_position_velocity_acceleration():
    start = np.array([[.1, -.2, .3], [.05, -.02, .01], [.02, .03, -.04]])
    first = segment(start, [.2, -.25, .4], 2)
    moving = sample(first, .8, 2)
    redirected = segment(moving, [.12, -.28, .35], 2)
    np.testing.assert_allclose(sample(redirected, 0, 2), moving, atol=1e-14)
    np.testing.assert_allclose(sample(redirected, 2, 2), [[.12, -.28, .35], [0,0,0], [0,0,0]], atol=1e-14)
    np.testing.assert_allclose(sample(redirected, 3, 2), sample(redirected, 2, 2), atol=1e-14)
    epsilon = 1e-6
    left, right = sample(first,.8-epsilon,2), sample(redirected,epsilon,2)
    np.testing.assert_allclose((right[0]-left[0])/(2*epsilon), moving[1], atol=1e-8)
    np.testing.assert_allclose((right[1]-left[1])/(2*epsilon), moving[2], atol=1e-6)
    for bad_duration in (0,-1,float('nan')):
        try:
            segment(start,[0,0,0],bad_duration)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid duration accepted')


def test_sequence_assessment_detects_late_fall_and_skips_cancelled_goal():
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_reactive_reaching import EVENTS, assess, load_configs, ROOT
    time = np.arange(4000)*.004
    count = len(time)
    points = np.zeros((count,3))
    for event in EVENTS:
        points[time>=event['time']] = event['offset']
    arrays = dict(time_s=time, com_world=np.zeros((count,3)), torso_rotation_error_rad=np.zeros(count),
        torso_angular_velocity_norm=np.zeros(count), contact_left_post_step=np.ones(count,dtype=bool),
        contact_right_post_step=np.ones(count,dtype=bool), torso_height_m=np.ones(count),
        torque_abs_max_Nm=np.zeros(count), qp_success=np.ones(count,dtype=bool),
        actual_friction_margin=np.ones(count), actual_friction_utilization_post_step=np.zeros((count,2)),
        foot_tangent_velocity_post_step=np.zeros((count,2)), foot_xy_displacement_post_step=np.zeros((count,2)),
        torque_utilization=np.zeros(count), joint_limit_violation=np.zeros(count,dtype=bool),
        numerical_valid=np.ones(count,dtype=bool), reach_point_world=points, reach_reference_world=points)
    initial = SimpleNamespace(com_reference=np.zeros(3))
    config = load_configs(ROOT, robot_name='unitree_g1')
    result = assess(arrays,initial,config,EVENTS,np.zeros(3),16)
    assert result['combined_success']
    assert [w['name'] for w in result['waypoint_results']] == ['A','B','C','B_retarget']
    arrays['torso_height_m'][time>14] = .2
    result = assess(arrays,initial,config,EVENTS,np.zeros(3),16)
    assert not result['combined_success'] and result['failure_reason']=='FALL'
