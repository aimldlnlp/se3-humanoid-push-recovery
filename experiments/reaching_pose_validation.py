"""Predeclared disturbance validation of the frozen pose-origin candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from common import ROOT, load_configs, make_push, write_csv
from reaching_tracking_study import verify_pair
from reaching_workspace import run_trial
from se3_whole_body_control.disturbance.push import push_force

# One reproduced positive control and eight one-factor perturbations; no tuning.
CASES = (
    ('control', 70, 0, 3.0),
    ('direction_45', 70, 45, 3.0),
    ('direction_90', 70, 90, 3.0),
    ('direction_180', 70, 180, 3.0),
    ('direction_270', 70, 270, 3.0),
    ('magnitude_60', 60, 0, 3.0),
    ('magnitude_80', 80, 0, 3.0),
    ('timing_early', 70, 0, 2.75),
    ('timing_late', 70, 0, 3.25),
)


def verify_inputs(reference, trial, magnitude, direction, start, control=False):
    """Verify fixed inputs and the intended disturbance before counting outcomes."""
    before = json.loads((reference/'summary.json').read_text())['provenance']
    after = json.loads((trial/'summary.json').read_text())['provenance']
    for key in ('reaching','initial_condition','model_sha256','dependency_versions','objective_weight_overrides'):
        assert before[key] == after[key], key
    assert after['push_N']==magnitude and after['push_start_s']==start
    assert after['push_direction_deg']==direction
    with np.load(reference/'trajectory.npz',allow_pickle=False) as a, np.load(trial/'trajectory.npz',allow_pickle=False) as b:
        for key in ('time_s','reach_reference_world','reach_goal_world','reach_weight_history'):
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        for key in ('qpos_history','qvel_history'):
            np.testing.assert_array_equal(a[key][0],b[key][0],err_msg=key)
        push=make_push(load_configs(ROOT,robot_name='unitree_g1'),magnitude=magnitude,direction_deg=direction,start=start)
        expected=np.array([push_force(push,float(t)) for t in b['time_s']])
        np.testing.assert_allclose(b['push_force'],expected,atol=1e-10,rtol=0)
        maximum_com=float(np.max(np.linalg.norm(b['com_world'][:,:2]-np.array(after['com_reference_world'])[:2],axis=1)))
    if control:
        verify_pair(reference,trial,same_dynamics=True)
    return dict(fixed_inputs_verified=True,push_array_verified=True,maximum_com_xy_m=maximum_com,
                positive_control_dynamics_reproduced=True if control else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir()
    reference=ROOT/'results/reaching_contact/pose_origin_pilot/trials/front_12.5cm_contact_pose_origin_hold'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=dict(cases=[dict(name=n,push_N=f,direction_deg=d,start_s=t) for n,f,d,t in CASES],
              source_version=source,candidate_source_version='a607d70b2c5acbf452154c814213115580ae2e9f',
              target_offset_m=[.125,0,0],reach_weight=3000,contact_velocity_damping_s_inv=20,
              push_duration_s=.15,reference=reference.relative_to(ROOT).as_posix(),
              gate='Positive control reproduces dynamics; all eight new conditions pass unchanged whole-trial, final-balance and reaching criteria. No gain search.',
              scope='Deterministic disturbance validation, not random-seed or payload validation. Panel task deferred until disturbance gate passes.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows=[]
    for case,force,direction,start in CASES:
        name='front_12.5cm_pose_validation_'+case
        row=run_trial(root,name,[.125,0,0],force,start,stage='frozen_disturbance_validation',
                      reach_weight=3000,contact_velocity_damping=20,contact_mode=True,pose_origin=True,
                      push_direction_deg=direction)
        row.update(case=case,push_direction_deg=direction,
                   **verify_inputs(reference,root/'trials'/name,force,direction,start,control=case=='control'))
        rows.append(row)
        write_csv(rows,root/'study.csv')
        print(case,row['failure_reason'],row['maximum_com_xy_m'],flush=True)
    novel=[r for r in rows if r['case']!='control']
    report=dict(trials=rows,new_conditions_count=len(novel),new_conditions_passed=sum(r['combined_success'] for r in novel),
                validation_gate_passed=all(r['combined_success'] for r in rows),fixed_inputs_verified=True,
                positive_control_dynamics_reproduced=True,controller_changed=False,gains_changed=False,
                contact_model_changed=False,physical_thresholds_changed=False,source_version=source,
                candidate_source_version=plan['candidate_source_version'],plan='plan.json',hardware_trials_run=0,
                real_time_deadline_demonstrated=False,production_promoted=False)
    (root/'study.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
