"""Frozen normal/tangent constraint controls, not a contact policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from common import ROOT,load_configs,prepare_paired_initial_condition,write_csv
from reaching_failure_onset import snapshot
from reaching_pose_origin import PoseOriginController
from reaching_task_audit import constraint_labels


def selected_rows(labels,foot,release):
    if release not in ('none','normal','tangent','all'):
        raise ValueError('unknown release mode')
    rows=[i for i,name in enumerate(labels) if name==f'point_acceleration_{foot}']
    assert len(rows)%3==0
    return [row for axis,row in enumerate(rows) if release=='all' or
            (release=='normal' and axis%3==2) or (release=='tangent' and axis%3!=2)]


class ConstraintSplitController(PoseOriginController):
    def __init__(self,model,cfg,settings,foot,release):
        super().__init__(model,cfg,settings)
        self.selected_foot,self.release=foot,release

    def _build_problem(self):
        problem=list(super()._build_problem())
        # In this horizontal-plane scene, world z is normal and x/y tangent.
        ground=self.internal_model.geom_ids['ground']
        np.testing.assert_allclose(abs(self.internal_model.data.geom_xmat[ground].reshape(3,3)[:,2]),[0,0,1],atol=1e-10)
        self.relaxed_point_rows=selected_rows(constraint_labels(self),self.selected_foot,self.release)
        self.original_problem=tuple(problem)
        problem[3],problem[4]=problem[3].copy(),problem[4].copy()
        problem[3][self.relaxed_point_rows]=-np.inf
        problem[4][self.relaxed_point_rows]=np.inf
        self.problem=tuple(problem)
        return self.problem


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    old=json.loads((ROOT/'results/reaching_contact/failure_onset_audit/audit.json').read_text())
    # Predeclared experimental feet, not a load threshold or adaptive policy.
    choices={'control':1,'direction_45':1,'direction_90':1,'direction_270':0}
    records,flat=[],[]
    for case in old['records']:
        if case['case'] not in choices:
            continue
        foot=choices[case['case']]
        path=ROOT/case['trajectory_path']
        assert hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()==case['trajectory_sha256']
        with np.load(path/'trajectory.npz',allow_pickle=False) as payload:
            a={k:payload[k] for k in payload.files}
        summary=json.loads((path/'summary.json').read_text())
        for old_sample in case['snapshots']:
            controls=[]
            for mode in ('none','normal','tangent','all'):
                factory=lambda model,config,settings: ConstraintSplitController(model,config,settings,foot,mode)
                result=snapshot(path,a,summary,old_sample['index'],cfg,initial,factory,verify_control=mode=='none')
                result['release_mode']=mode
                result['observed_data_source']='Original saved trajectory, not a counterfactual rollout'
                if result['success']:
                    result['control_difference_from_saved_Nm']=result.pop('replay_control_max_error_Nm')
                    result['wrench_difference_from_saved']=result.pop('replay_wrench_max_error')
                controls.append(result)
                if result['success']:
                    f=result['feet'][foot]
                    row=dict(case=case['case'],time_s=result['time_s'],foot=f['foot'],release=mode,
                             point_count=f['point_count'],relaxed_rows=len(result['relaxed_point_rows']),
                             control_difference_from_saved_Nm=result['control_difference_from_saved_Nm'],
                             constraint_budget_ratio=result['constraint_budget_ratio'],
                             force_prediction_error_N=f['force_prediction_error_N'],
                             acceleration_difference_from_saved_interval=f['point_acceleration_prediction_interval_difference_norm'],
                             point_target_residual=f['predicted_point_target_residual_norm'],
                             predicted_normal_force_N=f['predicted_wrench_world'][2],
                             predicted_inner_cone_residual=f['predicted_inner_cone_residual'],
                             max_qdd=result['max_qdd'],torque_peak_Nm=result['torque_peak_Nm'],
                             objective_sum=result['objective_sum'])
                    flat.append(row)
            records.append(dict(case=case['case'],foot=foot,index=old_sample['index'],time_s=old_sample['time_s'],
                                trajectory_path=case['trajectory_path'],trajectory_sha256=case['trajectory_sha256'],controls=controls))
        print(case['case'],'counterfactuals complete',flush=True)
    write_csv(flat,args.output/'controls.csv')
    report=dict(records=records,source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                physical_trials_run=0,production_controller_changed=False,gains_changed=False,force_cones_changed=False,
                method='Four modes per frozen state: retain all, relax normal, relax tangential, relax all point-kinematic equalities on a predeclared foot. Same objective, force constraints, torque/acceleration bounds and numerical budget.',
                limitation='Saved observed accelerations/wrenches belong to baseline, not counterfactual rollouts. Relaxed constraints can permit force/motion combinations that violate physical contact complementarity; no policy or load threshold is proposed. Norms concatenate all point components; empty-foot zero is not agreement evidence.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
