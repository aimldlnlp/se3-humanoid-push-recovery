"""Frozen factorial pose-component controls with minimum acceleration witnesses."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from common import ROOT,load_configs,make_model,prepare_paired_initial_condition,write_csv
from reaching_task_audit import LABELS,TaskAuditController,com_feasibility,task_component

COMPONENTS=('torso_linear','torso_angular','pelvis_linear','pelvis_angular')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    original=json.loads((ROOT/'results/reaching_contact/task_audit/audit.json').read_text())
    path=ROOT/original['trajectory_path']
    assert hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()==original['trajectory_sha256']
    with np.load(path/'trajectory.npz',allow_pickle=False) as payload:
        arrays={key:payload[key] for key in payload.files}
    summary=json.loads((path/'summary.json').read_text())
    records,flat,costs=[],[],[]
    for old in original['records']:
        i=int(np.argmin(abs(arrays['time_s']-old['time_s'])))
        model=make_model(cfg)
        model.reset(arrays['qpos_history'][0],arrays['qvel_history'][0])
        controller=TaskAuditController(model,cfg['controller'],summary['provenance']['reaching'])
        controller.q_des=arrays['joint_reference_history'][i].copy()
        controller.pd_fallback.q_des=controller.q_des.copy()
        controller.T_des_torso,controller.T_des_pelvis=initial.desired_torso.copy(),initial.desired_pelvis.copy()
        controller.com_des=initial.com_reference.copy()
        model.reset(arrays['qpos_history'][i],arrays['qvel_history'][i])
        model.data.time=float(arrays['time_s'][i])
        result=controller.solve()
        assert result.success,result.message
        np.testing.assert_allclose(result.control,arrays['control'][i],rtol=1e-6,atol=1e-6)
        objectives=dict(zip(LABELS,controller.objectives))
        split=[]
        for component in COMPONENTS:
            task=task_component(objectives,component)
            whole,part=component.rsplit('_',1)
            rows=slice(0,3) if part=='linear' else slice(3,6)
            target=objectives[whole][1][rows]
            residual=task@controller.solution[:task.shape[1]]-target
            weight=objectives[whole][2]
            record=dict(time_s=float(arrays['time_s'][i]),component=component,
                        residual=residual.tolist(),weighted_squared_residual=float(weight*(residual@residual)))
            split.append(record); costs.append(record)
        for whole in ('torso','pelvis'):
            task,target,weight=objectives[whole]
            residual=task@controller.solution[:task.shape[1]]-target
            np.testing.assert_allclose(sum(r['weighted_squared_residual'] for r in split if r['component'].startswith(whole)),weight*(residual@residual))
        controls=[]
        for count in range(5):
            for subset in itertools.combinations(COMPONENTS,count):
                control=com_feasibility(controller,('reach',)+subset,minimize_acceleration=True)
                controls.append(control)
                flat.append(dict(time_s=float(arrays['time_s'][i]),**control))
        records.append(dict(time_s=float(arrays['time_s'][i]),pose_component_costs=split,controls=controls,
                            control_replay_max_error_Nm=float(np.max(abs(result.control-arrays['control'][i])))))
        print(old['time_s'],[(c['preserved_attained_tasks'],c.get('minimum_com_L1_residual'),c.get('max_qdd')) for c in controls],flush=True)
    write_csv(flat,args.output/'controls.csv')
    write_csv(costs,args.output/'component_costs.csv')
    report=dict(records=records,physical_trials_run=0,controller_changed=False,task_weights_changed=False,
                source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                trajectory_path=original['trajectory_path'],trajectory_sha256=original['trajectory_sha256'],
                method='All 16 subsets of four pose output components preserved with attained reaching acceleration. Original point slack fixed. Minimize CoM L1 deviation, then minimum peak generalized acceleration within 1e-9 of optimal L1. No QP tuning.',
                limitation='Instantaneous witnesses only. Linear/angular refer to implemented SE(3) output rows, not uncoupled gain contributions. Infeasible exact fixed-slack controls are not physical infeasibility claims. Peak generalized acceleration combines linear and angular units as in the original bound.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
