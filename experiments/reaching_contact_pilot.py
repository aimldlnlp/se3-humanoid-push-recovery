"""One frozen contact-velocity damping candidate on three known conditions."""
import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT, write_csv
from reaching_tracking_study import verify_pair
from reaching_workspace import run_trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pose-stabilization',action='store_true')
    args=parser.parse_args()
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir()
    rows=[]
    for phase,force,start in (('no_push',0,1.5),('moving',70,1.5),('hold',70,3.0)):
        name=f'front_12.5cm_contact_{"pose" if args.pose_stabilization else "damping"}_{phase}'
        row=run_trial(root,name,[.125,0,0],force,start,stage='exploratory_pilot',
                      reach_weight=3000,contact_velocity_damping=20,
                      contact_pose_stiffness=100 if args.pose_stabilization else 0)
        suffix='' if phase=='no_push' else f'_{phase}_70N'
        baseline=ROOT/'results/reaching_tracking/fbc0b50_local/trials'/f'front_12.5cm_w3000{suffix}'
        if args.pose_stabilization:
            baseline=ROOT/'results/reaching_contact/damping_pilot/trials'/f'front_12.5cm_contact_damping_{phase}'
        verify_pair(baseline,root/'trials'/name)
        row.update(phase=phase,contact_velocity_damping_s_inv=20,baseline=baseline.relative_to(ROOT).as_posix())
        row['contact_pose_stiffness_s_inv2']=100 if args.pose_stabilization else 0
        rows.append(row)
        write_csv(rows,root/'study.csv')
        print(phase,row['failure_reason'],row['hold_max_error_mm'],flush=True)
    report=dict(trials=rows,pilot_gate_passed=all(r['combined_success'] for r in rows),
                contact_velocity_damping_s_inv=20,paired_inputs_verified=True,
                contact_pose_stiffness_s_inv2=100 if args.pose_stabilization else 0,
                scope='One exploratory candidate; no gain search, validation expansion, or default promotion',
                physical_thresholds_changed=False,objective_weights_changed=False,
                contact_rhs_changed=True,source_version=rows[0]['source_version'])
    (root/'study.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
