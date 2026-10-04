"""Counterfactual frozen-state contact-mode gate; not a physical recovery test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from common import ROOT,load_configs,prepare_paired_initial_condition
from reaching_contact_mode import ContactModeController
from reaching_wrench_audit import snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    original=json.loads((ROOT/'results/reaching_contact/wrench_audit_lp/audit.json').read_text())
    records=[]
    # Only damping trajectories: do not combine the rejected pose candidate.
    for case in original['records']:
        if not case['case'].startswith('damping_'):
            continue
        path=ROOT/case['path']
        assert hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest()==case['trajectory_sha256']
        for sample in case['snapshots']:
            baseline=snapshot(path,sample['time_s'],cfg,initial)
            candidate=snapshot(path,sample['time_s'],cfg,initial,ContactModeController,verify_control=False)
            passed=(candidate['success'] and candidate['constraint_budget_ratio']<=1
                    and all(f['predicted_outer_cone_relative_residual']<=1e-5 for f in candidate.get('feet',[])))
            # Outer-cone measured wrench is a positive control, not proof of
            # feasibility in the candidate's more conservative inner cone.
            measured_ok=all(f['measured_outer_cone_relative_residual']<=1e-5 for f in baseline['feet'])
            record=dict(case=case['case'],baseline=baseline,counterfactual=candidate,
                        frozen_gate_passed=bool(passed and measured_ok))
            records.append(record)
            print(case['case'],sample['time_s'],record['frozen_gate_passed'],candidate.get('message',''),flush=True)
    report=dict(records=records,frozen_gate_passed=all(r['frozen_gate_passed'] for r in records),
                physical_trials_run=0,source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                gate='All sampled QPs accepted by unchanged row-wise numerical budget; wrench fits reconstructed outer cone at 1e-5 normalized residual; original measured wrench positive control fits same outer cone',
                limitation='Frozen counterfactual only. Original measured/observed accelerations in candidate records are baseline data, not candidate rollout predictions. Geometric contacts may be unloaded. No touchdown hysteresis or force-bearing mode estimator.',
                slack_penalty_definition='Original scalar penalty, averaged across point acceleration constraints per foot; differs from original linear/angular body slack metric')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
