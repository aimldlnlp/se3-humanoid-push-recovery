"""Small qualified PD pilot against identical saved WBC reaching conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import ROOT
from reaching_workspace import assess_trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wbc-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    study=json.loads((args.wbc_study/'study.json').read_text())
    names=['front_5cm','front_5cm_moving_40N','front_5cm_moving_70N',
           'front_5cm_hold_40N','front_5cm_hold_70N']
    results=[]
    for name in names:
        wbc=next(r for r in study['trials'] if r['trial_id']==name)
        out=args.output/name
        command=[sys.executable,str(ROOT/'experiments/reach_and_balance.py'),'--output',str(out),
                 '--target-offset-m','.05','0','0','--controller','pd_nominal_ff',
                 '--push-N',str(wbc['push_N']),'--push-start-s',str(wbc['push_start_s'])]
        with (args.output/f'{name}.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        s,passed,reason=assess_trial(out)
        with np.load(out/'trajectory.npz',allow_pickle=False) as a, np.load(
                args.wbc_study/'trials'/name/'trajectory.npz',allow_pickle=False) as b:
            for key in ('time_s','reach_reference_world','reach_goal_world','push_force'):
                np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            np.testing.assert_array_equal(a['qpos_history'][0],b['qpos_history'][0])
            np.testing.assert_array_equal(a['qvel_history'][0],b['qvel_history'][0])
        row=dict(trial_id=name, pd_success=passed, pd_failure_reason=reason,
                 pd_hold_max_error_mm=s['hold_max_goal_error_m']*1000,
                 wbc_success=wbc['combined_success'],wbc_hold_max_error_mm=wbc['hold_max_error_mm'],
                 ik_max_reference_error_m=s['ik_max_reference_error_m'],
                 paired_state_reference_force_verified=True,
                 pd_source_version=s['provenance']['source_version'],wbc_source_version=wbc['source_version'])
        results.append(row)
        print(json.dumps(row),flush=True)
        if name=='front_5cm' and not passed:
            break
    payload=dict(pairs=results, baseline_qualified=results[0]['pd_success'],
                 method='PD + fixed nominal equilibrium feedforward, offline bounded seven-joint arm IK at 30 Hz; identical Cartesian reference and force trace',
                 scope='Single-target deterministic qualification pilot; not a general controller ranking')
    (args.output/'comparison.json').write_text(json.dumps(payload,indent=2))
    from se3_whole_body_control.visualization.style import apply_style
    import matplotlib.pyplot as plt
    apply_style()
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    x=np.arange(len(results))
    for shift,key,color,label in [(-.16,'pd_hold_max_error_mm','#b46720','PD + nominal FF'),
                                  (.16,'wbc_hold_max_error_mm','#257a59','SE(3) WBC')]:
        ax.bar(x+shift,[r[key] for r in results],width=.3,color=color,label=label)
    ax.axhline(15,color='#ae3946',ls='--',label='15 mm tracking tolerance')
    ax.set(xticks=x,xticklabels=[r['trial_id'].replace('front_5cm','No push').replace('No push_','') for r in results],
           ylabel='Maximum final-hold target error [mm]',
           title='Single-target PD pilot' if payload['baseline_qualified'] else 'PD qualification failed without push')
    ax.legend(fontsize=9)
    for suffix in ('png','pdf'):
        fig.savefig(args.output/f'comparison.{suffix}',dpi=250)
    plt.close(fig)
    files=[dict(path=p.relative_to(args.output).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(args.output.rglob('*')) if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2))


if __name__=='__main__':
    main()
