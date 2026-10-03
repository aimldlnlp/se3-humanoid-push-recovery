"""Frozen balance-guard pilot, then paired validation only if the pilot passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import write_csv
from reaching_tracking_study import verify_pair
from reaching_workspace import run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    rows = []
    def paired(name, offset, phase, magnitude, start, stage):
        paths = []
        for policy in ('fixed_high', 'balance_guard'):
            trial_id = f'{name}_{phase}_{policy}'
            row = run_trial(root, trial_id, offset, magnitude, start, stage=stage,
                            reach_weight=3000, balance_guard=policy=='balance_guard')
            path = root/'trials'/trial_id
            summary = json.loads((path/'summary.json').read_text())
            row.update(policy=policy, target=name, phase=phase,
                       first_guard_reduction_s=summary['balance_guard']['first_reduction_s'],
                       reduced_duration_s=summary['balance_guard']['reduced_duration_s'])
            rows.append(row)
            paths.append(path)
            write_csv(rows, root/'study.csv')
        verify_pair(*paths)
        if phase == 'no_push':
            verify_pair(*paths, same_dynamics=True)
            assert rows[-1]['reduced_duration_s']==0, 'guard changed nominal motion'
        return paths
    for phase, magnitude, start in (('no_push',0,1.5), ('moving',70,1.5), ('hold',70,3.0)):
        paths = paired('front_12.5cm', [.125,0,0], phase, magnitude, start, 'pilot')
        if phase == 'no_push':
            verify_pair(args.previous/'trials'/'front_12.5cm_w3000', paths[0], same_dynamics=True)
    gate = all(r['combined_success'] for r in rows if r['policy']=='balance_guard')
    print(json.dumps(dict(pilot_gate_passed=gate)), flush=True)
    if gate:
        for name, offset in (('right_5cm',[0,-.05,0]), ('up_5cm',[0,0,.05]), ('mixed_8cm',[.08,-.03,.02])):
            for phase, magnitude, start in (('no_push',0,1.5), ('moving',70,1.5), ('hold',70,3.0)):
                paired(name, offset, phase, magnitude, start, 'validation')
    assert len({r['initial_condition_sha256'] for r in rows})==1
    policy = json.loads((root/'trials'/'front_12.5cm_no_push_balance_guard'/'summary.json').read_text())['provenance']['reaching']['balance_guard']
    payload = dict(trials=rows, pilot_gate_passed=gate, baseline_regression_passed=True,
                   paired_inputs_verified=True, balance_guard_policy=policy,
                   scope='Frozen measured-state guard on known failures; separate paired validation if pilot passes; no threshold retuning',
                   source_version=rows[0]['source_version'],
                   observation='Fixed five-second window and original whole-task physical/final-hold tracking criteria')
    (root/'study.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot(root, rows)
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')


def plot(root, rows):
    import matplotlib.pyplot as plt
    from se3_whole_body_control.visualization.style import apply_style
    apply_style()
    fig, axes = plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for col, phase in enumerate(('no_push','moving','hold')):
        for policy, color in (('fixed_high','#ad3946'), ('balance_guard','#257a59')):
            row = next(r for r in rows if r['stage']=='pilot' and r['phase']==phase and r['policy']==policy)
            with np.load(root/'trials'/row['trial_id']/'trajectory.npz',allow_pickle=False) as a:
                error = 1000*np.linalg.norm(a['reach_point_world']-a['reach_goal_world'],axis=1)
                label = policy.replace('_',' ')+' | '+row['failure_reason']
                axes[0,col].plot(a['time_s'],error,color=color,label=label)
                axes[1,col].plot(a['time_s'],a['reach_weight_history'],color=color)
        axes[0,col].axhline(15,color='#555555',ls='--',lw=1)
        axes[0,col].set(title=phase.replace('_',' ').title(),yscale='symlog',ylabel='Target error [mm]')
        axes[0,col].set_yscale('symlog',linthresh=15)
        axes[0,col].legend(fontsize=8)
        axes[1,col].set(xlabel='Time [s]',ylabel='Effective reach weight',ylim=(0,3200))
        if phase!='no_push':
            start = 1.5 if phase=='moving' else 3
            for ax in axes[:,col]:
                ax.axvspan(start,start+.15,color='#777777',alpha=.15)
    for suffix in ('png','pdf'):
        fig.savefig(root/f'guard_results.{suffix}',dpi=250)
    plt.close(fig)


if __name__=='__main__':
    main()
