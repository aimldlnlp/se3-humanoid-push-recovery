"""One-factor reach-weight calibration, followed by frozen validation trials.

The original reaching configuration and physical success thresholds stay fixed.
Stop if no tested weight passes both calibration targets; preserve every failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT, load_configs, make_model, write_csv
from reaching_workspace import run_trial

TARGETS = {'front_12.5cm': [.125, 0, 0], 'up_5cm': [0, 0, .05]}


def verify_pair(old, new, same_dynamics=False):
    with np.load(old/'trajectory.npz', allow_pickle=False) as a, np.load(new/'trajectory.npz', allow_pickle=False) as b:
        for key in ('time_s', 'reach_reference_world', 'reach_goal_world', 'push_force'):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        for key in ('qpos_history', 'qvel_history'):
            np.testing.assert_array_equal(a[key][0], b[key][0], err_msg=key)
        if same_dynamics:
            for key in ('qpos_history', 'qvel_history', 'reach_point_world', 'control', 'actual_contact_wrench_post_step'):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-8, atol=1e-9, err_msg=key)


def diagnose(path):
    model = make_model(load_configs(ROOT, robot_name='unitree_g1'))
    with np.load(path/'trajectory.npz', allow_pickle=False) as a:
        model.reset(a['qpos_history'][-1], a['qvel_history'][-1])
        _, J, _ = model.attached_point_kinematics('right_wrist_yaw_link', [.08, 0, 0])
        arm = [i for i, name in enumerate(model.joint_names)
               if name.startswith('right_') and any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
        return dict(final_error_xyz_m=(a['reach_point_world'][-1]-a['reach_goal_world']).tolist(),
                    maximum_torque_utilization=float(np.max(a['torque_utilization'])),
                    final_arm_jacobian_singular_values=np.linalg.svd(J[:, model.joint_qvel_indices[arm]])[1].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    root, previous = args.output, args.previous
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    rows = []
    def trial(name, offset, weight, stage, magnitude=0, start=1.5):
        row = run_trial(root, name, offset, magnitude, start, stage=stage, reach_weight=weight)
        row['reach_weight'] = weight
        rows.append(row)
        write_csv(rows, root/'study.csv')
        return row
    # Repeat default trajectories before changing the single calibration factor.
    for name, offset in TARGETS.items():
        trial(name+'_w100', offset, 100, 'regression')
        verify_pair(previous/'trials'/name, root/'trials'/(name+'_w100'), same_dynamics=True)
    diagnostics = {name: diagnose(previous/'trials'/name) for name in TARGETS}
    selected = None
    for weight in (300, 1000, 3000):
        candidates = [trial(f'{name}_w{weight}', offset, weight, 'calibration') for name, offset in TARGETS.items()]
        for name in TARGETS:
            verify_pair(root/'trials'/(name+'_w100'), root/'trials'/f'{name}_w{weight}')
        if all(row['combined_success'] for row in candidates):
            selected = weight
            break
    # Freeze the first weight that passes both targets. Never retune on validation.
    if selected is not None:
        for name, offset in TARGETS.items():
            for phase, start in (('moving', 1.5), ('hold', 3.0)):
                trial(f'{name}_w{selected}_{phase}_70N', offset, selected, 'validation', 70, start)
        for name, offset in (('mixed_8cm', [.08, -.03, .02]), ('right_5cm', [0, -.05, 0]),
                             ('front_15cm', [.15, 0, 0]), ('up_10cm', [0, 0, .10])):
            trial(f'{name}_w{selected}', offset, selected, 'validation')
    assert len({row['initial_condition_sha256'] for row in rows}) == 1
    payload = dict(trials=rows, selected_reach_weight=selected, baseline_dynamics_reproduced=True,
                   paired_calibration_inputs_verified=True, diagnostics=diagnostics,
                   scope='Single-factor calibration on two known failures; separate frozen validation, not a general workspace claim',
                   selection_rule='First of 300, 1000, 3000 with both combined calibration successes',
                   unchanged='Plant, initial state, reference timing, all other gains/weights, constraints and physical evaluation',
                   source_version=rows[0]['source_version'])
    (root/'study.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot(root, rows)
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')


def plot(root, rows):
    import matplotlib.pyplot as plt
    from se3_whole_body_control.visualization.style import apply_style
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in TARGETS:
        selected = [r for r in rows if r['stage'] in ('regression', 'calibration') and r['trial_id'].startswith(name)]
        axes[0].plot([r['reach_weight'] for r in selected], [r['hold_max_error_mm'] for r in selected], marker='o', label=name)
    axes[0].set(xscale='log', xlabel='Reach objective weight', title='(a) Calibration on known tracking failures')
    axes[0].legend(fontsize=8)
    validation = [r for r in rows if r['stage']=='validation']
    axes[1].bar(range(len(validation)), [r['hold_max_error_mm'] for r in validation],
                color=['#257a59' if r['combined_success'] else '#ae3946' for r in validation])
    axes[1].set(xticks=range(len(validation)), xticklabels=[r['trial_id'].split('_w')[0]+'\n'+r['trial_id'].split('_w')[1] for r in validation],
                title='(b) Frozen validation; green = combined pass')
    axes[1].tick_params(axis='x', labelsize=7, rotation=45)
    for ax in axes:
        ax.axhline(15, color='#ae3946', ls='--')
        ax.set_ylabel('Maximum target error in final hold [mm]')
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'tracking_results.{suffix}', dpi=250)
    plt.close(fig)


if __name__ == '__main__':
    main()
