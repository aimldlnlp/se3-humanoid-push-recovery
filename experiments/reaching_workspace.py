"""Small deterministic reach study; simulation and rendering run on the worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

if os.name != 'nt' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

from common import ROOT, load_configs, make_model, recovery_config, write_csv
from se3_whole_body_control.evaluation.recovery import classify_recovery

DIRECTIONS = {'front': (1, 0, 0), 'right': (0, -1, 0), 'up': (0, 0, 1)}


def assess_trial(path):
    summary = json.loads((path/'summary.json').read_text())
    with np.load(path/'trajectory.npz', allow_pickle=False) as a:
        cfg = recovery_config(load_configs(ROOT, robot_name='unitree_g1'))
        com = np.linalg.norm(a['com_world'][:, :2]-np.asarray(summary['provenance']['com_reference_world'])[:2], axis=1)
        # The recovery classifier starts after a push. This additional check
        # catches physical failures during motion and during the push itself.
        whole = classify_recovery(
            a['time_s'], a['torso_rotation_error_rad'], a['torso_angular_velocity_norm'], com,
            a['contact_left_post_step'], a['contact_right_post_step'], a['torso_height_m'],
            a['torque_abs_max_Nm'], a['qp_success'], a['actual_friction_margin'], config=cfg,
            actual_friction_utilization=a['actual_friction_utilization_post_step'],
            foot_tangent_velocity=a['foot_tangent_velocity_post_step'],
            foot_xy_displacement=a['foot_xy_displacement_post_step'],
            torque_utilization=a['torque_utilization'], joint_limit_violation=a['joint_limit_violation'],
            numerical_valid=a['numerical_valid'], require_final_double_support=True)
        hold = a['time_s'] >= 4.5
        final_balanced = bool(np.all(a['torso_rotation_error_rad'][hold] <= cfg.orientation_threshold_rad)
                              and np.all(a['torso_angular_velocity_norm'][hold] <= cfg.angular_velocity_threshold_rad_s)
                              and np.all(com[hold] <= cfg.com_displacement_threshold_m))
        passed = bool(summary['combined_success'] and whole.success and final_balanced)
        reason = whole.failure_reason or summary['recovery']['failure_reason']
        if not reason and not summary['reach_success']:
            reason = 'TRACKING'
        if not reason and not final_balanced:
            reason = 'FINAL_BALANCE'
    return summary, passed, reason or 'PASS'


def run_trial(root, name, offset, magnitude=0, push_start=1.5, stage='workspace', direction='', distance=0,
              reach_weight=None, objective_overrides=None, balance_guard=False):
    path = root/'trials'/name
    command = [sys.executable, str(ROOT/'experiments/reach_and_balance.py'), '--output', str(path),
               '--target-offset-m', *map(str, offset), '--push-N', str(magnitude), '--push-start-s', str(push_start)]
    if reach_weight is not None:
        command.extend(['--reach-weight', str(reach_weight)])
    for key, value in (objective_overrides or {}).items():
        command.extend(['--'+key.replace('_', '-'), str(value)])
    if balance_guard:
        command.append('--balance-guard')
    with (root/'logs'/f'{name}.txt').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    s, passed, reason = assess_trial(path)
    row = dict(trial_id=name, stage=stage, direction=direction, distance_m=distance,
               push_N=magnitude, push_start_s=push_start, combined_success=passed,
               reach_success=s['reach_success'], balance_success=s['balance_success'], failure_reason=reason,
               final_error_mm=1000*s['final_goal_error_m'], hold_max_error_mm=1000*s['hold_max_goal_error_m'],
               tracking_rmse_mm=1000*s['tracking_rmse_m'], torque_peak=s['recovery']['max_joint_torque_Nm'],
               max_foot_displacement_m=s['max_foot_displacement_m'],
               max_foot_tangent_velocity_m_s=s['max_foot_tangent_velocity_m_s'],
               continuous_double_support=s['continuous_double_support'], qp_failures=s['qp_failures'],
               deadline_miss_percent=s['qp_deadline_miss_percent'],
               initial_condition_sha256=s['provenance']['initial_condition']['sha256'],
               source_version=s['provenance']['source_version'])
    print(json.dumps(row), flush=True)
    return row


def plot_results(root, rows):
    from se3_whole_body_control.visualization.style import apply_style
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {'PASS': '#257a59', 'TRACKING': '#b46720'}
    for index, direction in enumerate(DIRECTIONS):
        selected = sorted([r for r in rows if r['stage']=='workspace' and r['direction']==direction],
                          key=lambda r: r['distance_m'])
        for r in selected:
            x = r['distance_m']*100
            color = colors.get(r['failure_reason'], '#ae3946')
            axes[0].scatter(x, index, color=color, marker='s' if r['combined_success'] else 'X', s=95)
            axes[0].annotate(f"{r['hold_max_error_mm']:.1f}", (x, index), xytext=(0, 12),
                             textcoords='offset points', ha='center', fontsize=9)
        axes[1].plot([r['distance_m']*100 for r in selected], [r['hold_max_error_mm'] for r in selected],
                     marker='o', label=direction.title())
    axes[0].set(yticks=range(3), yticklabels=['Front (+x)', 'Right (-y)', 'Up (+z)'],
                ylim=(-.5, 2.6), xlabel='Target offset [cm]', title='(a) Tested outcomes; labels: hold error [mm]')
    axes[0].legend(handles=[Line2D([],[],color='#257a59',marker='s',ls='',label='Combined pass'),
                           Line2D([],[],color='#b46720',marker='X',ls='',label='Tracking fail')],
                   loc='upper right',fontsize=9)
    axes[1].axhline(15, color='#ae3946', ls='--', label='15 mm tolerance')
    axes[1].set(xlabel='Target offset [cm]', ylabel='Maximum target error in final hold [mm]',
                title='(b) Tracking limit')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(sorted({r['distance_m']*100 for r in rows if r['stage']=='workspace'}))
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'workspace_results.{suffix}', dpi=250)
    plt.close(fig)


def render_saved(root, rows):
    from se3_whole_body_control.visualization.renderer import render_trial_frames
    from se3_whole_body_control.visualization.video import encode_video, _ffmpeg_executable
    workspace = [r for r in rows if r['stage']=='workspace']
    passed = [r for r in workspace if r['combined_success']]
    failed = [r for r in workspace if not r['combined_success']]
    if not passed or not failed:
        raise RuntimeError('Montage requires observed successes and failures')
    easy = min(passed, key=lambda r: r['hold_max_error_mm'])
    boundary = max(passed, key=lambda r: r['distance_m'])
    same_direction = [r for r in failed if r['direction']==boundary['direction']]
    failure = min(same_direction or failed, key=lambda r: r['distance_m'])
    selected = [('Easy', easy), ('Near tested limit', boundary), ('Failed task', failure)]
    clips = []
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    for label, row in selected:
        with np.load(root/'trials'/row['trial_id']/'trajectory.npz', allow_pickle=False) as a:
            indices = np.argmin(np.abs(a['time_s'][:, None]-np.arange(150)[None, :]/30), axis=0)
            overlay = [dict(time_s=float(a['time_s'][i]), controller='SE(3) WBC | reaching',
                            status=str(a['qp_status'][i]), compact_overlay=True,
                            task_label=f"{label} | {row['direction']} {row['distance_m']*100:g} cm | {row['failure_reason']}",
                            com_world=a['com_world'][i], feet_xy=a['foot_xy_world'][i],
                            contact_left=bool(a['contact_left'][i]), contact_right=bool(a['contact_right'][i]),
                            active_support_vertices_world=a['foot_support_vertices_world'][i].reshape(2,4,2)[
                                np.array([a['contact_left'][i],a['contact_right'][i]], dtype=bool)],
                            reach_goal_world=a['reach_goal_world'], reach_point_world=a['reach_point_world'][i],
                            push_force=a['push_force'][i], push_point_world=a['torso_position'][i],
                            push_magnitude_N=row['push_N'], push_direction_deg=0) for i in indices]
            clip = root/'videos'/f"{row['trial_id']}.mp4"
            with tempfile.TemporaryDirectory(prefix='workspace-render-') as frames:
                render_trial_frames(make_model(cfg), a['qpos_history'][indices], frames,
                                    width=1920, height=1080, overlay_data=overlay)
                encode_video(frames, clip, fps=30)
            clips.append(clip)
    listing = root/'videos'/'concat.txt'
    listing.write_text(''.join(f"file '{p.name}'\n" for p in clips))
    subprocess.run([_ffmpeg_executable(), '-n', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', str(root/'videos'/'reaching_montage.mp4')], check=True)
    listing.unlink()
    (root/'montage_selection.json').write_text(json.dumps(selected, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    root = args.output
    if args.render_only:
        rows = json.loads((root/'study.json').read_text())['trials']
        (root/'videos').mkdir(exist_ok=False)
        presentation = root/'presentation'
        presentation.mkdir(exist_ok=False)
        plot_results(presentation, rows)
        render_saved(root, rows)
        return
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    rows = []
    for name, magnitude in [('baseline_no_push',0), ('baseline_push',70)]:
        r = run_trial(root, name, [.08,-.03,.02], magnitude, stage='regression')
        rows.append(r)
        old = ROOT/'results/reaching/ff989e0'/('push' if magnitude else 'no_push')/'trajectory.npz'
        with np.load(old, allow_pickle=False) as a, np.load(root/'trials'/name/'trajectory.npz', allow_pickle=False) as b:
            for key in ('qpos_history','qvel_history','reach_point_world','control','actual_contact_wrench_post_step'):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-8, atol=1e-9, err_msg=key)
        if not r['combined_success']:
            raise RuntimeError('Baseline regression failed')
    for direction, vector in DIRECTIONS.items():
        last_pass = 0
        for cm in (5,10,15,20):
            r = run_trial(root, f'{direction}_{cm}cm', np.asarray(vector)*cm/100,
                          direction=direction, distance=cm/100)
            rows.append(r)
            if not r['combined_success']:
                if last_pass:
                    midpoint = (last_pass+cm)/2
                    rows.append(run_trial(root, f'{direction}_{midpoint:g}cm', np.asarray(vector)*midpoint/100,
                                          direction=direction, distance=midpoint/100))
                break
            last_pass = cm
        write_csv(rows, root/'study.csv')
    candidates = [r for r in rows if r['stage']=='workspace' and r['combined_success']]
    if not candidates:
        raise RuntimeError('No successful workspace pilot')
    choices = [min(candidates, key=lambda r:r['hold_max_error_mm']),
               max(candidates, key=lambda r:r['hold_max_error_mm'])]
    for candidate in {r['trial_id']:r for r in choices}.values():
        offset = np.asarray(DIRECTIONS[candidate['direction']])*candidate['distance_m']
        for phase, start in [('moving',1.5), ('hold',3.0)]:
            for magnitude in (40,70):
                rows.append(run_trial(root, f"{candidate['trial_id']}_{phase}_{magnitude}N", offset,
                                      magnitude, start, stage='disturbance', direction=candidate['direction'],
                                      distance=candidate['distance_m']))
                write_csv(rows, root/'study.csv')
    assert len({r['initial_condition_sha256'] for r in rows}) == 1
    payload = dict(trials=rows, scope='Deterministic sampled rays, not a complete workspace or robustness study',
                   target_frame='World offsets from settled endpoint; front +x, right -y, up +z',
                   evaluation='Existing recovery plus whole-task physical classifier and final 0.5 s balance band',
                   source_version=rows[0]['source_version'], baseline_regression_passed=True)
    (root/'study.json').write_text(json.dumps(payload, indent=2))
    plot_results(root, rows)
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2))


if __name__ == '__main__':
    main()
