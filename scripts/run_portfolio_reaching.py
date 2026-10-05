"""Reproduce the corrected baseline, its predecessor, and a small sensitivity check."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from common import (load_configs, make_model, prepare_paired_initial_condition,
                    randomized_initial_state, save_paired_initial_condition, write_csv)
from reaching_tracking_study import verify_pair
from reaching_workspace import assess_trial, run_trial


PHASES = (('nominal', 0, 1.5), ('moving', 70, 1.5), ('hold', 70, 3.0))
SEEDS = (17, 29)
RANDOMIZATION = dict(initial_torso_tilt_rad=0, initial_com_offset_m=0,
                     initial_angular_velocity_rad_s=0.05)


def render_comparison(root, output=None):
    from PIL import Image, ImageDraw
    from render_com_support_animation import _nearest_indices
    from render_paired_comparison import overlay
    from se3_whole_body_control.visualization.fonts import pil_font
    from se3_whole_body_control.visualization.renderer import render_trial_frames
    from se3_whole_body_control.visualization.video import _ffmpeg_executable

    selections = (('before_hold', 'Before: contact correction'),
                  ('corrected_hold', 'After: pose-origin correction'),
                  ('lateral_hold', 'Known limit: lateral push'))
    output = output or root/'media'
    output.mkdir(exist_ok=False)
    records, panels = [], []
    with tempfile.TemporaryDirectory(prefix='reach-before-after-') as directory:
        temp = Path(directory)
        for trial, label in selections:
            path = root/'trials'/trial
            summary, passed, reason = assess_trial(path)
            with np.load(path/'trajectory.npz', allow_pickle=False) as payload:
                arrays = {k: payload[k] for k in payload.files}
            indices = _nearest_indices(arrays['time_s'], np.arange(150)/30)
            sampled = {k: v[indices] if v.ndim and len(v)==len(arrays['time_s']) else v
                       for k, v in arrays.items()}
            metadata = overlay(sampled, 'SE(3) WBC')
            for i, item in enumerate(metadata):
                item.update(reach_goal_world=arrays['reach_goal_world'],
                            reach_point_world=sampled['reach_point_world'][i])
            model = make_model(load_configs(ROOT, robot_name='unitree_g1'))
            model.model.cam_fovy[model.model.camera('track').id] = 60
            frames = render_trial_frames(model, sampled['qpos_history'], temp/trial,
                                         width=640, height=800, overlay_data=metadata)
            panels.append((frames, sampled, summary, label, reason))
            records.append(dict(trial=trial, result=reason, combined_success=passed,
                                trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest(),
                                frame_indices=indices.tolist()))
        combined = temp/'combined'
        combined.mkdir()
        for frame_index in range(150):
            canvas = Image.new('RGB', (1920, 1080), 'white')
            draw = ImageDraw.Draw(canvas)
            for panel, (frames, a, summary, label, reason) in enumerate(panels):
                x = panel*640
                with Image.open(frames[frame_index]) as robot:
                    canvas.paste(robot, (x, 90))
                draw.text((x+20, 15), label, font=pil_font(25, weight='bold'), fill='#20262a')
                draw.text((x+20, 51), f'Whole-trial result: {reason}', font=pil_font(23),
                          fill='#257a59' if reason=='PASS' else '#ad3946')
                i = frame_index
                error = 1000*np.linalg.norm(a['reach_point_world'][i]-a['reach_goal_world'])
                com = 1000*np.linalg.norm(a['com_world'][i, :2]-np.array(summary['provenance']['com_reference_world'])[:2])
                drift = 1000*np.max(a['foot_xy_displacement_post_step'][i])
                supported = bool(a['contact_left_post_step'][i] and a['contact_right_post_step'][i])
                drift_text = f'{drift:.1f} mm | Limit: 25 mm' if supported else 'n/a (contact lost)'
                lines = (f'Target distance: {error:.1f} mm | Hold limit: 15 mm',
                         f'CoM displacement: {com:.1f} mm | Band: 100 mm',
                         f'Foot drift: {drift_text}',
                         f'Torque utilization: {a["torque_utilization"][i]:.2f}')
                for line_index, line in enumerate(lines):
                    draw.text((x+20, 904+line_index*30), line, font=pil_font(21), fill='#20262a')
                if panel:
                    draw.line((x, 0, x, 1080), fill='#c1ccd1', width=1)
            draw.text((20, 1040), '12.5 cm reach | 70 N for 0.15 s at t=3 s | Fixed feet | Original simulation speed',
                      font=pil_font(23), fill='#505b61')
            canvas.save(combined/f'frame_{frame_index:06d}.png')
            if frame_index in (0, 90, 110, 149):
                canvas.save(output/f'frame_{frame_index:03d}.jpg', quality=94)
            if frame_index==110:
                canvas.save(output/'reach-before-after-preview.jpg', quality=94)
        subprocess.run([_ffmpeg_executable(), '-n', '-loglevel', 'error', '-framerate', '30',
                        '-i', str(combined/'frame_%06d.png'), '-c:v', 'libx264', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(output/'reach-before-after.mp4')], check=True)
    (output/'selection.json').write_text(json.dumps(dict(trials=records, fps=30, frames=150,
        duration_s=5, resolution=[1920, 1080], dynamics_rerun=False, camera_fovy_deg=60,
        renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='First two panels have paired inputs. Third changes only push direction to 90 degrees. '
              'Labels are whole-trial outcomes. Contact flags are pre-step; drift is post-step.'), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--media-output', type=Path)
    args = parser.parse_args()
    root = args.output
    if args.render_only:
        render_comparison(root, args.media_output)
        return
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    (root/'initial_states').mkdir()
    # Write the complete case list before any simulation; never tune after outcomes.
    cases = [dict(name='corrected_'+phase, force=force, start=start, seed=None, pose_origin=True, direction=0)
             for phase, force, start in PHASES]
    cases += [dict(name='before_hold', force=70, start=3, seed=None, pose_origin=False, direction=0),
              dict(name='lateral_hold', force=70, start=3, seed=None, pose_origin=True, direction=90)]
    cases += [dict(name=f'seed_{seed}_{phase}', force=force, start=start, seed=seed, pose_origin=True, direction=0)
              for seed in SEEDS for phase, force, start in PHASES]
    source_paths = [ROOT/'scripts/run_portfolio_reaching.py', ROOT/'experiments/reach_and_balance.py',
                    ROOT/'experiments/reaching_workspace.py', ROOT/'experiments/reaching_contact_mode.py',
                    ROOT/'experiments/reaching_pose_origin.py']
    plan = dict(cases=cases, randomization=RANDOMIZATION, target_offset_m=[.125, 0, 0], reach_weight=3000,
                contact_velocity_damping_s_inv=20, hierarchical=False, physical_thresholds_changed=False,
                scope='Six predeclared angular-rate sensitivity trials; not a general robustness estimate.',
                source_files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                              for p in source_paths])
    (root/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    save_paired_initial_condition(initial, root/'initial_states/nominal.npz')
    perturbations = {}
    for seed in SEEDS:
        _, delta, details = randomized_initial_state(cfg, seed, randomization=RANDOMIZATION)
        condition = replace(initial, qvel=initial.qvel+delta, seed=seed,
                            setup_policy='pd_nominal_ff_then_angular_rate_offset')
        save_paired_initial_condition(condition, root/f'initial_states/seed_{seed}.npz')
        perturbations[str(seed)] = dict(**details, qvel_offset=delta.tolist())
    (root/'perturbations.json').write_text(json.dumps(perturbations, indent=2), encoding='utf-8')
    rows = []
    for case in cases:
        state = 'nominal' if case['seed'] is None else f'seed_{case["seed"]}'
        row = run_trial(root, case['name'], [.125, 0, 0], case['force'], case['start'],
                        stage='portfolio_baseline' if case['seed'] is None else 'initial_rate_sensitivity',
                        reach_weight=3000, contact_velocity_damping=20, contact_mode=True,
                        pose_origin=case['pose_origin'], push_direction_deg=case['direction'],
                        initial_condition_path=root/f'initial_states/{state}.npz')
        row.update(seed=case['seed'], pose_origin=case['pose_origin'], push_direction_deg=case['direction'])
        rows.append(row)
        write_csv(rows, root/'study.csv')
    verify_pair(root/'trials/before_hold', root/'trials/corrected_hold')
    replay = {}
    for phase, _, _ in PHASES:
        reference = ROOT/f'results/reaching_contact/pose_origin_pilot/trials/front_12.5cm_contact_pose_origin_{"no_push" if phase=="nominal" else phase}'
        verify_pair(reference, root/f'trials/corrected_{phase}', same_dynamics=True)
        replay[phase] = True
    baseline = [r for r in rows if r['trial_id'].startswith('corrected_')]
    sensitivity = [r for r in rows if r['seed'] is not None]
    report = dict(trials=rows, baseline_reproduced=replay, before_after_inputs_paired=True,
                  baseline_passed=sum(r['combined_success'] for r in baseline), baseline_count=len(baseline),
                  sensitivity_passed=sum(r['combined_success'] for r in sensitivity), sensitivity_count=len(sensitivity),
                  hierarchical=False, production_promoted=False, hardware_trials_run=0,
                  scope=plan['scope'])
    (root/'study.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.render:
        render_comparison(root, args.media_output)
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k!='trials'}, indent=2))


if __name__=='__main__':
    main()
