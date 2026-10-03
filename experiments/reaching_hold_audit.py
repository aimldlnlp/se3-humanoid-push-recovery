"""Inspect saved hold failures: foot pose, contact geometry and paired QP replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, recovery_config, write_csv
from reaching_contact_audit import replay, timeline
from reaching_tracking_study import verify_pair


def foot_geometry(model, reference):
    """World-frame body-origin drift and sole geometry; not material-point slip."""
    rows = []
    for foot, name in enumerate(('left_foot', 'right_foot')):
        pose = model.body_pose(name)
        corners = model.support_vertices_local[foot] @ pose[:3, :3].T + pose[:3, 3]
        normal = pose[:3, 2]
        tilt = np.arctan2(np.linalg.norm(normal[:2]), normal[2])
        contacts = [model.data.contact[i] for i in range(model.data.ncon)
                    if model.geom_ids['ground'] in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                    and any(g in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                            for g in model.foot_contact_geom_ids[foot])]
        rows.append(dict(foot=name, body_xy_drift_m=float(np.linalg.norm(pose[:2, 3]-reference[foot][:2, 3])),
                         sole_tilt_rad=float(tilt), sole_corner_min_z_m=float(corners[:, 2].min()),
                         sole_corner_max_z_m=float(corners[:, 2].max()), contact_count=len(contacts),
                         contact_distance_min_m=min((float(c.dist) for c in contacts), default=None),
                         body_linear_speed_m_s=float(np.linalg.norm(model.body_velocity(name)[:3]))))
    return rows


def inspect(path, cfg, initial):
    with np.load(path/'trajectory.npz', allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    summary = json.loads((path/'summary.json').read_text())
    criteria = recovery_config(cfg)
    start = summary['provenance']['push_start_s']
    events = timeline(arrays, summary, criteria, start)
    stop = events['slip_velocity'] or events['contact_loss'] or 4.9
    model = make_model(cfg)
    model.reset(arrays['qpos_history'][0], arrays['qvel_history'][0])
    reference = [model.body_pose(name).copy() for name in ('left_foot', 'right_foot')]
    geometry = []
    for i, time in enumerate(arrays['time_s']):
        if not start-.1 <= time <= stop+.08:
            continue
        model.reset(arrays['qpos_history'][i], arrays['qvel_history'][i])
        for foot, row in enumerate(foot_geometry(model, reference)):
            # Forces and slip remain the original logged measurements; mj_forward
            # reconstructs geometry, not the original solver's contact-force history.
            row.update(time_s=float(time), normal_force_N=float(arrays['actual_contact_wrench'][i, 6*foot+2]),
                       contact_tangent_speed_post_step_m_s=float(arrays['foot_tangent_velocity_post_step'][i, foot]),
                       logged_contact_post_step=bool(arrays[('contact_left_post_step', 'contact_right_post_step')[foot]][i]))
            geometry.append(row)
    sample_times = sorted(set([start-.004, start+.152, start+.44, (start+stop)/2, stop-.052]))
    frozen = replay(path, arrays, summary, sample_times, initial, cfg)
    for snapshot in frozen:
        i = int(np.argmin(abs(arrays['time_s']-snapshot['time_s'])))
        model.reset(arrays['qpos_history'][i], arrays['qvel_history'][i])
        snapshot['geometry'] = foot_geometry(model, reference)
    record = dict(path=path.relative_to(ROOT).as_posix(), events=events, frozen_replay=frozen,
                  combined_success=summary['combined_success'], source_version=summary['provenance']['source_version'],
                  trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest())
    return record, geometry


def plot(root, all_rows):
    import matplotlib.pyplot as plt
    from se3_whole_body_control.visualization.style import apply_style
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for col, policy in enumerate(('original', 'damping_20')):
        for foot, color in (('left_foot', '#167c9c'), ('right_foot', '#ad3946')):
            rows = [r for r in all_rows if r['policy']==policy and r['foot']==foot]
            times = [r['time_s'] for r in rows]
            for ax, key, scale in zip(axes[:, col], ('body_xy_drift_m', 'sole_tilt_rad', 'normal_force_N'),
                                      (1000, 180/np.pi, 1)):
                ax.plot(times, [scale*r[key] for r in rows], color=color, label=foot)
        axes[0, col].set_title(policy+' | Hold push 70 N')
        for ax, label in zip(axes[:, col], ('Body-origin XY drift [mm]', 'Sole tilt [deg]', 'Logged pre-step normal [N]')):
            ax.set_ylabel(label)
            ax.axvspan(3, 3.15, color='#777777', alpha=.15)
            ax.legend(fontsize=8)
        axes[-1, col].set_xlabel('Time [s]')
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'hold_geometry.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    paths = dict(original=ROOT/'results/reaching_tracking/fbc0b50_local/trials/front_12.5cm_w3000_hold_70N',
                 damping_20=ROOT/'results/reaching_contact/damping_pilot/trials/front_12.5cm_contact_damping_hold')
    verify_pair(paths['original'], paths['damping_20'])
    records, rows = [], []
    for policy, path in paths.items():
        record, geometry = inspect(path, cfg, initial)
        record['policy'] = policy
        records.append(record)
        rows.extend(dict(policy=policy, **row) for row in geometry)
        print(policy, json.dumps(record['frozen_replay'][-1]), flush=True)
    write_csv(rows, args.output/'geometry.csv')
    report = dict(records=records, physical_trials_run=0, paired_inputs_verified=True,
                  analysis_source_version=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  geometry_definition='Saved pre-step qpos/qvel; reconstructed ground-contact manifold and body-local sole corners',
                  limitation='Body drift and tilt do not uniquely identify sliding; reconstructed manifold is not force-bearing contact history. Post-step slip is labeled separately. No controller or thresholds changed.')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot(args.output, rows)
    files = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
