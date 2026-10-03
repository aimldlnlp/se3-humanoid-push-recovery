"""One right-arm Cartesian reach using the existing fixed-foot WBC.

Run on the SSH worker. Previous benchmark artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

if os.name != 'nt' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

from common import (ROOT, execution_manifest, load_configs, make_model, make_push,
                    prepare_paired_initial_condition, recovery_config)
from se3_whole_body_control.config import load_yaml
from se3_whole_body_control.control.tasks import ReachTask, quintic_reference
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
from se3_whole_body_control.evaluation.metrics import save_trial_npz, summarize_trial
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner


class ReachingController(WholeBodyQPController):
    def __init__(self, model, controller_config, settings):
        super().__init__(model, controller_config)
        self.settings = settings
        self.start = model.attached_point_kinematics(settings['body_name'], settings['point_local_m'])[0]
        self.goal = self.start + np.asarray(settings['target_offset_world_m'])
        self.reach_task = ReachTask(settings['body_name'], np.asarray(settings['point_local_m']),
                                    self.start, np.zeros(3), np.zeros(3),
                                    settings['kp'], settings['kd'], settings['weight'])
        self.points, self.references = [], []

    def solve(self):
        task = self.reach_task
        task.position_world, task.velocity_world, task.acceleration_world = quintic_reference(
            self.start, self.goal, self.model.data.time-self.settings['start_time_s'],
            self.settings['movement_duration_s'])
        self.points.append(self.model.attached_point_kinematics(task.body_name, task.point_local)[0].copy())
        self.references.append(task.position_world.copy())
        return super().solve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--push-N', type=float, default=0)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.push_N) or args.push_N < 0:
        parser.error('--push-N must be finite and nonnegative')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('output must be empty: preserve existing run artifacts')
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    model = make_model(cfg)
    model.reset(initial.qpos, initial.qvel)
    controller = ReachingController(model, cfg['controller'], settings)
    runner = SimulationRunner(model, controller, duration_s=settings['experiment_duration_s'],
                              control_timestep_s=cfg['robot']['control_timestep'], warmup_duration_s=0)
    push = make_push(cfg, magnitude=args.push_N, start=1.5) if args.push_N else None
    run = runner.run(initial_qpos=initial.qpos, initial_qvel=initial.qvel,
                     desired_torso=initial.desired_torso, desired_pelvis=initial.desired_pelvis,
                     com_reference=initial.com_reference, joint_reference=initial.joint_reference,
                     initial_condition_metadata=initial.metadata(), push=push,
                     classify=True, recovery_config=recovery_config(cfg))
    times = run.log.arrays()['time_s']
    points, references = np.asarray(controller.points), np.asarray(controller.references)
    error = np.linalg.norm(points-references, axis=1)
    goal_error = np.linalg.norm(points-controller.goal, axis=1)
    hold = times >= settings['experiment_duration_s']-settings['hold_duration_s']
    reach_ok = bool(np.any(hold) and np.all(goal_error[hold] <= settings['position_tolerance_m']))
    balance_ok = bool(run.recovery.success)
    metadata = execution_manifest({'config': {'robot': cfg['robot'], 'controller': cfg['controller'],
                                  'reaching': settings, 'push_N': args.push_N}})
    metadata.update(initial_condition=initial.metadata(), reaching=settings,
                    goal_world_m=controller.goal.tolist(), actual_impulse_Ns=run.metadata['realized_impulse_Ns'])
    summary = {**summarize_trial(run.log), 'reach_success': reach_ok, 'balance_success': balance_ok,
               'combined_success': reach_ok and balance_ok, 'recovery': asdict(run.recovery),
               'tracking_rmse_m': float(np.sqrt(np.mean(error**2))),
               'final_goal_error_m': float(goal_error[-1]), 'provenance': metadata}
    save_trial_npz(run.log, output/'trajectory.npz', metadata, {
        'qpos_history': np.asarray(run.qpos_history), 'qvel_history': np.asarray(run.qvel_history),
        'reach_point_world': points, 'reach_reference_world': references,
        'reach_goal_world': controller.goal, 'reach_error_m': error})
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    from se3_whole_body_control.visualization.style import apply_style, COLORS
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    axes[0, 0].plot(times, error*1000, color=COLORS['wbc'])
    axes[0, 0].set(ylabel='Tracking error [mm]', title='(a) Cartesian tracking')
    axes[0, 1].plot(times, goal_error*1000, color=COLORS['actual'])
    axes[0, 1].axhline(settings['position_tolerance_m']*1000, ls='--', color=COLORS['boundary'])
    axes[0, 1].set(ylabel='Target distance [mm]', title='(b) Final target')
    arrays = run.log.arrays()
    for foot, color in enumerate((COLORS['left_foot'], COLORS['right_foot'])):
        axes[1, 0].plot(times, arrays['actual_contact_wrench_post_step'][:, 6*foot+2],
                        color=color, label=('Left', 'Right')[foot])
    axes[1, 0].set(ylabel='Actual vertical GRF [N]', title='(c) Physical contacts')
    axes[1, 0].legend()
    axes[1, 1].plot(times, arrays['torque_utilization'], color=COLORS['wbc'])
    axes[1, 1].axhline(1, ls='--', color=COLORS['boundary'])
    axes[1, 1].set(ylabel='Torque utilization', title='(d) Actuation')
    for ax in axes.flat:
        ax.set_xlabel('Time [s]')
        if push:
            ax.axvspan(push.start_time_s, push.start_time_s+push.duration_s, color=COLORS['push'], alpha=.12)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'reaching_response.{suffix}', dpi=300)
    plt.close(fig)
    if args.render:
        from se3_whole_body_control.visualization.renderer import render_trial_frames
        from se3_whole_body_control.visualization.video import encode_video, make_gif
        fps = 30
        indices = np.argmin(np.abs(times[:, None]-np.arange(round(settings['experiment_duration_s']*fps))[None, :]/fps), axis=0)
        overlay = [{'time_s': times[i], 'controller': 'SE(3) WBC | reaching', 'status': arrays['qp_status'][i],
                    'task_label': 'SE(3) WBC | right-arm reach',
                    'active_support_vertices_world': arrays['foot_support_vertices_world'][i].reshape(2, 4, 2)[
                        np.array([arrays['contact_left'][i], arrays['contact_right'][i]], dtype=bool)],
                    'compact_overlay': True, 'com_world': arrays['com_world'][i],
                    'feet_xy': arrays['foot_xy_world'][i], 'contact_left': arrays['contact_left'][i],
                    'contact_right': arrays['contact_right'][i], 'reach_goal_world': controller.goal,
                    'reach_point_world': points[i], 'push_force': arrays['push_force'][i],
                    'push_point_world': arrays['torso_position'][i], 'push_magnitude_N': args.push_N,
                    'push_direction_deg': 0} for i in indices]
        with tempfile.TemporaryDirectory(prefix='reach-render-') as frames:
            render_trial_frames(make_model(cfg), np.asarray(run.qpos_history)[indices], frames,
                                width=1920, height=1080, overlay_data=overlay)
            encode_video(frames, output/'reach_and_balance.mp4', fps=fps)
            make_gif(frames, output/'reach_and_balance.gif', fps=fps)
    print(json.dumps({key: summary[key] for key in ('reach_success', 'balance_success',
          'combined_success', 'final_goal_error_m', 'qp_failures')}, indent=2))


if __name__ == '__main__':
    main()
