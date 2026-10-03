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
from se3_whole_body_control.geometry.se3 import inverse_se3, log_se3


class ReachBalanceGuard:
    """Immediate reduction; one stable dwell, then one stable-duration ramp."""
    def __init__(self, high_weight, low_weight, stable_duration):
        if not np.all(np.isfinite([high_weight, low_weight, stable_duration])) or not 0 < low_weight <= high_weight or stable_duration <= 0:
            raise ValueError('guard requires positive ordered weights and stable duration')
        self.high, self.low, self.duration = high_weight, low_weight, stable_duration
        self.recovering, self.stable_since = False, None

    def update(self, time_s, stable):
        if not stable:
            self.recovering, self.stable_since = True, None
            return self.low
        if not self.recovering:
            return self.high
        if self.stable_since is None:
            self.stable_since = time_s
        fraction = float(np.clip((time_s-self.stable_since)/self.duration-1, 0, 1))
        if fraction >= 1:
            self.recovering = False
        return self.low+fraction*(self.high-self.low)


class ReachingController(WholeBodyQPController):
    def __init__(self, model, controller_config, settings):
        super().__init__(model, controller_config)
        self.settings = settings
        damping = settings.get('contact_velocity_damping_s_inv', 0)
        if not np.isfinite(damping) or damping < 0:
            raise ValueError('contact velocity damping must be finite and nonnegative')
        self.start = model.attached_point_kinematics(settings['body_name'], settings['point_local_m'])[0]
        self.goal = self.start + np.asarray(settings['target_offset_world_m'])
        self.reach_task = ReachTask(settings['body_name'], np.asarray(settings['point_local_m']),
                                    self.start, np.zeros(3), np.zeros(3),
                                    settings['kp'], settings['kd'], settings['weight'])
        self.points, self.references = [], []
        guard = settings.get('balance_guard')
        self.guard = ReachBalanceGuard(settings['weight'], guard['minimum_weight'], guard['stable_duration_s']) if guard else None
        self.weights, self.guard_risks = [], []
        self.joint_references = []
        self.guard_arm = np.array([i for i, name in enumerate(model.joint_names)
                                   if name.startswith('right_') and any(part in name for part in ('shoulder','elbow','wrist'))])
        if guard and guard.get('track_arm_posture_during_recovery') and len(self.guard_arm)!=7:
            raise ValueError('expected seven right-arm joints')

    def _build_problem(self):
        problem = list(super()._build_problem())
        damping = self.settings.get('contact_velocity_damping_s_inv', 0)
        if damping:
            # Only the experiment changes the contact acceleration target.
            correction = damping*(problem[10]@self.internal_model.data.qvel)
            rows = slice(self.internal_model.nv, self.internal_model.nv+self.nw)
            problem[3][rows] -= correction
            problem[4][rows] -= correction
            problem[11] = problem[11]+correction
        return tuple(problem)

    def solve(self):
        task = self.reach_task
        task.position_world, task.velocity_world, task.acceleration_world = quintic_reference(
            self.start, self.goal, self.model.data.time-self.settings['start_time_s'],
            self.settings['movement_duration_s'])
        self.points.append(self.model.attached_point_kinematics(task.body_name, task.point_local)[0].copy())
        self.references.append(task.position_world.copy())
        risk = False
        if self.guard is not None:
            cfg = self.settings['balance_guard']
            rotation_error = np.linalg.norm(log_se3(self.model.body_pose('torso')@inverse_se3(self.T_des_torso))[3:])
            angular_speed = np.linalg.norm(self.model.body_velocity('torso')[3:])
            com_error = np.linalg.norm(self.model.center_of_mass()[:2]-self.com_des[:2])
            stable = bool(rotation_error <= cfg['orientation_threshold_rad']
                          and angular_speed <= cfg['angular_velocity_threshold_rad_s']
                          and com_error <= cfg['com_displacement_threshold_m']
                          and all(self.model.contact_flags()))
            risk = not stable
            task.weight = self.guard.update(self.model.data.time, stable)
            if cfg.get('track_arm_posture_during_recovery') and task.weight < self.settings['weight']:
                # Remove only the arm's pull toward its old standing pose;
                # retain damping and every other joint's standing reference.
                self.q_des[self.guard_arm] = self.model.joint_positions()[self.guard_arm]
                self.pd_fallback.q_des[self.guard_arm] = self.q_des[self.guard_arm]
        self.weights.append(task.weight)
        self.guard_risks.append(risk)
        self.joint_references.append(self.q_des.copy())
        return super().solve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--push-N', type=float, default=0)
    parser.add_argument('--target-offset-m', type=float, nargs=3)
    parser.add_argument('--push-start-s', type=float, default=1.5)
    parser.add_argument('--reach-weight', type=float)
    parser.add_argument('--qp-posture-weight', type=float)
    parser.add_argument('--qp-nominal-torque-weight', type=float)
    parser.add_argument('--balance-guard', action='store_true')
    parser.add_argument('--guard-arm-posture', action='store_true')
    parser.add_argument('--contact-velocity-damping', type=float, default=0)
    parser.add_argument('--controller', choices=('se3_wbc', 'pd_nominal_ff'), default='se3_wbc')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.contact_velocity_damping) or args.contact_velocity_damping < 0 or (args.contact_velocity_damping and args.controller != 'se3_wbc'):
        parser.error('--contact-velocity-damping requires a finite nonnegative WBC value')
    if args.balance_guard and args.controller != 'se3_wbc':
        parser.error('--balance-guard applies only to se3_wbc')
    if args.guard_arm_posture and not args.balance_guard:
        parser.error('--guard-arm-posture requires --balance-guard')
    if not np.isfinite(args.push_N) or args.push_N < 0:
        parser.error('--push-N must be finite and nonnegative')
    overrides = {}
    for key in ('qp_posture_weight', 'qp_nominal_torque_weight'):
        value = getattr(args, key)
        if value is not None:
            if not np.isfinite(value) or value < 0 or args.controller != 'se3_wbc':
                parser.error(f'--{key.replace("_", "-")} requires a finite nonnegative WBC weight')
            overrides[key] = value
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('output must be empty: preserve existing run artifacts')
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    if args.contact_velocity_damping:
        settings['contact_velocity_damping_s_inv'] = args.contact_velocity_damping
    default_reach_weight = settings['weight']
    if args.reach_weight is not None:
        if not np.isfinite(args.reach_weight) or args.reach_weight <= 0:
            parser.error('--reach-weight must be finite and positive')
        if args.controller != 'se3_wbc':
            parser.error('--reach-weight applies only to se3_wbc')
        settings['weight'] = args.reach_weight
    if args.target_offset_m is not None:
        if not np.all(np.isfinite(args.target_offset_m)):
            parser.error('target offset must be finite')
        settings['target_offset_world_m'] = args.target_offset_m
    if not np.isfinite(args.push_start_s) or not 0 <= args.push_start_s <= settings['experiment_duration_s'] - 0.15:
        parser.error('push must fit inside the observation window')
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    if args.balance_guard:
        recovery = recovery_config(cfg)
        settings['balance_guard'] = {key: getattr(recovery, key) for key in (
            'orientation_threshold_rad', 'angular_velocity_threshold_rad_s',
            'com_displacement_threshold_m', 'stable_duration_s')}
        settings['balance_guard']['minimum_weight'] = min(default_reach_weight, settings['weight'])
        settings['balance_guard']['track_arm_posture_during_recovery'] = bool(args.guard_arm_posture)
    initial = prepare_paired_initial_condition(cfg)
    cfg['controller'].update(overrides)
    model = make_model(cfg)
    model.reset(initial.qpos, initial.qvel)
    if args.controller == 'pd_nominal_ff':
        from reaching_pd import ReachingPDController
        controller = ReachingPDController(model, cfg['controller'], settings)
    else:
        controller = ReachingController(model, cfg['controller'], settings)
    runner = SimulationRunner(model, controller, duration_s=settings['experiment_duration_s'],
                              control_timestep_s=cfg['robot']['control_timestep'], warmup_duration_s=0)
    push = make_push(cfg, magnitude=args.push_N, start=args.push_start_s) if args.push_N else None
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
                                  'reaching': settings, 'push_N': args.push_N,
                                  'push_start_s': args.push_start_s, 'controller_name': args.controller}})
    metadata.update(initial_condition=initial.metadata(), reaching=settings,
                    push_N=args.push_N, push_start_s=args.push_start_s,
                    com_reference_world=initial.com_reference.tolist(),
                    controller=args.controller,
                    objective_weight_overrides=overrides,
                    ik_max_reference_error_m=getattr(controller, 'ik_max_error_m', None),
                    goal_world_m=controller.goal.tolist(), actual_impulse_Ns=run.metadata['realized_impulse_Ns'])
    summary = {**summarize_trial(run.log), 'reach_success': reach_ok, 'balance_success': balance_ok,
               'combined_success': reach_ok and balance_ok, 'recovery': asdict(run.recovery),
               'tracking_rmse_m': float(np.sqrt(np.mean(error**2))),
               'hold_max_goal_error_m': float(np.max(goal_error[hold])),
               'continuous_double_support': bool(np.all(run.log.arrays()['contact_left_post_step'])
                                                and np.all(run.log.arrays()['contact_right_post_step'])),
               'max_foot_displacement_m': float(np.max(run.log.arrays()['foot_xy_displacement_post_step'])),
               'max_foot_tangent_velocity_m_s': float(np.max(run.log.arrays()['foot_tangent_velocity_post_step'])),
               'final_goal_error_m': float(goal_error[-1]), 'provenance': metadata}
    if args.controller == 'pd_nominal_ff':
        summary['ik_max_reference_error_m'] = controller.ik_max_error_m
    extras = {}
    if args.controller == 'se3_wbc':
        extras = {'reach_weight_history': np.asarray(controller.weights), 'balance_guard_risk': np.asarray(controller.guard_risks),
                  'joint_reference_history': np.asarray(controller.joint_references)}
        reduced = extras['reach_weight_history'] < settings['weight']-1e-9
        summary['balance_guard'] = dict(enabled=bool(args.balance_guard),
                                       reduced_duration_s=float(np.sum(reduced)*cfg['robot']['control_timestep']),
                                       first_reduction_s=float(times[np.flatnonzero(reduced)[0]]) if np.any(reduced) else None,
                                       minimum_weight=float(np.min(extras['reach_weight_history'])),
                                       maximum_weight=float(np.max(extras['reach_weight_history'])))
    save_trial_npz(run.log, output/'trajectory.npz', metadata, {
        'qpos_history': np.asarray(run.qpos_history), 'qvel_history': np.asarray(run.qvel_history),
        'reach_point_world': points, 'reach_reference_world': references,
        'reach_goal_world': controller.goal, 'reach_error_m': error, **extras})
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
        overlay = [{'time_s': times[i], 'controller': args.controller, 'status': arrays['qp_status'][i],
                    'task_label': args.controller + ' | right-arm reach',
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
