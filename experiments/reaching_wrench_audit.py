"""Frozen contact-wrench feasibility and CoM response; no controller changes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linprog

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, write_csv
from reach_and_balance import ReachingController
from se3_whole_body_control.control.tasks import com_jacobian


def wrench_generators(points, frames, friction, dimensions, origin):
    """Outer box friction cone: rejection is informative, acceptance not proof.

    Include available spin/rolling moments. No cap on normal force. All supplied
    geometric contacts are allowed, including unloaded ones, enlarging the cone.
    """
    columns = []
    for point, frame, mu, dimension in zip(points, frames, friction, dimensions):
        axes = frame.T.copy()
        if axes[2, 0] < 0:
            axes *= -1
        moment_limits = [mu[2] if dimension>=4 else 0,
                         mu[3] if dimension>=6 else 0, mu[4] if dimension>=6 else 0]
        for signs in itertools.product((-1, 1), repeat=5):
            force = axes @ np.array([1, signs[0]*mu[0], signs[1]*mu[1]])
            torque = axes @ (np.array(signs[2:])*moment_limits)
            columns.append(np.r_[force, np.cross(point-origin, force)+torque])
    return np.column_stack(columns) if columns else np.zeros((6, 0))


def cone_residual(generators, wrench):
    # Fixed 0.1 m moment scale makes force and torque residuals comparable.
    scale = np.array([1, 1, 1, 10, 10, 10])
    target = scale*np.asarray(wrench)
    if not generators.shape[1]:
        return float(np.linalg.norm(target)/max(1, np.linalg.norm(target)))
    matrix = scale[:, None]*generators
    # LP avoids singular normal equations for redundant contact generators.
    result = linprog(np.r_[np.zeros(matrix.shape[1]), np.ones(12)],
                     A_eq=np.c_[matrix, np.eye(6), -np.eye(6)], b_eq=target,
                     bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun/max(1, np.linalg.norm(target, ord=1)))


def contact_cone(model, foot):
    contacts = [model.data.contact[i] for i in range(model.data.ncon)
                if model.geom_ids['ground'] in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                and any(g in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                        for g in model.foot_contact_geom_ids[foot])]
    origin = model.body_pose(('left_foot', 'right_foot')[foot])[:3, 3]
    points = [np.array(c.pos).copy() for c in contacts]
    frames = [np.array(c.frame).reshape(3, 3).copy() for c in contacts]
    friction = [np.array(c.friction).copy() for c in contacts]
    dimensions = [int(c.dim) for c in contacts]
    return wrench_generators(points, frames, friction, dimensions, origin), points, dimensions


def snapshot(path, time, cfg, initial):
    with np.load(path/'trajectory.npz', allow_pickle=False) as payload:
        a = {key: payload[key] for key in payload.files}
    summary = json.loads((path/'summary.json').read_text())
    i = int(np.argmin(abs(a['time_s']-time)))
    model = make_model(cfg)
    model.reset(a['qpos_history'][0], a['qvel_history'][0])
    controller = ReachingController(model, cfg['controller'], summary['provenance']['reaching'])
    controller.q_des = a['joint_reference_history'][i].copy()
    controller.pd_fallback.q_des = controller.q_des.copy()
    controller.T_des_torso, controller.T_des_pelvis = initial.desired_torso.copy(), initial.desired_pelvis.copy()
    controller.com_des = initial.com_reference.copy()
    model.reset(a['qpos_history'][i], a['qvel_history'][i])
    model.data.time = float(a['time_s'][i])
    result = controller.solve()
    assert result.success
    np.testing.assert_allclose(result.control, a['control'][i], rtol=1e-6, atol=1e-6)
    feet = []
    for foot in range(2):
        generators, points, dimensions = contact_cone(model, foot)
        measured = a['actual_contact_wrench'][i, 6*foot:6*foot+6]
        predicted = result.contact_wrench[6*foot:6*foot+6]
        feet.append(dict(foot=('left', 'right')[foot], contact_points_world_m=[p.tolist() for p in points],
                         contact_dimensions=dimensions, predicted_wrench_world=predicted.tolist(),
                         measured_pre_step_wrench_world=measured.tolist(),
                         predicted_outer_cone_relative_residual=cone_residual(generators, predicted),
                         measured_outer_cone_relative_residual=cone_residual(generators, measured)))
    com = model.center_of_mass()
    velocity = com_jacobian(model)@model.data.qvel
    desired_acceleration = -cfg['controller']['com_kp']*(com-controller.com_des)-cfg['controller']['com_kd']*velocity
    mass = float(np.sum(model.model.body_mass))
    gravity = np.array(model.model.opt.gravity)
    predicted_acceleration = result.contact_wrench.reshape(2, 6)[:, :3].sum(axis=0)/mass+gravity
    measured_acceleration = (a['actual_contact_wrench'][i].reshape(2, 6)[:, :3].sum(axis=0)+a['push_force'][i])/mass+gravity
    model.reset(a['qpos_history'][i+1], a['qvel_history'][i+1])
    next_velocity = com_jacobian(model)@model.data.qvel
    observed_acceleration = (next_velocity-velocity)/(a['time_s'][i+1]-a['time_s'][i])
    return dict(time_s=float(a['time_s'][i]), feet=feet,
                replay_control_max_error_Nm=float(np.max(abs(result.control-a['control'][i]))),
                com_error_world_m=(com-controller.com_des).tolist(), com_velocity_world_m_s=velocity.tolist(),
                com_task_acceleration_world_m_s2=desired_acceleration.tolist(),
                predicted_force_com_acceleration_world_m_s2=predicted_acceleration.tolist(),
                measured_force_com_acceleration_world_m_s2=measured_acceleration.tolist(),
                observed_interval_com_acceleration_world_m_s2=observed_acceleration.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    cases = dict(damping_hold=('damping_pilot', 'damping', 'hold', [2.996, 3.152, 3.44, 3.668, 4.284]),
                 pose_hold=('pose_pilot', 'pose', 'hold', [2.996, 3.152, 3.44, 3.8, 4.168]),
                 damping_moving=('damping_pilot', 'damping', 'moving', [1.496, 2.5, 4.5, 4.9]),
                 pose_moving=('pose_pilot', 'pose', 'moving', [1.496, 2.5, 4.5, 4.9]))
    records, rows = [], []
    for case, (folder, candidate, phase, times) in cases.items():
        path = ROOT/'results/reaching_contact'/folder/'trials'/f'front_12.5cm_contact_{candidate}_{phase}'
        samples = [snapshot(path, time, cfg, initial) for time in times]
        record = dict(case=case, path=path.relative_to(ROOT).as_posix(), snapshots=samples,
                      trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest())
        records.append(record)
        for sample in samples:
            for foot in sample['feet']:
                row = dict(case=case, time_s=sample['time_s'], foot=foot['foot'],
                           geometric_contacts=len(foot['contact_dimensions']),
                           predicted_residual=foot['predicted_outer_cone_relative_residual'],
                           measured_residual=foot['measured_outer_cone_relative_residual'])
                rows.append(row)
                print(json.dumps(row), flush=True)
    write_csv(rows, args.output/'wrench.csv')
    report = dict(records=records, physical_trials_run=0,
                  source_version=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  method='LP minimum L1 residual over outer box friction cone at all reconstructed geometric contacts, including available spin and rolling moments; normalized force/moment residual with fixed 0.1 m scale',
                  limitation='Instantaneous reconstructed manifold is not force-bearing solver history. Acceptance in outer cone does not prove feasibility. Rejecting a prediction alone does not establish failure causality. CoM force acceleration excludes other non-foot contacts; next-interval acceleration is a distinct quantity.')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    files = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
