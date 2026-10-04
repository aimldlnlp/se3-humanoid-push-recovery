"""Audit pose-task conventions on saved states without changing the controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, write_csv
from reaching_task_audit import TaskAuditController
from se3_whole_body_control.control.tasks import pose_task_acceleration
from se3_whole_body_control.geometry.se3 import adjoint_se3, inverse_se3, log_se3
from se3_whole_body_control.geometry.so3 import hat_so3, log_so3


def measure(model, body, desired, gains, qdd, dt=1e-6):
    """Central directional differences; restore state even if a check fails."""
    qpos, qvel = model.data.qpos.copy(), model.data.qvel.copy()
    current = model.body_pose(body)
    J = model.body_jacobian(body)
    velocity = J @ qvel
    H = np.eye(6)
    H[:3, 3:] = hat_so3(current[:3, 3])
    spatial_J = H @ J
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacDot(model.model, model.data, jacp, jacr, current[:3, 3], model.body_ids[body])
    geometric_bias = np.vstack([jacp, jacr]) @ qvel
    spatial_bias = H @ geometric_bias + np.r_[np.cross(velocity[:3], velocity[3:]), np.zeros(3)]
    poses, jacobians, spatial_velocities = [], [], []
    try:
        for sign in (-1, 1):
            displaced = qpos.copy()
            mujoco.mj_integratePos(model.model, displaced, qvel, sign * dt)
            model.reset(displaced, qvel)
            T = model.body_pose(body)
            Ji = model.body_jacobian(body)
            Hi = np.eye(6)
            Hi[:3, 3:] = hat_so3(T[:3, 3])
            poses.append(T)
            jacobians.append(Ji)
            spatial_velocities.append(Hi @ Ji @ qvel)
    finally:
        model.reset(qpos, qvel)
    geometric_fd = np.r_[(poses[1][:3, 3] - poses[0][:3, 3]) / (2 * dt),
                         log_so3(poses[1][:3, :3] @ poses[0][:3, :3].T) / (2 * dt)]
    spatial_fd = log_se3(poses[1] @ inverse_se3(poses[0])) / (2 * dt)
    geometric_bias_fd = (jacobians[1] - jacobians[0]) @ qvel / (2 * dt)
    spatial_bias_fd = (spatial_velocities[1] - spatial_velocities[0]) / (2 * dt)
    np.testing.assert_allclose(velocity, geometric_fd, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(spatial_J @ qvel, spatial_fd, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(geometric_bias, geometric_bias_fd, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(spatial_bias, spatial_bias_fd, atol=2e-6, rtol=2e-6)

    _, original_target, _ = pose_task_acceleration(current, desired, J, qvel, *gains)
    _, spatial_target, _ = pose_task_acceleration(current, desired, spatial_J, qvel, *gains)
    # A world-origin translation leaves a geometric Jacobian unchanged.
    shift = np.eye(4)
    shift[:3, 3] = [0.7, -0.4, 0.2]
    _, shifted_target, _ = pose_task_acceleration(shift @ current, shift @ desired, J, qvel, *gains)
    shifted_spatial_J = adjoint_se3(shift) @ spatial_J
    _, shifted_spatial_target, _ = pose_task_acceleration(
        shift @ current, shift @ desired, shifted_spatial_J, qvel, *gains)
    np.testing.assert_allclose(shifted_spatial_target, adjoint_se3(shift) @ spatial_target,
                               atol=2e-8, rtol=2e-8)
    # Convert the diagnostic spatial acceleration target back to body-origin
    # acceleration, including its derivative offset. No controller uses it.
    point_target = np.linalg.solve(H, spatial_target - spatial_bias) + geometric_bias
    spatial_residual = spatial_J @ qdd + spatial_bias - spatial_target
    return dict(body=body, dt_s=dt, velocity_geometric_fd_error=float(np.max(abs(velocity-geometric_fd))),
                velocity_spatial_fd_error=float(np.max(abs(spatial_J@qvel-spatial_fd))),
                raw_jacobian_spatial_velocity_error=float(np.linalg.norm(velocity-spatial_fd)),
                geometric_bias_fd_error=float(np.max(abs(geometric_bias-geometric_bias_fd))),
                spatial_bias_fd_error=float(np.max(abs(spatial_bias-spatial_bias_fd))),
                raw_target_origin_shift_change=float(np.linalg.norm(shifted_target-original_target)),
                spatial_target_adjoint_error=float(np.max(abs(shifted_spatial_target-adjoint_se3(shift)@spatial_target))),
                original_target=original_target.tolist(), diagnostic_body_origin_target=point_target.tolist(),
                diagnostic_target_difference_norm=float(np.linalg.norm(point_target-original_target)),
                omitted_geometric_bias=geometric_bias.tolist(),
                original_residual=(J@qdd-original_target).tolist(),
                diagnostic_spatial_residual=spatial_residual.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    old = json.loads((ROOT/'results/reaching_contact/task_audit/audit.json').read_text())
    trajectory = ROOT/old['trajectory_path']/'trajectory.npz'
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == old['trajectory_sha256']
    with np.load(trajectory, allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    summary = json.loads((trajectory.parent/'summary.json').read_text())
    records = []
    for snapshot in old['records']:
        i = int(np.argmin(abs(arrays['time_s']-snapshot['time_s'])))
        model = make_model(cfg)
        model.reset(arrays['qpos_history'][0], arrays['qvel_history'][0])
        controller = TaskAuditController(model, cfg['controller'], summary['provenance']['reaching'])
        controller.q_des = arrays['joint_reference_history'][i].copy()
        controller.pd_fallback.q_des = controller.q_des.copy()
        controller.T_des_torso, controller.T_des_pelvis = initial.desired_torso.copy(), initial.desired_pelvis.copy()
        controller.com_des = initial.com_reference.copy()
        model.reset(arrays['qpos_history'][i], arrays['qvel_history'][i])
        model.data.time = float(arrays['time_s'][i])
        result = controller.solve()
        assert result.success, result.message
        np.testing.assert_allclose(result.control, arrays['control'][i], rtol=1e-6, atol=1e-6)
        for body, desired in (('torso', initial.desired_torso), ('pelvis', initial.desired_pelvis)):
            gains = [float(cfg['controller'][body+'_'+key]) for key in
                     ('position_kp', 'position_kd', 'rotation_kp', 'rotation_kd')]
            for dt in (1e-6, 5e-7):
                row = measure(model, body, desired, gains, result.qdd, dt)
                row.update(time_s=float(arrays['time_s'][i]),
                           control_replay_max_error_Nm=float(np.max(abs(result.control-arrays['control'][i]))))
                records.append(row)
        print(snapshot['time_s'], 'frame checks passed', flush=True)
    write_csv(records, args.output/'frames.csv')
    report = dict(records=records, source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  trajectory_path=old['trajectory_path'], trajectory_sha256=old['trajectory_sha256'],
                  physical_trials_run=0, controller_changed=False, task_weights_changed=False,
                  origin_translation_m=[0.7,-0.4,0.2],
                  finding='MuJoCo body Jacobian is geometric at body origin, not spatial at world origin. Production pose task mixes conventions and omits pose Jdot*qvel.',
                  limitation='Diagnostic conversion only, no QP or rollout change. Spatial linear/angular values have mixed units. Frame correction is not proof of recovery; SE(3) resolved acceleration remains approximate.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
