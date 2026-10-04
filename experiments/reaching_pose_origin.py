"""Opt-in body-origin pose acceleration correction; production stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, write_csv
from reaching_task_audit import TaskAuditController, objective_records
from se3_whole_body_control.control.tasks import pose_task_acceleration
from se3_whole_body_control.geometry.so3 import hat_so3


def body_origin_target(current, desired, J, qvel, bias, gains):
    """Return J*qdd target, with a world-aligned body-origin residual metric."""
    H = np.eye(6)
    H[:3, 3:] = hat_so3(current[:3, 3])
    velocity = J @ qvel
    _, spatial_target, _ = pose_task_acceleration(current, desired, H @ J, qvel, *gains)
    derivative_offset = np.r_[np.cross(velocity[:3], velocity[3:]), np.zeros(3)]
    return np.linalg.solve(H, spatial_target-derivative_offset) - bias


class PoseOriginController(TaskAuditController):
    """Only the first two pose objective targets change; contact equations stay."""
    def _add_objective(self, P, q, A, b, weight):
        index = len(self.objectives)
        if index < 2:
            body = ('torso', 'pelvis')[index]
            model = self.internal_model
            current = model.body_pose(body)
            J = model.body_jacobian(body)
            np.testing.assert_allclose(A[:, :model.nv], J, rtol=0, atol=0)
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacDot(model.model, model.data, jacp, jacr, current[:3, 3], model.body_ids[body])
            bias = np.vstack([jacp, jacr]) @ model.data.qvel
            gains = [float(self.cfg[body+'_'+key]) for key in
                     ('position_kp', 'position_kd', 'rotation_kp', 'rotation_kd')]
            b = body_origin_target(current, getattr(self, 'T_des_'+body), J, model.data.qvel, bias, gains)
        return super()._add_objective(P, q, A, b, weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    old = json.loads((ROOT/'results/reaching_contact/task_audit/audit.json').read_text())
    path = ROOT/old['trajectory_path']
    assert hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest() == old['trajectory_sha256']
    with np.load(path/'trajectory.npz', allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    summary = json.loads((path/'summary.json').read_text())
    records, flat = [], []
    for sample in old['records']:
        i = int(np.argmin(abs(arrays['time_s']-sample['time_s'])))
        pair = []
        for controller_type in (TaskAuditController, PoseOriginController):
            model = make_model(cfg)
            model.reset(arrays['qpos_history'][0], arrays['qvel_history'][0])
            controller = controller_type(model, cfg['controller'], summary['provenance']['reaching'])
            controller.q_des = arrays['joint_reference_history'][i].copy()
            controller.pd_fallback.q_des = controller.q_des.copy()
            controller.T_des_torso, controller.T_des_pelvis = initial.desired_torso.copy(), initial.desired_pelvis.copy()
            controller.com_des = initial.com_reference.copy()
            model.reset(arrays['qpos_history'][i], arrays['qvel_history'][i])
            model.data.time = float(arrays['time_s'][i])
            result = controller.solve()
            assert result.success, result.message
            if controller_type is TaskAuditController:
                np.testing.assert_allclose(result.control, arrays['control'][i], rtol=1e-6, atol=1e-6)
            tasks, cost, _ = objective_records(controller)
            row = dict(variant=controller_type.__name__, time_s=float(arrays['time_s'][i]), objectives=tasks,
                       objective_sum=cost, constraint_budget_ratio=result.diagnostics['constraint_budget_ratio'],
                       torque_peak_Nm=float(np.max(abs(result.control))), max_qdd=float(np.max(abs(result.qdd))),
                       control_difference_from_saved_Nm=float(np.max(abs(result.control-arrays['control'][i]))))
            pair.append((controller, row))
        baseline, candidate = pair[0][0], pair[1][0]
        for a, b in zip(baseline.problem[2:5], candidate.problem[2:5]):
            np.testing.assert_array_equal(a.toarray() if hasattr(a,'toarray') else a,
                                          b.toarray() if hasattr(b,'toarray') else b)
        # The geometric Jacobian and metric do not change P. Only pose targets change q.
        np.testing.assert_array_equal(baseline.problem[0], candidate.problem[0])
        for original, corrected in zip(baseline.objectives[2:], candidate.objectives[2:]):
            for a, b in zip(original, corrected):
                np.testing.assert_array_equal(a, b)
        records.append(dict(time_s=float(arrays['time_s'][i]), baseline=pair[0][1], candidate=pair[1][1],
                            constraints_identical=True, quadratic_matrix_identical=True, non_pose_objectives_identical=True))
        for _, row in pair:
            flat.extend(dict(time_s=row['time_s'],variant=row['variant'],**task) for task in row['objectives'])
        print(sample['time_s'], 'frozen gate passed', flush=True)
    write_csv(flat, args.output/'objectives.csv')
    report = dict(records=records, frozen_gate_passed=True, physical_trials_run=0,
                  source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  trajectory_path=old['trajectory_path'], trajectory_sha256=old['trajectory_sha256'],
                  production_controller_changed=False, task_weights_changed=False,
                  scope='One pose-origin target correction, with bias compensation and unchanged body-origin metric. No gain search or default promotion.',
                  limitation='Frozen QP differences are not physical recovery results. SE(3) resolved acceleration is still approximate.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files = [dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
