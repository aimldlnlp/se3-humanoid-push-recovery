"""Frozen-state objective audit and paired full-trajectory ablations.

Only soft posture and nominal-torque objectives are removed. This is an
exploratory audit on known failures, not tuning or held-out validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, write_csv
from reach_and_balance import ReachingController
from reaching_tracking_study import verify_pair
from reaching_workspace import assess_trial, run_trial

MODES = {'original': {}, 'no_posture': {'qp_posture_weight': 0},
         'no_nominal': {'qp_nominal_torque_weight': 0},
         'neither': {'qp_posture_weight': 0, 'qp_nominal_torque_weight': 0}}
OBJECTIVES = ('torso', 'pelvis', 'com', 'posture', 'reach', 'acceleration', 'torque', 'slack', 'nominal_torque')


class ObjectiveAuditController(ReachingController):
    def _build_problem(self):
        self.objectives = []
        problem = super()._build_problem()
        assert len(self.objectives) == len(OBJECTIVES), 'update audit labels after QP topology changes'
        self.constraint_hash = hashlib.sha256(b''.join(
            np.ascontiguousarray(v).tobytes() for v in (problem[2].toarray(), problem[3], problem[4]))).hexdigest()
        return problem

    def _add_objective(self, P, q, A, b, weight):
        self.objectives.append((A.copy(), b.copy(), weight))
        super()._add_objective(P, q, A, b, weight)


def frozen_audit(path, sample_times):
    import mujoco
    summary = json.loads((path/'summary.json').read_text())
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    settings = summary['provenance']['reaching']
    rows = []
    with np.load(path/'trajectory.npz', allow_pickle=False) as a:
        for time in sample_times:
            i = int(np.argmin(abs(a['time_s']-time)))
            hashes = set()
            for mode, overrides in MODES.items():
                model = make_model(cfg)
                model.reset(a['qpos_history'][0], a['qvel_history'][0])
                controller = ObjectiveAuditController(model, {**cfg['controller'], **overrides}, settings)
                controller.q_des = initial.joint_reference.copy()
                controller.pd_fallback.q_des = initial.joint_reference.copy()
                controller.T_des_torso, controller.T_des_pelvis = initial.desired_torso.copy(), initial.desired_pelvis.copy()
                controller.com_des = initial.com_reference.copy()
                model.reset(a['qpos_history'][i], a['qvel_history'][i])
                model.data.time = float(a['time_s'][i])
                if np.any(a['push_force'][i]):
                    model.set_external_force(cfg['experiments']['push']['application_body'], a['push_force'][i])
                mujoco.mj_forward(model.model, model.data)
                result = controller.solve()
                assert result.success, (time, mode, result.message)
                hashes.add(controller.constraint_hash)
                if mode == 'original':
                    np.testing.assert_allclose(result.control, a['control'][i], rtol=1e-7, atol=1e-7)
                slack = -model.contact_jacobian()@result.qdd-model.contact_bias_acceleration()
                x = np.r_[result.qdd, result.control, result.contact_wrench, slack]
                costs, gradients, residuals = {}, {}, {}
                for name, (A, b, weight) in zip(OBJECTIVES, controller.objectives):
                    residual = A@x-b
                    costs[name] = float(weight*np.dot(residual, residual))
                    residuals[name] = float(np.linalg.norm(residual))
                    gradients[name] = 2*weight*A.T@residual
                g, h = gradients['reach'][:model.nv], gradients['posture'][:model.nv]
                norm = np.linalg.norm(g)*np.linalg.norm(h)
                rows.append(dict(time_s=float(a['time_s'][i]), mode=mode, objective_costs=costs,
                                 objective_residual_norms=residuals,
                                 reach_posture_gradient_cosine=float(np.dot(g, h)/norm) if norm > 1e-12 else None,
                                 control_delta_norm_Nm=float(np.linalg.norm(result.control-a['control'][i])),
                                 predicted_normal_forces_N=result.contact_wrench.reshape(2,6)[:,2].tolist(),
                                 recorded_actual_normal_forces_N=a['actual_contact_wrench'][i].reshape(2,6)[:,2].tolist(),
                                 recorded_contacts=a['contact_left'][i].item() and a['contact_right'][i].item(),
                                 constraint_hash=controller.constraint_hash))
            assert len(hashes)==1, 'an ablation changed physical constraints'
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    rows, frozen = [], {}
    for phase, magnitude, start in (('no_push', 0, 1.5), ('moving', 70, 1.5), ('hold', 70, 3.0)):
        baseline_name = 'front_12.5cm_w3000'+(f'_{phase}_70N' if magnitude else '')
        baseline = args.previous/'trials'/baseline_name
        if magnitude:
            frozen[phase] = frozen_audit(baseline, [start-.004, start+.076, start+.152,
                                                    2.852 if phase=='moving' else 3.772])
        for mode, overrides in MODES.items():
            name = f'front_12.5cm_w3000_{mode}_{phase}'
            if mode == 'original':
                shutil.copytree(baseline, root/'trials'/name)
                s, passed, reason = assess_trial(root/'trials'/name)
                row = dict(trial_id=name, stage='reused_baseline', push_N=magnitude, push_start_s=start,
                           combined_success=passed, balance_success=s['balance_success'], failure_reason=reason,
                           hold_max_error_mm=1000*s['hold_max_goal_error_m'], qp_failures=s['qp_failures'],
                           source_version=s['provenance']['source_version'],
                           initial_condition_sha256=s['provenance']['initial_condition']['sha256'])
            else:
                row = run_trial(root, name, [.125,0,0], magnitude, start, stage='ablation', reach_weight=3000,
                                objective_overrides=overrides)
            verify_pair(baseline, root/'trials'/name)
            row.update(mode=mode, phase=phase, objective_overrides=overrides)
            rows.append(row)
            fields = ('trial_id', 'mode', 'phase', 'push_N', 'push_start_s', 'combined_success',
                      'balance_success', 'failure_reason', 'hold_max_error_mm', 'qp_failures', 'source_version')
            write_csv([{key: r[key] for key in fields} for r in rows], root/'study.csv')
    assert len({r['initial_condition_sha256'] for r in rows})==1
    payload = dict(trials=rows, frozen_state_audit=frozen, paired_inputs_verified=True,
                   baseline_regression_passed=False, baseline_policy='Reused immutable baseline; original controls reproduced at all eight sampled states',
                   scope='Exploratory objective ablations on a known front-reaching failure; no default change',
                   objective_order=OBJECTIVES, source_version=next(r['source_version'] for r in rows if r['stage']=='ablation'))
    (root/'study.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    import matplotlib.pyplot as plt
    from se3_whole_body_control.visualization.style import apply_style
    apply_style()
    fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
    values = np.array([[next(r['combined_success'] for r in rows if r['mode']==m and r['phase']==p)
                        for p in ('no_push','moving','hold')] for m in MODES], dtype=int)
    from matplotlib.colors import ListedColormap
    ax.imshow(values, cmap=ListedColormap(['#f2d5d9','#d8ede3']), vmin=0, vmax=1, aspect='auto')
    for i, mode in enumerate(MODES):
        for j, phase in enumerate(('no_push','moving','hold')):
            row = next(r for r in rows if r['mode']==mode and r['phase']==phase)
            ax.text(j,i,f"{row['failure_reason']}\n{row['hold_max_error_mm']:.1f} mm",ha='center',va='center',fontsize=10)
    ax.set(xticks=range(3), xticklabels=['No push','Moving: 70 N','Holding: 70 N'],
           yticks=range(4), yticklabels=list(MODES), title='Front 12.5 cm | reach weight 3000 | final-hold error')
    for suffix in ('png','pdf'):
        fig.savefig(root/f'conflict_results.{suffix}',dpi=250)
    plt.close(fig)
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
