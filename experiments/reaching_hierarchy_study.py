"""Frozen numerical gate, then fixed-input hierarchical WBC rollout comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, write_csv
from reaching_hierarchy import HierarchicalController
from reaching_pose_origin import PoseOriginController
from reaching_pose_validation import CASES
from reaching_tracking_study import verify_pair
from reaching_workspace import run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cases = [('nominal', 0, 0, 1.5), ('moving', 70, 0, 1.5), *CASES]
    plan = dict(source_version=source, cases=cases,
                hierarchy=['contact_slack', 'weighted_torso_pelvis_com', 'reach', 'posture_and_regularization'],
                formulation='Hard equalities eliminated; attained outputs preserved by null-space parameterization',
                output_validation_tolerance='1e-7*(1+abs(attained weighted output)); no lock bands',
                numerical_rank_relative_threshold=1e-10,
                gains_changed=False, contact_model_changed=False, physical_thresholds_changed=False,
                gate='All frozen states accepted before rollout. Improve at least one failed case, retain every prior PASS and <=15 mm hand tolerance.',
                scope='One candidate, no gain or lock-tolerance search. No production promotion.')
    (root/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    cfg = load_configs(ROOT, robot_name='unitree_g1')
    initial = prepare_paired_initial_condition(cfg)
    audit = json.loads((ROOT/'results/reaching_contact/failure_onset_audit/audit.json').read_text())
    frozen = []
    for record in audit['records']:
        path = ROOT/record['trajectory_path']
        assert hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest() == record['trajectory_sha256']
        summary = json.loads((path/'summary.json').read_text())
        with np.load(path/'trajectory.npz', allow_pickle=False) as a:
            for sample in record['snapshots']:
                i = sample['index']
                for variant in (PoseOriginController, HierarchicalController):
                    model = make_model(cfg)
                    model.reset(a['qpos_history'][0], a['qvel_history'][0])
                    controller = variant(model, cfg['controller'], summary['provenance']['reaching'])
                    controller.q_des = a['joint_reference_history'][i].copy()
                    controller.pd_fallback.q_des = controller.q_des.copy()
                    controller.T_des_torso = initial.desired_torso.copy()
                    controller.T_des_pelvis = initial.desired_pelvis.copy()
                    controller.com_des = initial.com_reference.copy()
                    model.reset(a['qpos_history'][i], a['qvel_history'][i])
                    model.data.time = float(a['time_s'][i])
                    model.set_external_force(cfg['experiments']['push']['application_body'], a['push_force'][i])
                    model.data.ctrl[:] = a['control'][i]
                    mujoco.mj_forward(model.model, model.data)
                    result = controller.solve()
                    if variant is PoseOriginController:
                        assert result.success, result.message
                        np.testing.assert_allclose(result.control, a['control'][i], rtol=1e-6, atol=1e-6)
                    row = dict(case=record['case'], index=i, time_s=float(a['time_s'][i]),
                               variant=variant.__name__, success=result.success, message=result.message,
                               solve_time_s=result.solve_time_s,
                               constraint_budget_ratio=result.diagnostics.get('constraint_budget_ratio'),
                               hierarchy_max_lock_excess=result.diagnostics.get('hierarchy_max_lock_excess'),
                               hierarchy_levels=result.diagnostics.get('hierarchy_levels'))
                    if variant is HierarchicalController and hasattr(controller, 'baseline_problem'):
                        before, after = controller.baseline_problem, controller.problem
                        n = before[2].shape[0]
                        np.testing.assert_array_equal(before[2].toarray(), after[2][:n].toarray())
                        np.testing.assert_array_equal(before[3], after[3][:n])
                        np.testing.assert_array_equal(before[4], after[4][:n])
                    frozen.append(row)
    (root/'frozen.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    candidates = [r for r in frozen if r['variant']=='HierarchicalController']
    gate = all(r['success'] for r in candidates)
    print('Frozen gate:', gate, 'accepted', sum(r['success'] for r in candidates), '/', len(candidates), flush=True)
    rows = []
    if gate:
        for case, force, direction, start in cases:
            if case in ('nominal', 'moving'):
                suffix = 'no_push' if case=='nominal' else 'moving'
                baseline = ROOT/'results/reaching_contact/pose_origin_pilot/trials'/('front_12.5cm_contact_pose_origin_'+suffix)
            else:
                baseline = ROOT/'results/reaching_contact/pose_disturbance_validation/trials'/('front_12.5cm_pose_validation_'+case)
            name = 'hierarchy_'+case
            row = run_trial(root, name, [.125, 0, 0], force, start, stage='hierarchy_validation',
                            reach_weight=3000, contact_velocity_damping=20, contact_mode=True,
                            pose_origin=True, hierarchical=True, push_direction_deg=direction)
            trial = root/'trials'/name
            verify_pair(baseline, trial)
            old = json.loads((baseline/'summary.json').read_text())
            new = json.loads((trial/'summary.json').read_text())
            for key in ('reaching', 'initial_condition', 'model_sha256', 'dependency_versions', 'objective_weight_overrides'):
                assert old['provenance'][key] == new['provenance'][key], key
            with np.load(trial/'trajectory.npz', allow_pickle=False) as a:
                assert 'qp_solve_time_s' in a.files
                times = a['qp_solve_time_s']
                row.update(solve_median_ms=float(np.median(times)*1000),
                           solve_p95_ms=float(np.percentile(times, 95)*1000),
                           solve_max_ms=float(np.max(times)*1000))
            row.update(case=case, baseline_success=old['combined_success'])
            rows.append(row)
            write_csv(rows, root/'study.csv')
    improved = sum(r['combined_success'] and not r['baseline_success'] for r in rows)
    regressions = sum(not r['combined_success'] and r['baseline_success'] for r in rows)
    report = dict(source_version=source, frozen_gate_passed=gate, frozen_states=len(candidates),
                  frozen_accepted=sum(r['success'] for r in candidates), trials=rows,
                  physical_trials_run=len(rows), improved_cases=improved, regressions=regressions,
                  candidate_gate_passed=bool(gate and improved and not regressions),
                  decision='Numerically rejected before rollout' if not gate else
                  ('Physical comparison gate passed; still experimental' if improved and not regressions else 'Do not promote candidate'),
                  production_promoted=False, hardware_trials_run=0)
    (root/'study.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    files = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'manifest.json').write_text(json.dumps(dict(files=files), indent=2), encoding='utf-8')
    print(report['decision'], flush=True)


if __name__ == '__main__':
    main()
