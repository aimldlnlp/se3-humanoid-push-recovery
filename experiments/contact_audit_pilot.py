"""Small paired physics/runtime/mismatch pilot; never overwrites old results."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

import common
from common import (
    ROOT, execution_manifest, load_configs, make_push, prepare_paired_initial_condition,
    run_trial, save_paired_initial_condition, save_run, write_csv,
)
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController


def timing_statistics(seconds, deadline_s):
    values = np.asarray(seconds, dtype=float)
    if not len(values):
        return {}
    return {
        'mean_ms': float(values.mean() * 1000),
        'p95_ms': float(np.percentile(values, 95) * 1000),
        'p99_ms': float(np.percentile(values, 99) * 1000),
        'max_ms': float(values.max() * 1000),
        'deadline_ms': float(deadline_s * 1000),
        'deadline_miss_percent': float(np.mean(values > deadline_s) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise FileExistsError(f'refusing to overwrite existing pilot: {root}')
    root.mkdir(parents=True)
    configs = load_configs(ROOT, robot_name='unitree_g1')
    condition = prepare_paired_initial_condition(configs, setup_duration_s=0.6)
    save_paired_initial_condition(condition, root / 'conditions' / 'nominal.npz')
    rows, timing_rows = [], []
    # Observe production calls without changing the task or solver output.
    class TimedController(WholeBodyQPController):
        def solve(self):
            start = time.perf_counter()
            result = super().solve()
            timing_rows.append({
                'call_time_s': time.perf_counter() - start,
                'build_time_s': result.diagnostics.get('qp_build_time_s', 0.0),
                'prepare_time_s': result.diagnostics.get('qp_prepare_time_s', 0.0),
                'numerical_solve_time_s': result.diagnostics.get('qp_numerical_solve_time_s', 0.0),
                'workspace_reused': result.diagnostics.get('workspace_reused', False),
                'mass_relative_error': result.diagnostics.get('internal_mass_relative_error', 0.0),
            })
            return result

    common.WholeBodyQPController = TimedController

    def trial(name, controller='se3_wbc', magnitude=None, direction=0, reuse=False,
              initial=condition, mass=1.0, friction=0.7, unknown=False):
        local = deepcopy(configs)
        local['controller']['solver']['reuse_workspace'] = reuse
        local['controller']['friction_coefficient'] = 0.7 if unknown else friction
        push = None if magnitude is None else make_push(local, magnitude=magnitude, direction_deg=direction)
        duration = 6.0 if push is None else push.start_time_s + push.duration_s + local['experiments']['recovery']['timeout_s'] + 0.1
        offset = len(timing_rows)
        plant, run = run_trial(
            controller, local, push=push, duration=duration, classify=True,
            initial_condition=initial, mass_scale=mass, friction_coefficient=friction,
            controller_mass_scale=1.0 if unknown else None,
            controller_friction_coefficient=0.7 if unknown else None,
        )
        # Bootstrap solves precede the experiment for nominal-feedforward PD.
        samples = timing_rows[offset:] if controller == 'se3_wbc' else []
        arrays = run.log.arrays()
        if not np.all(arrays['numerical_valid']):
            raise RuntimeError(f'non-finite state in {name}')
        for sample in timing_rows[offset:]:
            sample['trial_id'] = name
        save_run(run, root / 'raw' / f'{name}.npz', {
            'trial_id': name, 'controller': controller, 'config': local,
            'push': asdict(push) if push is not None else None,
        })
        row = {
            'trial_id': name, 'controller': controller, 'seed': initial.seed,
            'reuse_workspace': reuse, 'unknown_plant_parameters': unknown,
            'plant_mass_scale': mass, 'plant_friction': friction,
            'controller_mass_scale': 1.0 if unknown else mass,
            'controller_friction': 0.7 if unknown else friction,
            'push_magnitude_N': magnitude or 0, 'push_direction_deg': direction,
            **asdict(run.recovery),
            'initial_condition_sha256': initial.sha256,
            'force_trace_sha256': run.metadata['force_trace_sha256'],
            'realized_impulse_Ns': run.metadata['realized_impulse_Ns'],
            'qp_failure_count': int(np.sum(~arrays['qp_success'])),
            'workspace_reuse_percent': float(100 * np.mean([r['workspace_reused'] for r in samples])) if samples else None,
            'peak_actual_grf_N': float(np.max(arrays['actual_contact_wrench'][:, [2, 8]])),
            'peak_predicted_coulomb_utilization': float(np.max(
                np.linalg.norm(arrays['predicted_contact_wrench'].reshape(-1, 2, 6)[:, :, :2], axis=2)
                / np.maximum(local['controller']['friction_coefficient'] * arrays['predicted_contact_wrench'][:, [2, 8]], 1e-9)
            )) if controller == 'se3_wbc' else None,
            'max_internal_mass_relative_error': max((s['mass_relative_error'] for s in samples), default=0),
            **{f'call_{key}': value for key, value in timing_statistics(
                [s['call_time_s'] for s in samples], local['robot']['control_timestep'],
            ).items()},
        }
        rows.append(row)
        write_csv(rows, root / 'pilot.csv')
        print(json.dumps(row), flush=True)

    for label, magnitude, direction in (
        ('quiet', None, 0), ('canonical', 70, 0),
        ('lateral_40', 40, 90), ('lateral_70', 70, 90),
    ):
        for reuse in (False, True):
            trial(f'{label}_{"warm" if reuse else "cold"}', magnitude=magnitude, direction=direction, reuse=reuse)
    for controller in ('pure_pd', 'pd_nominal_ff'):
        trial(f'canonical_{controller}', controller=controller, magnitude=70)
    # Reproducible contact-preserving angular-rate and push perturbations.
    # Both model-knowledge conditions see exactly the same plant/state/force.
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        velocity = condition.qvel.copy()
        velocity[3:6] += rng.normal(0, 0.01, 3)
        initial = replace(condition, qvel=velocity, seed=seed)
        save_paired_initial_condition(initial, root / 'conditions' / f'seed{seed}.npz')
        magnitude = float(70 * rng.uniform(0.92, 1.08))
        direction = float(rng.normal(0, 8))
        for factor, mass, friction in (('mass', 1.1, 0.7), ('friction', 1.0, 0.4)):
            for unknown in (False, True):
                trial(f'{factor}_seed{seed}_{"unknown" if unknown else "known"}',
                      magnitude=magnitude, direction=direction, initial=initial,
                      mass=mass, friction=friction, unknown=unknown)
    write_csv(timing_rows, root / 'timing.csv')
    manifest = execution_manifest({'config': configs, 'run_id': root.name})
    manifest['command'] = 'python experiments/contact_audit_pilot.py --output <isolated-output>'
    summary = {'manifest': manifest, 'trials': rows, 'scope': '18-trial pilot; not a replacement push sweep',
               'friction_model': 'world-XY L1 inner approximation; not an exact distributed contact wrench cone',
               'timing_note': 'Entire production solve call; build, preparation, numerical solve measured separately. No hard-real-time claim.'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
