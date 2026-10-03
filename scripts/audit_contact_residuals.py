"""Audit force-unit residuals without modifying torques or recovery labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def friction_audit(wrenches, mu, eps_abs, eps_rel):
    """Engineering row budget, not OSQP's scaled global stopping guarantee.

    Predeclared from solver settings: abs + rel * max(lhs, rhs), in N.
    Always report the raw residual too. The physical Coulomb circle is audited
    independently; the L1 approximation is not the recovery classifier.
    """
    w = np.asarray(wrenches, dtype=float)
    if w.shape[-1] != 6 or not np.all(np.isfinite(w)):
        raise ValueError('expected finite [..., 6] contact wrenches')
    if mu <= 0 or eps_abs <= 0 or eps_rel < 0:
        raise ValueError('invalid friction or tolerance')
    tangent = np.sum(np.abs(w[..., :2]), axis=-1)
    capacity = mu * w[..., 2]
    excess = np.maximum(tangent - capacity, 0)
    budget = eps_abs + eps_rel * np.maximum(tangent, np.abs(capacity))
    normal_excess = np.maximum(-w[..., 2], 0)
    normal_budget = eps_abs + eps_rel * np.abs(w[..., 2])
    circle_excess = np.maximum(np.linalg.norm(w[..., :2], axis=-1) - capacity, 0)
    return {
        'max_l1_excess_N': float(excess.max()),
        'max_l1_budget_ratio': float((excess / budget).max()),
        'max_coulomb_excess_N': float(circle_excess.max()),
        'max_negative_normal_N': float(normal_excess.max()),
        'row_budget_pass': bool(np.all(excess <= budget) and np.all(normal_excess <= normal_budget)),
        'strict_1mN_pass': bool(np.all(excess <= 0.001) and np.all(normal_excess <= 0.001)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.raw_root.rglob('*.npz')):
        metadata_path = path.with_suffix('.json')
        if not metadata_path.exists():
            continue
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta.get('controller') != 'se3_wbc':
            continue
        cfg = meta['config']['controller']
        settings = cfg['solver']
        with np.load(path, allow_pickle=False) as data:
            successful = data['qp_success'].astype(bool)
            if not np.any(successful):
                raise RuntimeError(f'no successful QP calls: {path.name}')
            wrench = data['predicted_contact_wrench'][successful].reshape(-1, 2, 6)
            record = friction_audit(wrench, cfg['friction_coefficient'], settings['eps_abs'], settings['eps_rel'])
            record.update({
                'trial_id': path.stem, 'qp_failure_count': int(np.sum(~successful)),
                'max_dynamics_residual': float(np.max(data['dynamics_residual_norm'][successful])),
                'max_contact_equation_residual': float(np.max(data['contact_acceleration_residual_norm'][successful])),
            })
        records.append(record)
    if not records:
        raise RuntimeError('no WBC trajectories audited')
    payload = {'scope': 'successful QP predictions; contact equation includes slack',
               'budget': 'eps_abs + eps_rel * max(|Fx|+|Fy|, |mu Fz|), in N; not a solver guarantee',
               'all_row_budgets_pass': all(r['row_budget_pass'] for r in records),
               'trials': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'trials'}))


if __name__ == '__main__':
    main()
