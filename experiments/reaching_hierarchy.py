"""Opt-in null-space hierarchy with the original physical constraint budget."""
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import null_space

from reaching_pose_origin import PoseOriginController
from reaching_task_audit import LABELS
from se3_whole_body_control.control.whole_body_qp import QPResult, WholeBodyQPController


def solve_level(P, q, A, low, high, settings, initial=None):
    """Solve one reduced convex QP; scaling changes no mathematical minimizer."""
    scale = max(float(np.max(np.abs(P))), float(np.max(np.abs(q))), 1)
    A = sparse.csc_matrix(A)
    solver = osqp.OSQP()
    absolute = min(settings.get('eps_abs', 1e-4), 1e-8)
    relative = min(settings.get('eps_rel', 1e-4), 1e-8)
    solver.setup(P=sparse.triu(sparse.csc_matrix(P/scale), format='csc'), q=q/scale,
                 A=A, l=low, u=high, verbose=False, eps_abs=absolute, eps_rel=relative,
                 max_iter=settings.get('max_iter', 4000), polishing=settings.get('polish', True),
                 adaptive_rho=settings.get('adaptive_rho', True), scaled_termination=False)
    if initial is not None:
        solver.warm_start(x=initial)
    result = solver.solve()
    def accepted(result):
        return (result.info.status.lower() in ('solved', 'solved inaccurate')
                and result.x is not None and np.all(np.isfinite(result.x))
                and WholeBodyQPController._constraint_budget_ratio(
                    A, result.x, low, high, absolute, relative) <= 1)
    if not accepted(result):
        solver.update_settings(eps_abs=min(absolute, 1e-9), eps_rel=min(relative, 1e-9))
        result = solver.solve()
    if not accepted(result):
        raise RuntimeError('Reduced hierarchy level rejected: '+result.info.status)
    return result.x


def solve_hierarchy(levels, A, low, high, settings):
    """Eliminate hard equalities, then preserve attained outputs in null spaces."""
    dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
    equality = np.isfinite(low) & np.isfinite(high) & (low == high)
    E, target = dense[equality], low[equality]
    row_scale = np.maximum(np.linalg.norm(E, axis=1), 1e-12)
    E, target = E/row_scale[:, None], target/row_scale
    x = np.linalg.lstsq(E, target, rcond=1e-10)[0]
    basis = null_space(E, rcond=1e-10)
    records = []
    for name, matrix, desired in levels:
        reduced = matrix@basis
        if basis.shape[1] and matrix.shape[0]:
            C = dense[~equality]@basis
            shift = dense[~equality]@x
            # Normalize inequalities only for conditioning; validate originals below.
            row_scale = np.maximum(np.linalg.norm(C, axis=1), 1)
            z = solve_level(2*reduced.T@reduced, 2*reduced.T@(matrix@x-desired),
                            C/row_scale[:, None], (low[~equality]-shift)/row_scale,
                            (high[~equality]-shift)/row_scale, settings, np.zeros(basis.shape[1]))
            x = x+basis@z
        ratio = WholeBodyQPController._constraint_budget_ratio(
            A, x, low, high, settings.get('eps_abs', 1e-4), settings.get('eps_rel', 1e-4))
        if ratio > 1:
            raise RuntimeError(f'{name}: original constraint budget={ratio:.6g}')
        records.append(dict(name=name, matrix=matrix, output=matrix@x,
                            residual_norm=float(np.linalg.norm(matrix@x-desired)),
                            free_dimensions=basis.shape[1]))
        if reduced.shape[0] and reduced.shape[1]:
            _, singular, vectors = np.linalg.svd(reduced, full_matrices=True)
            # Roundoff-only projections must not consume a null direction.
            rank = int(np.sum(singular > 1e-10*max(float(np.linalg.norm(matrix, 2)), 1)))
            basis = basis@vectors[rank:].T
    for record in records:
        tolerance = 1e-7*(1+np.abs(record['output']))
        if np.any(np.abs(record['matrix']@x-record['output']) > tolerance):
            raise RuntimeError('Higher-priority output changed: '+record['name'])
    return x, records


class HierarchicalController(PoseOriginController):
    """Contact slack > weighted balance > reach > remaining baseline costs."""
    def _build_problem(self):
        problem = super()._build_problem()
        self.baseline_problem = self.problem = problem
        nx = len(problem[1])
        tasks = {name: (np.pad(matrix, ((0, 0), (0, nx-matrix.shape[1]))), desired, weight)
                 for name, (matrix, desired, weight) in zip(LABELS, self.objectives)}
        count = sum(spec['J'].shape[0] for spec in self.specs)
        contact = np.eye(nx)[nx-count:] if count else np.zeros((0, nx))
        start = 0
        for spec in self.specs:
            size = spec['J'].shape[0]
            contact[start:start+size] *= np.sqrt(self.cfg['qp_slack_weight']/max(1, len(spec['points'])))
            start += size
        levels = [('contact_slack', contact, np.zeros(count))]
        for name, names in (('balance', ('torso', 'pelvis', 'com')), ('reach', ('reach',)),
                            ('remaining', tuple(n for n in LABELS if n not in ('torso', 'pelvis', 'com', 'reach')))):
            selected = [tasks[n] for n in names]
            matrix = np.vstack([np.sqrt(weight)*m for m, b, weight in selected])
            desired = np.concatenate([np.sqrt(weight)*b for m, b, weight in selected])
            if name == 'remaining':
                matrix = np.vstack([matrix, np.sqrt(.5e-9)*np.eye(nx)])
                desired = np.r_[desired, np.zeros(nx)]
            levels.append((name, matrix, desired))
        self.levels = levels
        return problem

    def solve(self):
        start = time.perf_counter()
        self._update_reach_reference()
        self.hierarchy_levels = []
        try:
            self._sync_internal_model()
            P, q, A, low, high, torso_error, pelvis_error, M, h, B, Jc, bias, external = self._build_problem()
            x, self.hierarchy_levels = solve_hierarchy(self.levels, A, low, high, self.cfg.get('solver', {}))
            nv, nu = self.model.nv, self.model.nu
            tau = np.clip(x[nv:nv+nu], self.model.actuator_limits[:, 0], self.model.actuator_limits[:, 1])
            x[nv:nv+nu] = tau
            ratio = self._constraint_budget_ratio(A, x, low, high, self.cfg['solver']['eps_abs'], self.cfg['solver']['eps_rel'])
            drift = max((float(np.max(np.abs(v['matrix']@x-v['output'])/(1+np.abs(v['output']))))
                         for v in self.hierarchy_levels if len(v['output'])), default=0)
            if ratio > 1 or drift > 1e-7:
                raise RuntimeError(f'Final validation failed: budget={ratio:.6g}, output drift={drift:.6g}')
            wrench = x[nv+nu:nv+nu+self.nw]
            margin = min(min(self.mu*w[2]-abs(w[0])-abs(w[1]), w[2]) for w in wrench.reshape(-1, 6))
            result = QPResult(control=tau, qdd=x[:nv], contact_wrench=wrench, status='solved hierarchy',
                              success=True, solve_time_s=0, friction_margin=float(margin),
                              objective=float(.5*x@P@x+q@x),
                              dynamics_residual_norm=float(np.linalg.norm(M@x[:nv]+h-B@tau-Jc.T@wrench-external)),
                              diagnostics=dict(constraint_budget_ratio=ratio, hierarchy_max_lock_excess=drift,
                                               hierarchy_levels=[dict(name=v['name'], residual_norm=v['residual_norm'],
                                                                      free_dimensions=v['free_dimensions']) for v in self.hierarchy_levels],
                                               external_force_oracle=bool(self.cfg.get('use_external_force_oracle', False)),
                                               active_contacts=list(self.contact_names), swing_foot=self.swing_foot,
                                               torso_se3_error=torso_error, pelvis_se3_error=pelvis_error))
        except Exception as exc:
            result = self._fallback(f'Hierarchy exception: {type(exc).__name__}: {exc}', time.perf_counter()-start)
        result = self._record_contact_result(result)
        result.solve_time_s = time.perf_counter()-start
        self.last_result = result
        return result
