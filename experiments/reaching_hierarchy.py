"""Opt-in sequential QPs; attained higher-priority outputs are locked."""
import numpy as np
import osqp
from scipy import sparse

from reaching_pose_origin import PoseOriginController
from reaching_task_audit import LABELS


def solve_level(P, q, A, low, high, settings):
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(sparse.csc_matrix(P), format='csc'), q=q,
                 A=sparse.csc_matrix(A), l=low, u=high, verbose=False,
                 eps_abs=settings.get('eps_abs', 1e-4), eps_rel=settings.get('eps_rel', 1e-4),
                 max_iter=settings.get('max_iter', 4000), polish=settings.get('polish', True),
                 adaptive_rho=settings.get('adaptive_rho', True),
                 scaled_termination=settings.get('scaled_termination', True))
    result = solver.solve()
    from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
    def accepted(result):
        return (result.info.status.lower() in ('solved', 'solved inaccurate')
                and result.x is not None and np.all(np.isfinite(result.x))
                and WholeBodyQPController._constraint_budget_ratio(
                    sparse.csc_matrix(A), result.x, low, high,
                    settings.get('eps_abs', 1e-4), settings.get('eps_rel', 1e-4)) <= 1)
    if not accepted(result):
        solver.update_settings(eps_abs=min(settings.get('eps_abs', 1e-4), 1e-6),
                               eps_rel=min(settings.get('eps_rel', 1e-4), 1e-6),
                               scaled_termination=False)
        result = solver.solve()
    if not accepted(result):
        raise RuntimeError('Hierarchy level rejected: '+result.info.status)
    return result.x


class HierarchicalController(PoseOriginController):
    """Contact slack > weighted balance > reach > remaining baseline costs."""
    def _build_problem(self):
        problem = list(super()._build_problem())
        self.baseline_problem = tuple(problem)
        P, q, A, low, high = problem[:5]
        nx = len(q)
        tasks = {name: (np.pad(matrix, ((0, 0), (0, nx-matrix.shape[1]))), target, weight)
                 for name, (matrix, target, weight) in zip(LABELS, self.objectives)}
        point_count = sum(spec['J'].shape[0] for spec in self.specs)
        contact = np.eye(nx)[nx-point_count:] if point_count else np.zeros((0, nx))
        start = 0
        for spec in self.specs:
            count = spec['J'].shape[0]
            contact[start:start+count] *= np.sqrt(self.cfg['qp_slack_weight']/max(1, len(spec['points'])))
            start += count
        balance = [tasks[name] for name in ('torso', 'pelvis', 'com')]
        matrices = [contact, np.vstack([np.sqrt(w)*m for m, b, w in balance]),
                    np.sqrt(tasks['reach'][2])*tasks['reach'][0]]
        targets = [np.zeros(point_count), np.concatenate([np.sqrt(w)*b for m, b, w in balance]),
                   np.sqrt(tasks['reach'][2])*tasks['reach'][1]]
        remaining_P, remaining_q = P.copy(), q.copy()
        self.hierarchy_levels = []
        for name, matrix, target in zip(('contact_slack', 'balance', 'reach'), matrices, targets):
            level_P = 2*matrix.T@matrix
            level_q = -2*matrix.T@target
            remaining_P -= level_P
            remaining_q -= level_q
            if not len(matrix):
                continue
            x = solve_level(level_P, level_q, A, low, high, self.cfg.get('solver', {}))
            # Row normalization changes only lock conditioning, not the task cost.
            matrix = matrix/np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)[:, None]
            output = matrix@x
            # Lock attained outputs, not desired targets: infeasible tasks remain feasible.
            tolerance = 1e-5*(1+np.abs(output))
            A = sparse.vstack([A, sparse.csc_matrix(matrix)], format='csc')
            low, high = np.r_[low, output-tolerance], np.r_[high, output+tolerance]
            self.hierarchy_levels.append(dict(name=name, matrix=matrix, output=output,
                                              tolerance=tolerance))
        problem[:5] = [(remaining_P+remaining_P.T)*.5, remaining_q, A, low, high]
        self.problem = tuple(problem)
        return self.problem

    def solve(self):
        result = super().solve()
        levels = getattr(self, 'hierarchy_levels', [])
        result.diagnostics['hierarchy_levels'] = [dict(name=v['name']) for v in levels]
        if result.success:
            result.diagnostics['hierarchy_max_lock_excess'] = max(
                (float(np.max(np.abs(v['matrix']@self.solution-v['output'])-v['tolerance']))
                 for v in levels), default=0)
        return result
