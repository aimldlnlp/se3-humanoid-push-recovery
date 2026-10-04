"""Experimental condim=3 contact forces and matching material-point kinematics."""
import numpy as np
from scipy import sparse

from reach_and_balance import ReachingController


def contact_spec(model, foot, name):
    contacts = [model.data.contact[i] for i in range(model.data.ncon)
                if model.geom_ids['ground'] in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                and any(g in (model.data.contact[i].geom1, model.data.contact[i].geom2)
                        for g in model.foot_contact_geom_ids[foot])]
    if any(c.dim!=3 for c in contacts):
        raise ValueError('contact-mode prototype supports only condim=3')
    pose = model.body_pose(name)
    points = [np.array(c.pos).copy() for c in contacts]
    # Inner diamond rays, consistent with the existing L1 friction convention.
    generators = []
    jacobians, biases = [], []
    for c, point in zip(contacts, points):
        axes = np.array(c.frame).reshape(3, 3).T.copy()
        if axes[2, 0]<0:
            axes *= -1
        for tangent in ([c.friction[0],0],[-c.friction[0],0],[0,c.friction[1]],[0,-c.friction[1]]):
            force = axes@np.r_[1,tangent]
            generators.append(np.r_[force,np.cross(point-pose[:3,3],force)])
        _, J, bias = model.attached_point_kinematics(name, pose[:3,:3].T@(point-pose[:3,3]))
        jacobians.append(J)
        biases.append(bias)
    G = np.column_stack(generators) if generators else np.zeros((6,0))
    J = np.vstack(jacobians) if jacobians else np.zeros((0,model.nv))
    bias = np.concatenate(biases) if biases else np.zeros(0)
    # Rank of rigid-body velocity constraints: point=3, edge=5, flat patch=6.
    maps = [np.c_[np.eye(3), -np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])]
            for r in (point-pose[:3,3] for point in points)]
    rank = int(np.linalg.matrix_rank(np.vstack(maps),tol=1e-7)) if maps else 0
    return dict(G=G,J=J,bias=bias,points=points,rank=rank)


class ContactModeController(ReachingController):
    """Separate opt-in prototype; the library controller remains untouched."""
    def _friction_rows(self, rows, lower, upper, start):
        first = len(rows)
        super()._friction_rows(rows,lower,upper,start)
        keep = [first+11*foot+i for foot in range(len(self.contact_names)) for i in range(5)]
        rows[first:] = [rows[i] for i in keep]
        lower[first:] = [lower[i] for i in keep]
        upper[first:] = [upper[i] for i in keep]

    def _build_problem(self):
        base_size = self.model.nv+self.model.nu+self.nw+self.nslack
        self.nx = base_size
        problem = list(super()._build_problem())
        P,q,A,low,high = problem[:5]
        self.specs = [contact_spec(self.internal_model,('left_foot','right_foot').index(name),name)
                      for name in self.contact_names]
        force_count = sum(s['G'].shape[1] for s in self.specs)
        point_count = sum(s['J'].shape[0] for s in self.specs)
        self.nx = base_size+force_count+point_count
        expanded_P = np.eye(self.nx)*1e-9
        expanded_P[:base_size,:base_size] = P
        expanded_q = np.r_[q,np.zeros(force_count+point_count)]
        # Replace body-origin contact rows. Old slack variables stay fixed at zero.
        keep = np.r_[np.arange(self.model.nv),np.arange(self.model.nv+self.nw,A.shape[0])]
        matrix = np.pad(A.toarray()[keep],((0,0),(0,force_count+point_count)))
        lower,upper = low[keep].tolist(),high[keep].tolist()
        rows = list(matrix)
        iw = self.model.nv+self.model.nu
        for i in range(iw+self.nw,base_size):
            row = np.zeros(self.nx); row[i]=1
            rows.append(row); lower.append(0); upper.append(0)
        alpha,slack = base_size,base_size+force_count
        damping = self.settings.get('contact_velocity_damping_s_inv',0)
        for foot,spec in enumerate(self.specs):
            G,J,bias = spec['G'],spec['J'],spec['bias']
            for i in range(6):
                row = np.zeros(self.nx)
                row[iw+6*foot+i]=1
                row[alpha:alpha+G.shape[1]]=-G[i]
                rows.append(row); lower.append(0); upper.append(0)
            for i in range(G.shape[1]):
                row = np.zeros(self.nx); row[alpha+i]=1
                rows.append(row); lower.append(0); upper.append(np.inf)
            target = -bias-damping*(J@self.internal_model.data.qvel)
            for i in range(J.shape[0]):
                row = np.zeros(self.nx); row[:self.model.nv]=J[i]; row[slack+i]=1
                rows.append(row); lower.append(float(target[i])); upper.append(float(target[i]))
                # Per-foot average point acceleration penalty avoids contact-count gain.
                expanded_P[slack+i,slack+i] += 2*self.cfg['qp_slack_weight']/max(1,len(spec['points']))
            alpha += G.shape[1]
            slack += J.shape[0]
        problem[:5] = [expanded_P,expanded_q,sparse.csc_matrix(np.array(rows)),np.array(lower),np.array(upper)]
        return tuple(problem)

    def solve(self):
        result = super().solve()
        if not hasattr(self,'mode_ranks'):
            self.mode_ranks,self.mode_counts = [],[]
        self.mode_ranks.append([s['rank'] for s in self.specs])
        self.mode_counts.append([len(s['points']) for s in self.specs])
        if result.success:
            damping = self.settings.get('contact_velocity_damping_s_inv',0)
            residual = np.concatenate([s['J']@result.qdd+s['bias']+damping*(s['J']@self.internal_model.data.qvel)
                                       for s in self.specs])
            result.contact_slack_norm = float(np.linalg.norm(residual))
            # Here this field is the physical target deviation, before slack;
            # original row-budget validation still checks the augmented equalities.
            result.contact_acceleration_residual_norm = float(np.linalg.norm(residual))
            result.diagnostics.update(contact_motion_ranks=[s['rank'] for s in self.specs],
                                      contact_point_counts=[len(s['points']) for s in self.specs],
                                      contact_model='condim3_point_force_inner_cone',
                                      contact_residual_definition='Point acceleration target deviation before slack')
        return result
