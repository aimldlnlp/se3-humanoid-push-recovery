"""Qualified PD+nominal-feedforward comparator with offline arm-only IK."""
import numpy as np
from scipy.optimize import least_squares

from common import ROOT, load_configs, make_model, run_controller
from se3_whole_body_control.control.joint_pd import JointPDController
from se3_whole_body_control.control.tasks import quintic_reference


class ReachingPDController(JointPDController):
    def __init__(self, model, controller_config, settings):
        cfg = load_configs(ROOT, robot_name='unitree_g1')
        nominal = run_controller('pd_nominal_ff', model, cfg)
        super().__init__(model, kp=nominal.kp, kd=nominal.kd,
                         q_des=nominal.q_des, feedforward_control=nominal.feedforward_control)
        self.settings = settings
        self.start = model.attached_point_kinematics(settings['body_name'], settings['point_local_m'])[0]
        self.goal = self.start + np.asarray(settings['target_offset_world_m'])
        self.points, self.references = [], []
        self.reference_times = np.linspace(0, settings['movement_duration_s'], 61)
        self.joint_references, errors = [], []
        arm = np.array([i for i, name in enumerate(model.joint_names)
                        if name.startswith('right_') and any(part in name for part in ('shoulder','elbow','wrist'))])
        if len(arm) != 7:
            raise ValueError('expected seven right-arm joints')
        ik = make_model(cfg)
        qpos = model.data.qpos.copy()
        nominal_q = model.joint_positions().copy()
        lo, hi = model.joint_position_limits()
        previous = nominal_q[arm].copy()
        def residual(joints, target):
            qpos[ik.joint_qpos_indices[arm]] = joints
            ik.reset(qpos, np.zeros(ik.nv))
            point, _, _ = ik.attached_point_kinematics(settings['body_name'], settings['point_local_m'])
            return np.r_[point-target, .001*(joints-previous)]
        for t in self.reference_times:
            target = quintic_reference(self.start, self.goal, t, settings['movement_duration_s'])[0]
            result = least_squares(residual, np.clip(previous,lo[arm]+1e-8,hi[arm]-1e-8),
                                   args=(target,), bounds=(lo[arm],hi[arm]),
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=150)
            previous = result.x.copy()
            joints = nominal_q.copy()
            joints[arm] = previous
            self.joint_references.append(joints)
            errors.append(float(np.linalg.norm(residual(previous,target)[:3])))
        self.joint_references = np.asarray(self.joint_references)
        self.ik_max_error_m = max(errors)
        if self.ik_max_error_m > .001:
            raise ValueError(f'IK reference cannot represent target within 1 mm: {self.ik_max_error_m}')

    def compute(self):
        t = self.model.data.time-self.settings['start_time_s']
        self.q_des = np.array([np.interp(t,self.reference_times,self.joint_references[:,j])
                               for j in range(self.model.nu)])
        self.points.append(self.model.attached_point_kinematics(
            self.settings['body_name'],self.settings['point_local_m'])[0].copy())
        self.references.append(quintic_reference(self.start,self.goal,t,self.settings['movement_duration_s'])[0])
        return super().compute()
