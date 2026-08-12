"""Contact-constrained floating-base whole-body QP solved with OSQP."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy import sparse

from .joint_pd import JointPDController
from .tasks import com_jacobian, pose_task_acceleration, posture_task
from se3_whole_body_control.geometry.se3 import inverse_se3

try:
    import osqp
except ImportError:  # pragma: no cover
    osqp = None


@dataclass
class QPResult:
    control: np.ndarray
    qdd: np.ndarray
    contact_wrench: np.ndarray
    status: str
    success: bool
    solve_time_s: float
    primal_residual: float = float("nan")
    dual_residual: float = float("nan")
    objective: float = float("nan")
    friction_margin: float = float("nan")
    contact_slack_norm: float = 0.0
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


class WholeBodyQPController:
    """One-step QP controller.

    The contact wrench is ordered per foot as ``[Fx, Fy, Fz, Mx, My, Mz]`` in
    world coordinates. Foot Jacobians are evaluated at the named foot body
    origin and use MuJoCo's world-frame linear/angular convention.
    """

    def __init__(self, model, controller_config: dict, recovery_config: dict | None = None):
        self.model = model
        self.cfg = controller_config
        self.mu = float(controller_config.get("friction_coefficient", 0.7))
        self.nw = 12
        self.nslack = 12
        self.nx = model.nv + model.nu + self.nw + self.nslack
        self.q_des = model.joint_positions()
        self.T_des_torso = model.body_pose("torso")
        self.T_des_pelvis = model.body_pose("pelvis")
        self.com_des = model.center_of_mass()
        self.pd_fallback = JointPDController(
            model,
            kp=float(controller_config.get("posture_kp", 120.0)),
            kd=float(controller_config.get("posture_kd", 18.0)),
            q_des=self.q_des,
        )
        self.last_result: QPResult | None = None

    def _add_objective(self, P: np.ndarray, q: np.ndarray, A: np.ndarray, b: np.ndarray, weight: float) -> None:
        if weight <= 0 or A.size == 0:
            return
        P += 2.0 * weight * (A.T @ A)
        q -= 2.0 * weight * (A.T @ b)

    def _constraint(self, rows, lower, upper, row):
        rows.append(row)
        lower.append(row[1])
        upper.append(row[2])

    def _friction_rows(self, rows, lower, upper, start: int) -> None:
        mu = self.mu
        for foot in range(2):
            off = start + 6 * foot
            # Fz >= 0; moments are bounded for numerical conditioning.
            row = np.zeros(self.nx); row[off + 2] = 1.0
            rows.append(row); lower.append(0.0); upper.append(np.inf)
            for tangential in (0, 1):
                row = np.zeros(self.nx); row[off + tangential] = 1.0; row[off + 2] = -mu
                rows.append(row); lower.append(-np.inf); upper.append(0.0)
                row = np.zeros(self.nx); row[off + tangential] = -1.0; row[off + 2] = -mu
                rows.append(row); lower.append(-np.inf); upper.append(0.0)
            for moment in (3, 4, 5):
                row = np.zeros(self.nx); row[off + moment] = 1.0
                rows.append(row); lower.append(-500.0); upper.append(500.0)

    def _build_problem(self):
        nv, nu = self.model.nv, self.model.nu
        iw = nv + nu
        islack = iw + self.nw
        M = self.model.mass_matrix()
        h = self.model.data.qfrc_bias.copy()
        external = self.model.external_generalized_force()
        B = self.model.actuator_matrix
        Jc = self.model.contact_jacobian()
        contact_bias = self.model.contact_bias_acceleration()
        P = np.eye(self.nx) * 1e-9
        q = np.zeros(self.nx)

        torso_J, torso_b, torso_error = pose_task_acceleration(
            self.model.body_pose("torso"), self.T_des_torso, self.model.body_jacobian("torso"),
            self.model.data.qvel,
            float(self.cfg.get("torso_position_kp", 180.0)), float(self.cfg.get("torso_position_kd", 28.0)),
            float(self.cfg.get("torso_rotation_kp", 220.0)), float(self.cfg.get("torso_rotation_kd", 32.0)),
        )
        A = np.zeros((6, self.nx)); A[:, :nv] = torso_J
        self._add_objective(P, q, A, torso_b, float(self.cfg.get("qp_torso_weight", 20.0)))

        pelvis_J, pelvis_b, pelvis_error = pose_task_acceleration(
            self.model.body_pose("pelvis"), self.T_des_pelvis, self.model.body_jacobian("pelvis"),
            self.model.data.qvel,
            float(self.cfg.get("pelvis_position_kp", 140.0)), float(self.cfg.get("pelvis_position_kd", 24.0)),
            float(self.cfg.get("pelvis_rotation_kp", 160.0)), float(self.cfg.get("pelvis_rotation_kd", 26.0)),
        )
        A = np.zeros((6, self.nx)); A[:, :nv] = pelvis_J
        self._add_objective(P, q, A, pelvis_b, float(self.cfg.get("qp_pelvis_weight", 8.0)))

        Jcom = com_jacobian(self.model)
        bcom = -float(self.cfg.get("com_kp", 70.0)) * (self.model.center_of_mass() - self.com_des)
        bcom -= float(self.cfg.get("com_kd", 18.0)) * (Jcom @ self.model.data.qvel)
        A = np.zeros((3, self.nx)); A[:, :nv] = Jcom
        self._add_objective(P, q, A, bcom, float(self.cfg.get("qp_com_weight", 3.0)))

        Apost, bpost = posture_task(
            self.model.joint_positions(), self.q_des, self.model.joint_velocities(),
            self.model.joint_qvel_indices, nv,
            float(self.cfg.get("posture_kp", 120.0)), float(self.cfg.get("posture_kd", 18.0)),
        )
        A = np.zeros((Apost.shape[0], self.nx)); A[:, :nv] = Apost
        self._add_objective(P, q, A, bpost, float(self.cfg.get("qp_posture_weight", 2.0)))

        self._add_objective(P, q, np.eye(self.nx)[:nv], np.zeros(nv), float(self.cfg.get("qp_acceleration_weight", 0.02)))
        self._add_objective(P, q, np.eye(self.nx)[nv:nv + nu], np.zeros(nu), float(self.cfg.get("qp_torque_weight", 0.0005)))
        slack_selector = np.zeros((self.nslack, self.nx)); slack_selector[:, islack:] = np.eye(self.nslack)
        self._add_objective(P, q, slack_selector, np.zeros(self.nslack), float(self.cfg.get("qp_slack_weight", 100000.0)))
        # A contact wrench is an optimization variable, not an actuator input
        # that MuJoCo will apply after the solve. Bias the solution toward the
        # gravity-compensated posture torque so the simulated robot receives a
        # physically realizable equilibrium command.
        nominal_torque = self.pd_fallback.compute().control
        A_nominal = np.zeros((nu, self.nx)); A_nominal[:, nv:nv + nu] = np.eye(nu)
        self._add_objective(P, q, A_nominal, nominal_torque, float(self.cfg.get("qp_nominal_torque_weight", 0.5)))

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        dyn = np.zeros((nv, self.nx))
        dyn[:, :nv] = M
        dyn[:, nv:nv + nu] = -B
        dyn[:, iw:iw + self.nw] = -Jc.T
        for i in range(nv):
            rows.append(dyn[i]); lower.append(-h[i] + external[i]); upper.append(-h[i] + external[i])
        for i in range(Jc.shape[0]):
            row = np.zeros(self.nx); row[:nv] = Jc[i]; row[islack + i] = 1.0
            rows.append(row); lower.append(-contact_bias[i]); upper.append(-contact_bias[i])

        qdd_limit = float(self.cfg.get("max_joint_acceleration", 250.0))
        for i in range(nv):
            row = np.zeros(self.nx); row[i] = 1.0
            rows.append(row); lower.append(-qdd_limit); upper.append(qdd_limit)
        for i in range(nu):
            row = np.zeros(self.nx); row[nv + i] = 1.0
            rows.append(row); lower.append(float(self.model.actuator_limits[i, 0])); upper.append(float(self.model.actuator_limits[i, 1]))
        self._friction_rows(rows, lower, upper, iw)
        Acons = sparse.csc_matrix(np.vstack(rows))
        return P, q, Acons, np.asarray(lower), np.asarray(upper), torso_error, pelvis_error

    def _fallback(self, message: str, elapsed: float) -> QPResult:
        pd = self.pd_fallback.compute()
        result = QPResult(
            control=pd.control, qdd=np.zeros(self.model.nv), contact_wrench=np.zeros(self.nw),
            status="fallback_pd", success=False, solve_time_s=elapsed, message=message,
        )
        self.last_result = result
        return result

    def solve(self) -> QPResult:
        start = time.perf_counter()
        if osqp is None:
            return self._fallback("osqp is not installed", time.perf_counter() - start)
        try:
            P, q, A, l, u, torso_error, pelvis_error = self._build_problem()
            solver = osqp.OSQP()
            settings = self.cfg.get("solver", {})
            solver.setup(
                P=sparse.csc_matrix((P + P.T) * 0.5), q=q, A=A, l=l, u=u,
                verbose=False, eps_abs=float(settings.get("eps_abs", 1e-4)),
                eps_rel=float(settings.get("eps_rel", 1e-4)),
                max_iter=int(settings.get("max_iter", 4000)), polish=bool(settings.get("polish", True)),
                adaptive_rho=bool(settings.get("adaptive_rho", True)),
                scaled_termination=bool(settings.get("scaled_termination", True)),
            )
            sol = solver.solve()
            elapsed = time.perf_counter() - start
            info = sol.info
            status = str(info.status)
            ok = status.lower() in {"solved", "solved inaccurate"} and sol.x is not None and np.all(np.isfinite(sol.x))
            if not ok:
                return self._fallback(f"OSQP status: {status}", elapsed)
            x = np.asarray(sol.x)
            iw = self.model.nv + self.model.nu
            wrench = x[iw:iw + self.nw]
            contact_slack_norm = float(np.linalg.norm(x[iw + self.nw:iw + self.nw + self.nslack]))
            margins = []
            for foot in range(2):
                off = 6 * foot
                fz = wrench[off + 2]
                margins.extend([self.mu * fz - abs(wrench[off]), self.mu * fz - abs(wrench[off + 1]), fz])
            result = QPResult(
                control=np.clip(x[self.model.nv:self.model.nv + self.model.nu], self.model.actuator_limits[:, 0], self.model.actuator_limits[:, 1]),
                qdd=x[:self.model.nv], contact_wrench=wrench, status=status, success=True,
                solve_time_s=elapsed, primal_residual=float(getattr(info, "prim_res", np.nan)),
                dual_residual=float(getattr(info, "dual_res", np.nan)), objective=float(getattr(info, "obj_val", np.nan)),
                friction_margin=float(np.min(margins)),
                contact_slack_norm=contact_slack_norm,
                diagnostics={"torso_se3_error": torso_error, "pelvis_se3_error": pelvis_error},
            )
            self.last_result = result
            return result
        except Exception as exc:  # keep the simulation observable and recoverable
            return self._fallback(f"QP exception: {type(exc).__name__}: {exc}", time.perf_counter() - start)
