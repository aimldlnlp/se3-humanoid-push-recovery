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
    dynamics_residual_norm: float = float("nan")
    contact_acceleration_residual_norm: float = float("nan")
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


class WholeBodyQPController:
    """One-step QP controller.

    The contact wrench is ordered per foot as ``[Fx, Fy, Fz, Mx, My, Mz]`` in
    world coordinates. Foot Jacobians are evaluated at the named foot body
    origin and use MuJoCo's world-frame linear/angular convention.
    """

    def __init__(self, model, controller_config: dict, recovery_config: dict | None = None, internal_model=None):
        self.model = model
        # ``model`` is the physical plant receiving the returned torque.
        # ``internal_model`` is optional and exists to make plant/controller
        # mismatch explicit in diagnostics and tests.  The normal project
        # path leaves it unset, so the controller uses the same model.
        self.internal_model = internal_model or model
        if (self.internal_model.nq, self.internal_model.nv, self.internal_model.nu) != (model.nq, model.nv, model.nu):
            raise ValueError("plant and internal controller models must have identical dimensions")
        self.cfg = controller_config
        self.mu = float(controller_config.get("friction_coefficient", 0.7))
        self.contact_names = ("left_foot", "right_foot")
        self.nw = 0
        self.nslack = 0
        self.nx = 0
        self.set_active_contacts(self.contact_names)
        self.q_des = model.joint_positions()
        self.T_des_torso = model.body_pose("torso")
        self.T_des_pelvis = model.body_pose("pelvis")
        self.com_des = model.center_of_mass()
        self.swing_foot: str | None = None
        self.swing_target: np.ndarray | None = None
        self.com_task_weight_override: float | None = None
        self.com_task_kp_override: float | None = None
        self.com_task_kd_override: float | None = None
        self.pd_fallback = JointPDController(
            self.internal_model,
            kp=float(controller_config.get("posture_kp", 120.0)),
            kd=float(controller_config.get("posture_kd", 18.0)),
            q_des=self.q_des,
        )
        self.last_result: QPResult | None = None
        self._solver = None
        self._solver_patterns = None

    def set_active_contacts(self, contact_names: tuple[str, ...] | list[str]) -> None:
        """Select the feet constrained as fixed contacts in the next solve.

        Contact mode is deliberately explicit.  A one-foot mode removes the
        swing foot from both the dynamics wrench variables and the fixed-foot
        acceleration constraints; the physical simulator still reports any
        accidental ground contact separately.
        """
        names = tuple(str(name) for name in contact_names)
        valid = {"left_foot", "right_foot"}
        if not names or any(name not in valid for name in names) or len(set(names)) != len(names):
            raise ValueError(f"active contacts must be a non-empty subset of {sorted(valid)}")
        self.contact_names = names
        self.nw = 6 * len(names)
        self.nslack = self.nw
        self.nx = self.model.nv + self.model.nu + self.nw + self.nslack

    def set_swing_target(self, foot_name: str | None, target_pose: np.ndarray | None = None) -> None:
        """Set or clear a Cartesian target for the non-contact swing foot."""
        if foot_name is None:
            self.swing_foot = None
            self.swing_target = None
            return
        foot_name = str(foot_name)
        if foot_name not in {"left_foot", "right_foot"} or foot_name in self.contact_names:
            raise ValueError("swing foot must be a named foot outside the active contact set")
        target = np.asarray(target_pose, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("swing target must be a finite 4x4 pose")
        self.swing_foot = foot_name
        self.swing_target = target.copy()

    def _sync_internal_model(self) -> None:
        """Put an optional internal model at the plant state before solving."""
        if self.internal_model is self.model:
            return
        self.internal_model.reset(qpos=self.model.data.qpos.copy(), qvel=self.model.data.qvel.copy())
        # Copying the applied wrench is only relevant to the explicitly
        # enabled oracle diagnostic.  The primary controller ignores it.
        self.internal_model.data.xfrc_applied[:] = self.model.data.xfrc_applied

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
        cop_x_min = float(self.cfg.get("support_polygon_x_min_m", -0.115))
        cop_x_max = float(self.cfg.get("support_polygon_x_max_m", 0.225))
        cop_y_min = float(self.cfg.get("support_polygon_y_min_m", -0.12))
        cop_y_max = float(self.cfg.get("support_polygon_y_max_m", 0.12))
        torsional_mu = float(self.cfg.get("torsional_friction_coefficient", 0.02))
        for foot in range(len(self.contact_names)):
            off = start + 6 * foot
            # Inner Coulomb approximation: |Fx| + |Fy| <= mu Fz. Independent
            # component bounds admit sqrt(2) excess friction at their corners.
            row = np.zeros(self.nx); row[off + 2] = 1.0
            rows.append(row); lower.append(0.0); upper.append(np.inf)
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                row = np.zeros(self.nx)
                row[off:off + 3] = [sx, sy, -mu]
                rows.append(row); lower.append(-np.inf); upper.append(0.0)
            # For a horizontal rectangular support patch, Mx = y*Fz and
            # My = -x*Fz. These are CoP limits, not arbitrary moment boxes.
            row = np.zeros(self.nx); row[off + 3] = 1.0; row[off + 2] = -cop_y_min
            rows.append(row); lower.append(0.0); upper.append(np.inf)
            row = np.zeros(self.nx); row[off + 3] = 1.0; row[off + 2] = -cop_y_max
            rows.append(row); lower.append(-np.inf); upper.append(0.0)
            row = np.zeros(self.nx); row[off + 4] = 1.0; row[off + 2] = cop_x_max
            rows.append(row); lower.append(0.0); upper.append(np.inf)
            row = np.zeros(self.nx); row[off + 4] = 1.0; row[off + 2] = cop_x_min
            rows.append(row); lower.append(-np.inf); upper.append(0.0)
            row = np.zeros(self.nx); row[off + 5] = 1.0; row[off + 2] = -torsional_mu
            rows.append(row); lower.append(-np.inf); upper.append(0.0)
            row = np.zeros(self.nx); row[off + 5] = -1.0; row[off + 2] = -torsional_mu
            rows.append(row); lower.append(-np.inf); upper.append(0.0)

    def _build_problem(self):
        model = self.internal_model
        nv, nu = model.nv, model.nu
        iw = nv + nu
        islack = iw + self.nw
        M = model.mass_matrix()
        h = model.data.qfrc_bias.copy()
        external = model.external_generalized_force() if bool(self.cfg.get("use_external_force_oracle", False)) else np.zeros(nv)
        B = model.actuator_matrix
        Jc = model.contact_jacobian(self.contact_names)
        contact_bias = model.contact_bias_acceleration(foot_names=self.contact_names)
        P = np.eye(self.nx) * 1e-9
        q = np.zeros(self.nx)

        torso_J, torso_b, torso_error = pose_task_acceleration(
            model.body_pose("torso"), self.T_des_torso, model.body_jacobian("torso"),
            model.data.qvel,
            float(self.cfg.get("torso_position_kp", 180.0)), float(self.cfg.get("torso_position_kd", 28.0)),
            float(self.cfg.get("torso_rotation_kp", 220.0)), float(self.cfg.get("torso_rotation_kd", 32.0)),
        )
        A = np.zeros((6, self.nx)); A[:, :nv] = torso_J
        self._add_objective(P, q, A, torso_b, float(self.cfg.get("qp_torso_weight", 20.0)))

        pelvis_J, pelvis_b, pelvis_error = pose_task_acceleration(
            model.body_pose("pelvis"), self.T_des_pelvis, model.body_jacobian("pelvis"),
            model.data.qvel,
            float(self.cfg.get("pelvis_position_kp", 140.0)), float(self.cfg.get("pelvis_position_kd", 24.0)),
            float(self.cfg.get("pelvis_rotation_kp", 160.0)), float(self.cfg.get("pelvis_rotation_kd", 26.0)),
        )
        A = np.zeros((6, self.nx)); A[:, :nv] = pelvis_J
        self._add_objective(P, q, A, pelvis_b, float(self.cfg.get("qp_pelvis_weight", 8.0)))

        Jcom = com_jacobian(model)
        com_kp = self.com_task_kp_override if self.com_task_kp_override is not None else float(self.cfg.get("com_kp", 70.0))
        com_kd = self.com_task_kd_override if self.com_task_kd_override is not None else float(self.cfg.get("com_kd", 18.0))
        com_weight = self.com_task_weight_override if self.com_task_weight_override is not None else float(self.cfg.get("qp_com_weight", 3.0))
        bcom = -com_kp * (model.center_of_mass() - self.com_des)
        bcom -= com_kd * (Jcom @ model.data.qvel)
        A = np.zeros((3, self.nx)); A[:, :nv] = Jcom
        self._add_objective(P, q, A, bcom, com_weight)

        Apost, bpost = posture_task(
            model.joint_positions(), self.q_des, model.joint_velocities(),
            model.joint_qvel_indices, nv,
            float(self.cfg.get("posture_kp", 120.0)), float(self.cfg.get("posture_kd", 18.0)),
        )
        A = np.zeros((Apost.shape[0], self.nx)); A[:, :nv] = Apost
        self._add_objective(P, q, A, bpost, float(self.cfg.get("qp_posture_weight", 2.0)))

        if self.swing_foot is not None and self.swing_target is not None:
            swing_J, swing_b, _ = pose_task_acceleration(
                model.body_pose(self.swing_foot), self.swing_target,
                model.body_jacobian(self.swing_foot), model.data.qvel,
                float(self.cfg.get("swing_position_kp", 180.0)),
                float(self.cfg.get("swing_position_kd", 24.0)),
                float(self.cfg.get("swing_rotation_kp", 80.0)),
                float(self.cfg.get("swing_rotation_kd", 14.0)),
            )
            A = np.zeros((6, self.nx)); A[:, :nv] = swing_J
            self._add_objective(P, q, A, swing_b, float(self.cfg.get("qp_swing_weight", 35.0)))

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
            # Torque limits belong to the physical actuator interface.  The
            # internal model may differ in mass/friction, but cannot change
            # what torque the plant can safely receive.
            rows.append(row); lower.append(float(self.model.actuator_limits[i, 0])); upper.append(float(self.model.actuator_limits[i, 1]))
        self._friction_rows(rows, lower, upper, iw)
        Acons = sparse.csc_matrix(np.vstack(rows))
        return P, q, Acons, np.asarray(lower), np.asarray(upper), torso_error, pelvis_error, M, h, B, Jc, contact_bias, external

    def _fallback(self, message: str, elapsed: float) -> QPResult:
        pd = self.pd_fallback.compute()
        result = QPResult(
            control=np.clip(pd.control, self.model.actuator_limits[:, 0], self.model.actuator_limits[:, 1]),
            qdd=np.zeros(self.model.nv), contact_wrench=np.zeros(self.nw),
            status="fallback_pd", success=False, solve_time_s=elapsed, message=message,
            diagnostics={"active_contacts": list(self.contact_names), "swing_foot": self.swing_foot},
        )
        self.last_result = result
        return result

    @staticmethod
    def _constraint_budget_ratio(A, x, lower, upper, eps_abs, eps_rel):
        """Check every original row, rather than a globally scaled residual.

        Each finite bound gets its own absolute + relative tolerance in that
        row's physical units. A large dynamics force cannot excuse a negative
        unloaded-foot normal force or a violated zero-bound friction row.
        """
        values = np.asarray(A @ x)
        if not np.all(np.isfinite(values)):
            return float('inf')
        ratios = []
        for bound, sign in ((lower, -1), (upper, 1)):
            finite = np.isfinite(bound)
            scale = np.maximum(np.abs(values[finite]), np.abs(bound[finite]))
            excess = np.maximum(sign * (values[finite] - bound[finite]), 0)
            ratios.append(excess / (eps_abs + eps_rel * scale))
        return float(np.max(np.concatenate(ratios)))

    def solve(self) -> QPResult:
        start = time.perf_counter()
        if osqp is None:
            return self._fallback("osqp is not installed", time.perf_counter() - start)
        try:
            self._sync_internal_model()
            P, q, A, l, u, torso_error, pelvis_error, M, h, B, Jc, contact_bias, external = self._build_problem()
            build_time = time.perf_counter() - start
            preparation_start = time.perf_counter()
            # Value updates require identical CSC patterns and contact order.
            # Rebuild safely when topology, sparsity, or settings change.
            P = sparse.triu(sparse.csc_matrix((P + P.T) * 0.5), format="csc")
            P.sort_indices()
            A.sort_indices()
            patterns = (self.contact_names, P.shape, P.indptr.tobytes(), P.indices.tobytes(),
                        A.shape, A.indptr.tobytes(), A.indices.tobytes())
            settings = self.cfg.get("solver", {})
            patterns += (repr(sorted(settings.items())),)
            reused = bool(settings.get("reuse_workspace", False)) and self._solver is not None and patterns == self._solver_patterns
            if reused:
                solver = self._solver
                solver.update(Px=P.data, Ax=A.data, q=q, l=l, u=u)
            else:
                solver = osqp.OSQP()
                solver.setup(
                    P=P, q=q, A=A, l=l, u=u,
                    verbose=False, eps_abs=float(settings.get("eps_abs", 1e-4)),
                    eps_rel=float(settings.get("eps_rel", 1e-4)),
                    max_iter=int(settings.get("max_iter", 4000)), polish=bool(settings.get("polish", True)),
                    adaptive_rho=bool(settings.get("adaptive_rho", True)),
                    scaled_termination=bool(settings.get("scaled_termination", True)),
                )
                self._solver = solver
                self._solver_patterns = patterns
            preparation_time = time.perf_counter() - preparation_start
            solve_start = time.perf_counter()
            sol = solver.solve()
            numerical_solve_time = time.perf_counter() - solve_start
            elapsed = time.perf_counter() - start
            info = sol.info
            status = str(info.status)
            ok = status.lower() in {"solved", "solved inaccurate"} and sol.x is not None and np.all(np.isfinite(sol.x))
            if not ok:
                self._solver = None
                return self._fallback(f"OSQP status: {status}", elapsed)
            x = np.asarray(sol.x)
            eps_abs = float(settings.get('eps_abs', 1e-4))
            eps_rel = float(settings.get('eps_rel', 1e-4))
            row_ratio = self._constraint_budget_ratio(A, x, l, u, eps_abs, eps_rel)
            refinement_count = 0
            if row_ratio > 1.0:
                # Same objective/constraints, one tighter numerical retry.
                # Never clip the wrench or relax the physical recovery test.
                refinement_count = 1
                solver.update_settings(eps_abs=min(eps_abs, 1e-6),
                                       eps_rel=min(eps_rel, 1e-6), scaled_termination=False)
                retry_start = time.perf_counter()
                sol = solver.solve()
                numerical_solve_time += time.perf_counter() - retry_start
                info = sol.info
                status = str(info.status)
                ok = status.lower() in {'solved', 'solved inaccurate'} and sol.x is not None and np.all(np.isfinite(sol.x))
                x = np.asarray(sol.x) if ok else np.zeros(self.nx)
                row_ratio = self._constraint_budget_ratio(A, x, l, u, eps_abs, eps_rel) if ok else float('inf')
                if not ok or row_ratio > 1.0:
                    self._solver = None
                    result = self._fallback(
                        f'unscaled constraint validation failed after refinement: {status}, budget ratio={row_ratio:.6g}',
                        time.perf_counter() - start,
                    )
                    result.diagnostics.update(constraint_refinement_count=1, constraint_budget_ratio=row_ratio)
                    return result
            elapsed = time.perf_counter() - start
            iw = self.model.nv + self.model.nu
            wrench = x[iw:iw + self.nw]
            contact_slack_norm = float(np.linalg.norm(x[iw + self.nw:iw + self.nw + self.nslack]))
            margins = []
            for foot in range(len(self.contact_names)):
                off = 6 * foot
                fz = wrench[off + 2]
                margins.extend([self.mu * fz - abs(wrench[off]) - abs(wrench[off + 1]), fz])
            tau = np.clip(x[self.model.nv:self.model.nv + self.model.nu], self.model.actuator_limits[:, 0], self.model.actuator_limits[:, 1])
            slack = x[iw + self.nw:iw + self.nw + self.nslack]
            dynamics_residual = M @ x[:self.model.nv] + h - B @ tau - Jc.T @ wrench - external
            contact_residual = Jc @ x[:self.model.nv] + contact_bias + slack[:Jc.shape[0]]
            result = QPResult(
                control=tau,
                qdd=x[:self.model.nv], contact_wrench=wrench, status=status, success=True,
                solve_time_s=elapsed, primal_residual=float(getattr(info, "prim_res", np.nan)),
                dual_residual=float(getattr(info, "dual_res", np.nan)), objective=float(getattr(info, "obj_val", np.nan)),
                friction_margin=float(np.min(margins)),
                contact_slack_norm=contact_slack_norm,
                dynamics_residual_norm=float(np.linalg.norm(dynamics_residual)),
                contact_acceleration_residual_norm=float(np.linalg.norm(contact_residual)),
                diagnostics={
                    "constraint_refinement_count": refinement_count,
                    "constraint_budget_ratio": row_ratio,
                    "workspace_reused": reused,
                    "qp_build_time_s": build_time,
                    "qp_prepare_time_s": preparation_time,
                    "qp_numerical_solve_time_s": numerical_solve_time,
                    "friction_approximation": "world_xy_l1_inner",
                    "torso_se3_error": torso_error,
                    "pelvis_se3_error": pelvis_error,
                    "external_force_oracle": bool(self.cfg.get("use_external_force_oracle", False)),
                    "internal_model_is_plant": self.internal_model is self.model,
                    "active_contacts": list(self.contact_names),
                    "swing_foot": self.swing_foot,
                    "internal_mass_relative_error": float(
                        np.linalg.norm(self.model.mass_matrix() - self.internal_model.mass_matrix())
                        / max(np.linalg.norm(self.model.mass_matrix()), 1e-12)
                    ),
                },
            )
            result.solve_time_s = time.perf_counter() - start
            self.last_result = result
            return result
        except Exception as exc:  # keep the simulation observable and recoverable
            self._solver = None
            return self._fallback(f"QP exception: {type(exc).__name__}: {exc}", time.perf_counter() - start)
