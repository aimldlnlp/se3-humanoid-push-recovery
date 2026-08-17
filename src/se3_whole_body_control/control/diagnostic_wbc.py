"""Diagnostic-only WBC extensions for post-touchdown mechanism studies.

The production :class:`WholeBodyQPController` is intentionally left
unchanged.  This module subclasses it and adds only explicitly selected
diagnostic objectives or constraints to the same physical QP.  The
controller is intended for replay studies, not for the stepping controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

import numpy as np
from scipy import sparse

from .whole_body_qp import WholeBodyQPController


@dataclass(frozen=True)
class DiagnosticProfile:
    """Fixed diagnostic profile definition.

    ``pose_pruned`` removes only weighted torso/pelvis/posture objectives;
    it does not change any physical constraint.  ``centroidal_momentum``
    adds a one-step predictive momentum objective.  ``joint_limit_guarded``
    adds a conservative next-step position guard for actuated hinge joints.
    """

    name: str
    pose_pruned: bool = False
    centroidal_momentum: bool = False
    joint_limit_guarded: bool = False
    momentum_decay_time_s: float = 0.25
    momentum_objective_weight: float = 100.0
    joint_guard_rad: float = 0.02


PROFILE_DEFINITIONS: dict[str, DiagnosticProfile] = {
    "baseline_landed_support": DiagnosticProfile("baseline_landed_support"),
    "pose_pruned": DiagnosticProfile("pose_pruned", pose_pruned=True),
    "centroidal_momentum": DiagnosticProfile("centroidal_momentum", centroidal_momentum=True),
    "joint_limit_guarded": DiagnosticProfile("joint_limit_guarded", joint_limit_guarded=True),
    "combined_minimal": DiagnosticProfile(
        "combined_minimal",
        pose_pruned=True,
        centroidal_momentum=True,
        joint_limit_guarded=True,
    ),
}


def diagnostic_profile(name: str) -> DiagnosticProfile:
    """Return a fixed profile or raise for an unapproved experiment profile."""

    try:
        return PROFILE_DEFINITIONS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown diagnostic profile {name!r}; choose from {sorted(PROFILE_DEFINITIONS)}") from exc


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


class DiagnosticWholeBodyQPController(WholeBodyQPController):
    """Production-compatible WBC with explicitly isolated diagnostic changes."""

    def __init__(
        self,
        model,
        controller_config: dict,
        recovery_config: dict | None = None,
        *,
        profile: str = "baseline_landed_support",
        control_timestep_s: float = 0.004,
    ):
        self.profile = diagnostic_profile(profile)
        diagnostic_config = copy.deepcopy(controller_config)
        if self.profile.pose_pruned:
            diagnostic_config["qp_torso_weight"] = 0.0
            diagnostic_config["qp_pelvis_weight"] = 0.0
            diagnostic_config["qp_posture_weight"] = 0.0
        super().__init__(model, diagnostic_config, recovery_config)
        self.control_timestep_s = float(control_timestep_s)
        if self.control_timestep_s <= 0.0:
            raise ValueError("control_timestep_s must be positive")
        self.diagnostic_history: list[dict] = []
        self._pending_diagnostic: dict = {}
        self._step_index = 0

    def reset_trial(self) -> None:
        """Clear per-replay diagnostic state without changing controller targets."""

        self.diagnostic_history.clear()
        self._pending_diagnostic = {}
        self._step_index = 0

    def _momentum_task(self, mass_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Build a normalized one-step centroidal momentum objective.

        The approximation keeps ``M(q)`` and the current CoM shift fixed over
        one control interval.  This makes the target affine in the QP's
        generalized acceleration and keeps the physical dynamics/contact
        constraints untouched.  The resulting target is deliberately
        diagnostic: logged measured momentum is compared against it later.
        """

        model = self.internal_model
        nv = model.nv
        dt = self.control_timestep_s
        tau = float(self.profile.momentum_decay_time_s)
        if tau <= 0.0:
            raise ValueError("momentum decay time must be positive")
        qvel = np.asarray(model.data.qvel, dtype=float)
        generalized_momentum = np.asarray(mass_matrix, dtype=float) @ qvel
        root_body_id = int(model.body_ids["floating_base"])
        shift = model.center_of_mass() - np.asarray(model.data.xpos[root_body_id], dtype=float)

        # ``[p_linear, H_centroidal]`` uses the same generalized-momentum
        # convention as replay_touchdown_recoverability.py.
        momentum_map = np.zeros((6, nv), dtype=float)
        momentum_map[:3] = mass_matrix[:3]
        momentum_map[3:6] = mass_matrix[3:6] - _skew(shift) @ mass_matrix[:3]
        current = momentum_map @ qvel
        target_delta = -dt / tau * current

        mass = float(np.sum(model.model.body_mass))
        length_scale = max(float(np.linalg.norm(shift)), 0.5)
        scale = np.r_[np.full(3, max(mass, 1e-9)), np.full(3, max(mass * length_scale, 1e-9))]
        A_task = np.zeros((6, self.nx), dtype=float)
        A_task[:, :nv] = dt * momentum_map / scale[:, None]
        b_task = target_delta / scale
        metadata = {
            "momentum_current": current.copy(),
            "momentum_target_delta": target_delta.copy(),
            "momentum_scale": scale.copy(),
            "momentum_map": momentum_map.copy(),
            "momentum_decay_time_s": tau,
            "momentum_objective_weight": float(self.profile.momentum_objective_weight),
            "momentum_prediction_valid": True,
        }
        return A_task, b_task, metadata

    def _joint_guard_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return one-step actuated hinge position guard rows."""

        model = self.internal_model
        dt = self.control_timestep_s
        guard = float(self.profile.joint_guard_rad)
        lo, hi = model.joint_position_limits()
        limited = np.asarray(
            [bool(model.model.jnt_limited[j]) for j in model.joint_ids.values()],
            dtype=bool,
        )
        if np.any(limited & ((hi - lo) <= 2.0 * guard)):
            raise ValueError("joint guard is larger than a configured joint range")

        q = np.asarray(model.joint_positions(), dtype=float)
        qdot = np.asarray(model.joint_velocities(), dtype=float)
        base = q + dt * qdot
        coefficient = 0.5 * dt * dt
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        for index, (qvel_index, is_limited) in enumerate(zip(model.joint_qvel_indices, limited)):
            if not is_limited:
                continue
            row = np.zeros(self.nx, dtype=float)
            row[int(qvel_index)] = coefficient
            rows.append(row)
            lower.append(float(lo[index] + guard - base[index]))
            upper.append(float(hi[index] - guard - base[index]))
        if not rows:
            return np.zeros((0, self.nx)), np.zeros(0), np.zeros(0), {
                "joint_guard_enabled": True,
                "joint_guard_rad": guard,
                "joint_guard_lower": np.zeros(0),
                "joint_guard_upper": np.zeros(0),
                "joint_guard_limited": limited.copy(),
            }
        return np.vstack(rows), np.asarray(lower), np.asarray(upper), {
            "joint_guard_enabled": True,
            "joint_guard_rad": guard,
            "joint_guard_lower": np.asarray(lower),
            "joint_guard_upper": np.asarray(upper),
            "joint_guard_limited": limited.copy(),
        }

    def _build_problem(self):
        result = super()._build_problem()
        P, q, Acons, lower, upper, torso_error, pelvis_error, M, h, B, Jc, contact_bias, external = result
        pending: dict = {
            "profile": self.profile.name,
            "profile_definition": asdict(self.profile),
            "active_contacts": list(self.contact_names),
            "momentum_task_enabled": False,
            "momentum_task_valid": False,
            "joint_guard_enabled": False,
            "joint_guard_valid": False,
            "joint_guard_prediction": {},
        }

        if self.profile.centroidal_momentum:
            A_momentum, b_momentum, momentum_metadata = self._momentum_task(M)
            self._add_objective(
                P,
                q,
                A_momentum,
                b_momentum,
                float(self.profile.momentum_objective_weight),
            )
            pending.update({
                "momentum_task_enabled": True,
                "momentum_task_valid": True,
                **momentum_metadata,
            })

        if self.profile.joint_limit_guarded:
            guard_rows, guard_lower, guard_upper, guard_metadata = self._joint_guard_constraints()
            if len(guard_rows):
                Acons = sparse.vstack(
                    [Acons, sparse.csc_matrix(guard_rows)],
                    format="csc",
                )
                lower = np.r_[lower, guard_lower]
                upper = np.r_[upper, guard_upper]
            pending.update({
                "joint_guard_enabled": True,
                "joint_guard_valid": True,
                **guard_metadata,
            })

        self._pending_diagnostic = pending
        return P, q, Acons, lower, upper, torso_error, pelvis_error, M, h, B, Jc, contact_bias, external

    @staticmethod
    def _padded_wrench(wrench: np.ndarray) -> np.ndarray:
        padded = np.full(12, np.nan, dtype=float)
        values = np.asarray(wrench, dtype=float).reshape(-1)
        padded[: min(len(values), 12)] = values[:12]
        return padded

    def _predicted_friction_utilization(self, wrench: np.ndarray) -> np.ndarray:
        values = np.asarray(wrench, dtype=float).reshape(-1)
        utilization = np.full(2, np.nan, dtype=float)
        for contact_index in range(len(self.contact_names)):
            source = values[6 * contact_index : 6 * contact_index + 6]
            if len(source) != 6 or not np.all(np.isfinite(source)) or source[2] <= 1e-12:
                utilization[0 if self.contact_names[contact_index] == "left_foot" else 1] = 0.0
                continue
            foot_index = 0 if self.contact_names[contact_index] == "left_foot" else 1
            utilization[foot_index] = max(abs(source[0]), abs(source[1])) / max(self.mu * source[2], 1e-12)
        return utilization

    def solve(self):
        result = super().solve()
        diagnostic = dict(self._pending_diagnostic or {})
        model = self.internal_model
        qvel = np.asarray(model.data.qvel, dtype=float).copy()
        diagnostic["step_index"] = int(self._step_index)
        diagnostic["commanded_torque"] = np.asarray(result.control, dtype=float).copy()
        torque_limits = np.maximum(np.max(np.abs(model.actuator_limits), axis=1), 1e-12)
        diagnostic["commanded_torque_utilization"] = float(
            np.max(np.abs(result.control) / torque_limits) if len(result.control) else 0.0
        )
        diagnostic["predicted_contact_wrench"] = self._padded_wrench(result.contact_wrench)
        diagnostic["predicted_friction_utilization"] = self._predicted_friction_utilization(result.contact_wrench)
        diagnostic["qp_status"] = str(result.status)
        diagnostic["qp_success"] = bool(result.success)
        diagnostic["qp_slack_norm"] = float(result.contact_slack_norm)
        diagnostic["dynamics_residual_norm"] = float(result.dynamics_residual_norm)
        diagnostic["contact_acceleration_residual_norm"] = float(result.contact_acceleration_residual_norm)
        diagnostic["predicted_friction_margin"] = float(result.friction_margin)
        diagnostic["qvel_norm"] = float(np.linalg.norm(qvel))

        if bool(diagnostic.get("momentum_task_valid", False)):
            current = np.asarray(diagnostic["momentum_current"], dtype=float)
            target_delta = np.asarray(diagnostic["momentum_target_delta"], dtype=float)
            momentum_map = np.asarray(diagnostic["momentum_map"], dtype=float)
            scale = np.asarray(diagnostic["momentum_scale"], dtype=float)
            predicted = current + self.control_timestep_s * momentum_map @ np.asarray(result.qdd, dtype=float)
            target = current + target_delta
            diagnostic["momentum_prediction"] = predicted
            diagnostic["momentum_target"] = target
            diagnostic["momentum_task_residual"] = (predicted - target) / scale
            diagnostic["momentum_task_residual_norm"] = float(np.linalg.norm(diagnostic["momentum_task_residual"]))
        else:
            diagnostic["momentum_current"] = np.full(6, np.nan)
            diagnostic["momentum_target"] = np.full(6, np.nan)
            diagnostic["momentum_prediction"] = np.full(6, np.nan)
            diagnostic["momentum_task_residual"] = np.full(6, np.nan)
            diagnostic["momentum_task_residual_norm"] = float("nan")

        joint_q = np.asarray(model.joint_positions(), dtype=float)
        joint_qdot = np.asarray(model.joint_velocities(), dtype=float)
        lo, hi = model.joint_position_limits()
        predicted_joint_q = joint_q + self.control_timestep_s * joint_qdot
        qdd_joint = np.asarray(result.qdd, dtype=float)[model.joint_qvel_indices]
        predicted_joint_q = predicted_joint_q + 0.5 * self.control_timestep_s**2 * qdd_joint
        joint_margin = np.minimum(predicted_joint_q - lo, hi - predicted_joint_q)
        diagnostic["predicted_joint_positions"] = predicted_joint_q
        diagnostic["predicted_joint_limit_margins"] = joint_margin
        diagnostic["predicted_min_joint_limit_margin_rad"] = float(np.min(joint_margin)) if len(joint_margin) else float("nan")
        diagnostic["predicted_min_joint_guard_margin_rad"] = float(
            np.min(joint_margin - float(self.profile.joint_guard_rad)) if len(joint_margin) else float("nan")
        )
        diagnostic["actual_joint_positions"] = joint_q
        diagnostic["actual_joint_limit_margins"] = np.minimum(joint_q - lo, hi - joint_q)

        result.diagnostics = dict(result.diagnostics or {})
        result.diagnostics.update({
            "diagnostic_profile": self.profile.name,
            "diagnostic_step_index": int(self._step_index),
            "diagnostic_momentum_task_residual_norm": diagnostic["momentum_task_residual_norm"],
            "diagnostic_predicted_min_joint_limit_margin_rad": diagnostic["predicted_min_joint_limit_margin_rad"],
            "diagnostic_predicted_min_joint_guard_margin_rad": diagnostic["predicted_min_joint_guard_margin_rad"],
        })
        self.diagnostic_history.append(diagnostic)
        self._step_index += 1
        return result

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return row-aligned controller diagnostics for artifact serialization."""

        history = self.diagnostic_history
        if not history:
            return {
                "diagnostic_step_index": np.zeros(0, dtype=int),
                "diagnostic_profile": np.asarray([], dtype="U64"),
            }

        def stack(name: str, shape: tuple[int, ...], fill: float = np.nan) -> np.ndarray:
            values = []
            for item in history:
                value = item.get(name)
                if value is None:
                    values.append(np.full(shape, fill, dtype=float))
                else:
                    array = np.asarray(value, dtype=float)
                    if array.shape != shape:
                        raise ValueError(f"diagnostic field {name!r} has shape {array.shape}, expected {shape}")
                    values.append(array)
            return np.stack(values)

        def scalar(name: str, fill: float = np.nan) -> np.ndarray:
            return np.asarray([item.get(name, fill) for item in history], dtype=float)

        return {
            "diagnostic_step_index": np.asarray([item.get("step_index", i) for i, item in enumerate(history)], dtype=int),
            "diagnostic_profile": np.asarray([item.get("profile", self.profile.name) for item in history], dtype="U64"),
            "active_contacts_json": np.asarray([str(item.get("active_contacts", [])) for item in history], dtype="U128"),
            "commanded_torque": stack("commanded_torque", (self.model.nu,), fill=0.0),
            "commanded_torque_utilization": scalar("commanded_torque_utilization", 0.0),
            "predicted_contact_wrench": stack("predicted_contact_wrench", (12,)),
            "predicted_friction_utilization": stack("predicted_friction_utilization", (2,)),
            "qp_success": np.asarray([bool(item.get("qp_success", False)) for item in history], dtype=bool),
            "qp_status": np.asarray([str(item.get("qp_status", "")) for item in history], dtype="U64"),
            "qp_slack_norm": scalar("qp_slack_norm"),
            "dynamics_residual_norm": scalar("dynamics_residual_norm"),
            "contact_acceleration_residual_norm": scalar("contact_acceleration_residual_norm"),
            "predicted_friction_margin": scalar("predicted_friction_margin"),
            "momentum_current": stack("momentum_current", (6,)),
            "momentum_target": stack("momentum_target", (6,)),
            "momentum_prediction": stack("momentum_prediction", (6,)),
            "momentum_task_residual": stack("momentum_task_residual", (6,)),
            "momentum_task_residual_norm": scalar("momentum_task_residual_norm"),
            "predicted_joint_positions": stack("predicted_joint_positions", (self.model.nu,)),
            "predicted_joint_limit_margins": stack("predicted_joint_limit_margins", (self.model.nu,)),
            "predicted_min_joint_limit_margin_rad": scalar("predicted_min_joint_limit_margin_rad"),
            "predicted_min_joint_guard_margin_rad": scalar("predicted_min_joint_guard_margin_rad"),
            "actual_joint_positions": stack("actual_joint_positions", (self.model.nu,)),
            "actual_joint_limit_margins": stack("actual_joint_limit_margins", (self.model.nu,)),
        }

    def summary(self) -> dict:
        return {
            "controller_type": self.__class__.__name__,
            "diagnostic_profile": self.profile.name,
            "profile_definition": asdict(self.profile),
            "diagnostic_history_length": len(self.diagnostic_history),
            "active_contacts": list(self.contact_names),
        }
