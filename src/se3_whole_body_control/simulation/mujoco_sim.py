"""Deterministic MuJoCo experiment runner shared by all entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from se3_whole_body_control.disturbance.push import Push, active_push, push_force
from se3_whole_body_control.evaluation.metrics import TrialLog
from se3_whole_body_control.evaluation.recovery import RecoveryConfig, RecoveryResult, classify_recovery
from se3_whole_body_control.geometry.se3 import inverse_se3, log_se3
from se3_whole_body_control.control.joint_pd import JointPDController


@dataclass
class TrialRun:
    log: TrialLog
    recovery: RecoveryResult | None
    qpos_history: list[np.ndarray]
    metadata: dict


class SimulationRunner:
    def __init__(self, model, controller, duration_s: float = 6.0, control_timestep_s: float = 0.004, warmup_duration_s: float = 0.4, warmup_reanchor: bool = True):
        self.model = model
        self.controller = controller
        self.duration_s = float(duration_s)
        self.control_timestep_s = float(control_timestep_s)
        self.warmup_duration_s = float(warmup_duration_s)
        self.warmup_reanchor = bool(warmup_reanchor)
        ratio = self.control_timestep_s / float(model.model.opt.timestep)
        self.substeps = max(1, int(round(ratio)))
        if abs(ratio - self.substeps) > 1e-6:
            raise ValueError("control timestep must be an integer multiple of simulation timestep")

    def run(
        self,
        push: Push | None = None,
        recovery_config: RecoveryConfig | None = None,
        classify: bool = False,
        frame_callback: Callable[[float, object], None] | None = None,
        frame_period_s: float = 1 / 30,
        seed: int = 0,
        initial_qpos: np.ndarray | None = None,
        initial_qvel: np.ndarray | None = None,
        desired_torso: np.ndarray | None = None,
        desired_pelvis: np.ndarray | None = None,
        com_reference: np.ndarray | None = None,
    ) -> TrialRun:
        np.random.seed(seed)
        self.model.reset(qpos=initial_qpos, qvel=initial_qvel)
        self._warmup_and_reanchor()
        log = TrialLog.empty()
        qpos_history: list[np.ndarray] = []
        desired_torso = self.model.body_pose("torso") if desired_torso is None else np.asarray(desired_torso, dtype=float).copy()
        desired_pelvis = self.model.body_pose("pelvis") if desired_pelvis is None else np.asarray(desired_pelvis, dtype=float).copy()
        com0 = self.model.center_of_mass().copy() if com_reference is None else np.asarray(com_reference, dtype=float).copy()
        foot_xy_reference = np.full((2, 2), np.nan, dtype=float)
        next_frame = 0.0
        while self.model.data.time < self.duration_s - 1e-10:
            t = float(self.model.data.time)
            force = push_force(push, t)
            if np.any(force):
                self.model.set_external_force(push.application_body, force, push.application_point_local)
            else:
                self.model.clear_external_force()
            # Forward dynamics refreshes kinematics after xfrc_applied changes.
            try:
                import mujoco
                mujoco.mj_forward(self.model.model, self.model.data)
            except ImportError:
                pass
            if hasattr(self.controller, "solve"):
                result = self.controller.solve()
                control = result.control
                qp_status = result.status
                qp_time = result.solve_time_s
                wrench = result.contact_wrench
                qp_ok = bool(result.success)
                friction_margin = result.friction_margin if np.isfinite(result.friction_margin) else 0.0
                dynamics_residual_norm = result.dynamics_residual_norm
                contact_acceleration_residual_norm = result.contact_acceleration_residual_norm
            else:
                result = self.controller.compute()
                control = result.control
                qp_status = "pd"
                qp_time = 0.0
                wrench = np.zeros(12)
                qp_ok = True
                friction_margin = np.nan
                dynamics_residual_norm = np.nan
                contact_acceleration_residual_norm = np.nan
            T_torso = self.model.body_pose("torso")
            T_pelvis = self.model.body_pose("pelvis")
            # Match the world/spatial convention used by pose_task_acceleration.
            torso_error = log_se3(T_torso @ inverse_se3(desired_torso))
            pelvis_error = log_se3(T_pelvis @ inverse_se3(desired_pelvis))
            torso_velocity = self.model.body_velocity("torso")
            actual_contact = self.model.actual_contact_data()
            left, right = (bool(actual_contact.contact_flags[0]), bool(actual_contact.contact_flags[1]))
            actual_friction_margin = float(1.0 - np.max(actual_contact.friction_utilization)) if np.any(actual_contact.contact_flags) else float("nan")
            foot_xy = np.vstack([self.model.body_pose("left_foot")[:2, 3], self.model.body_pose("right_foot")[:2, 3]])
            foot_displacement = np.zeros(2, dtype=float)
            for foot_id in range(2):
                if actual_contact.contact_flags[foot_id] and not np.all(np.isfinite(foot_xy_reference[foot_id])):
                    foot_xy_reference[foot_id] = foot_xy[foot_id]
                if actual_contact.contact_flags[foot_id] and np.all(np.isfinite(foot_xy_reference[foot_id])):
                    foot_displacement[foot_id] = float(np.linalg.norm(foot_xy[foot_id] - foot_xy_reference[foot_id]))
            com = self.model.center_of_mass()
            log.append(
                time_s=t,
                torso_error=torso_error.tolist(),
                pelvis_error=pelvis_error.tolist(),
                com_world=com.tolist(),
                torso_position=T_torso[:3, 3].tolist(),
                torso_rotation_error_rad=float(np.linalg.norm(torso_error[3:])),
                contact_left=left,
                contact_right=right,
                predicted_contact_wrench=np.asarray(wrench).tolist(),
                actual_contact_wrench=actual_contact.wrench_world.tolist(),
                actual_friction_utilization=actual_contact.friction_utilization.tolist(),
                foot_tangent_velocity=actual_contact.tangent_velocity_m_s.tolist(),
                foot_xy_displacement=foot_displacement.tolist(),
                foot_xy_world=foot_xy.reshape(-1).tolist(),
                control=np.asarray(control).tolist(),
                qp_status=qp_status,
                qp_solve_time_s=float(qp_time),
                push_force=force.tolist(),
                joint_velocity_norm=float(np.linalg.norm(self.model.joint_velocities())),
                torso_angular_velocity_norm=float(np.linalg.norm(torso_velocity[3:])),
                torso_height_m=float(T_torso[2, 3]),
                torque_abs_max_Nm=float(np.max(np.abs(control))) if len(control) else 0.0,
                qp_success=qp_ok,
                predicted_friction_margin=float(friction_margin),
                actual_friction_margin=actual_friction_margin,
                qp_message=getattr(result, "message", "") if hasattr(result, "message") else "",
                qp_slack_norm=float(getattr(result, "contact_slack_norm", 0.0)),
                dynamics_residual_norm=float(dynamics_residual_norm),
                contact_acceleration_residual_norm=float(contact_acceleration_residual_norm),
            )
            qpos_history.append(self.model.data.qpos.copy())
            if frame_callback is not None and t + 1e-10 >= next_frame:
                frame_callback(t, self.model)
                next_frame += frame_period_s
            self.model.step(control)
            # A final bad state is retained in the log on the next iteration;
            # stop immediately on non-finite physics to keep the cause visible.
            if not np.all(np.isfinite(self.model.data.qpos)) or not np.all(np.isfinite(self.model.data.qvel)):
                break
            for _ in range(self.substeps - 1):
                self.model.step(control)

        recovery = None
        if classify:
            arrays = log.arrays()
            com_disp = np.linalg.norm(arrays["com_world"] - com0, axis=1)
            recovery = classify_recovery(
                arrays["time_s"], arrays["torso_rotation_error_rad"], arrays["torso_angular_velocity_norm"], com_disp,
                arrays["contact_left"], arrays["contact_right"], arrays["torso_height_m"], arrays["torque_abs_max_Nm"],
                arrays["qp_success"], arrays["actual_friction_margin"],
                actual_friction_utilization=arrays["actual_friction_utilization"],
                foot_tangent_velocity=arrays["foot_tangent_velocity"],
                foot_xy_displacement=arrays["foot_xy_displacement"],
                config=recovery_config,
                push_end_s=(push.start_time_s + push.duration_s) if push is not None else 0.0,
            )
        return TrialRun(log=log, recovery=recovery, qpos_history=qpos_history, metadata={"seed": seed, "push": repr(push)})

    def _warmup_and_reanchor(self) -> None:
        """Settle the common initial state before measuring a trial."""
        if self.warmup_duration_s <= 0:
            return
        if hasattr(self.controller, "pd_fallback"):
            warmup_controller = self.controller.pd_fallback
        elif isinstance(self.controller, JointPDController):
            warmup_controller = self.controller
        else:
            warmup_controller = JointPDController(self.model)
        self.model.clear_external_force()
        while self.model.data.time < self.warmup_duration_s - 1e-10:
            self.model.step(warmup_controller.compute().control)
        if not self.warmup_reanchor:
            self.model.data.time = 0.0
            self.model.clear_external_force()
            try:
                import mujoco
                mujoco.mj_forward(self.model.model, self.model.data)
            except ImportError:
                pass
            return
        # Re-anchor all regulation targets to the settled common state.
        if hasattr(self.controller, "q_des"):
            self.controller.q_des = self.model.joint_positions().copy()
        if hasattr(self.controller, "pd_fallback"):
            self.controller.pd_fallback.q_des = self.model.joint_positions().copy()
            self.controller.T_des_torso = self.model.body_pose("torso")
            self.controller.T_des_pelvis = self.model.body_pose("pelvis")
            self.controller.com_des = self.model.center_of_mass()
        self.model.data.time = 0.0
        self.model.clear_external_force()
        try:
            import mujoco
            mujoco.mj_forward(self.model.model, self.model.data)
        except ImportError:
            pass
