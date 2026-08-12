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
    def __init__(self, model, controller, duration_s: float = 6.0, control_timestep_s: float = 0.004, warmup_duration_s: float = 0.4):
        self.model = model
        self.controller = controller
        self.duration_s = float(duration_s)
        self.control_timestep_s = float(control_timestep_s)
        self.warmup_duration_s = float(warmup_duration_s)
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
    ) -> TrialRun:
        np.random.seed(seed)
        self.model.reset()
        self._warmup_and_reanchor()
        log = TrialLog.empty()
        qpos_history: list[np.ndarray] = []
        desired_torso = self.model.body_pose("torso")
        desired_pelvis = self.model.body_pose("pelvis")
        com0 = self.model.center_of_mass().copy()
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
            else:
                result = self.controller.compute()
                control = result.control
                qp_status = "pd"
                qp_time = 0.0
                wrench = np.zeros(12)
                qp_ok = True
                friction_margin = np.nan
            T_torso = self.model.body_pose("torso")
            T_pelvis = self.model.body_pose("pelvis")
            torso_error = log_se3(inverse_se3(desired_torso) @ T_torso)
            pelvis_error = log_se3(inverse_se3(desired_pelvis) @ T_pelvis)
            torso_velocity = self.model.body_velocity("torso")
            left, right = self.model.contact_flags()
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
                contact_wrench=np.asarray(wrench).tolist(),
                control=np.asarray(control).tolist(),
                qp_status=qp_status,
                qp_solve_time_s=float(qp_time),
                push_force=force.tolist(),
                joint_velocity_norm=float(np.linalg.norm(self.model.joint_velocities())),
                torso_angular_velocity_norm=float(np.linalg.norm(torso_velocity[3:])),
                torso_height_m=float(T_torso[2, 3]),
                torque_abs_max_Nm=float(np.max(np.abs(control))) if len(control) else 0.0,
                qp_success=qp_ok,
                friction_margin=float(friction_margin),
                qp_message=getattr(result, "message", "") if hasattr(result, "message") else "",
                qp_slack_norm=float(getattr(result, "contact_slack_norm", 0.0)),
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
                arrays["qp_success"], arrays["friction_margin"], recovery_config,
                recovery_start_s=(push.start_time_s + push.duration_s) if push is not None else 0.0,
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
