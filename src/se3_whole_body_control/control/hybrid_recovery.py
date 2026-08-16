"""One-step contact-mode recovery built on the production whole-body QP.

The controller starts in the validated fixed-foot double-support mode.  It
does not receive the configured push as an oracle.  When measured state leaves
the double-support margin with sufficient horizontal motion, it selects the
trailing foot, switches the QP to the remaining support foot, tracks a smooth
clearance/placement target, and returns to double support after touchdown.

This is intentionally a one-step recovery primitive, not a walking planner.
The fixed-foot mode remains the default and is unchanged for existing runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .support import convex_hull_2d, normalize_xy, signed_support_margin
from .tasks import com_jacobian
from .whole_body_qp import QPResult, WholeBodyQPController


@dataclass
class StepEvent:
    swing_foot: str
    support_foot: str
    direction_xy: np.ndarray
    start_time_s: float
    target_pose: np.ndarray


class HybridRecoveryController(WholeBodyQPController):
    """Fixed-foot WBC with a measured-state-triggered one-step mode."""

    allows_single_support = True
    requires_final_double_support = True

    def __init__(self, model, controller_config: dict, recovery_config: dict | None = None, hybrid_config: dict | None = None, internal_model=None):
        super().__init__(model, controller_config, recovery_config=recovery_config, internal_model=internal_model)
        self.hybrid_config = dict(hybrid_config or {})
        self.reset_trial()

    def _support_geometry(self) -> tuple[float, np.ndarray]:
        vertices = np.asarray(self.model.foot_support_vertices_world(), dtype=float)
        finite = vertices[np.all(np.isfinite(vertices), axis=2)]
        hull = convex_hull_2d(finite)
        margin = signed_support_margin(self.model.center_of_mass()[:2], hull)
        return margin, hull

    def _motion_direction(self) -> np.ndarray:
        velocity = com_jacobian(self.model) @ self.model.data.qvel
        displacement = self.model.center_of_mass() - self.com_des
        direction = normalize_xy(velocity[:2] + float(self.hybrid_config.get("displacement_direction_gain", 3.0)) * displacement[:2])
        if np.linalg.norm(direction) < 1e-12:
            direction = normalize_xy(self.model.body_velocity("torso")[:2])
        if np.linalg.norm(direction) < 1e-12:
            direction = np.array([1.0, 0.0], dtype=float)
        return direction

    def _maybe_trigger_step(self) -> None:
        if self.mode != "double_support" or self.step_count >= int(self.hybrid_config.get("max_steps", 1)):
            return
        time_s = float(self.model.data.time)
        if self._nominal_com_des is None:
            self._nominal_com_des = self.com_des.copy()
        if time_s < float(self.hybrid_config.get("minimum_step_time_s", 0.35)) or time_s < self._cooldown_until_s:
            return
        margin, _ = self._support_geometry()
        self._support_margin_m = margin
        if not np.isfinite(margin):
            return
        velocity = com_jacobian(self.model) @ self.model.data.qvel
        speed = float(np.linalg.norm(velocity[:2]))
        torso_velocity = float(np.linalg.norm(self.model.body_velocity("torso")[:2]))
        speed = max(speed, torso_velocity)
        margin_limit = float(self.hybrid_config.get("trigger_margin_m", 0.008))
        speed_limit = float(self.hybrid_config.get("trigger_speed_m_s", 0.10))
        orientation_limit = float(self.hybrid_config.get("trigger_torso_error_rad", 0.35))
        torso_error = float(np.linalg.norm(self._last_result.diagnostics.get("torso_se3_error", np.zeros(6))[3:])) if self._last_result else 0.0
        slack_limit = float(self.hybrid_config.get("trigger_contact_slack_norm", np.inf))
        slack_trigger = self._last_result is not None and self._last_result.contact_slack_norm > slack_limit
        nominal_com = self._nominal_com_des if self._nominal_com_des is not None else self.com_des
        com_displacement = float(np.linalg.norm(self.model.center_of_mass()[:2] - nominal_com[:2]))
        com_trigger = com_displacement >= float(self.hybrid_config.get("trigger_com_displacement_m", np.inf))
        if not com_trigger and (margin > margin_limit or (speed < speed_limit and torso_error < orientation_limit and not slack_trigger)):
            return
        self._step_trigger_reason = "com_displacement" if com_trigger else "support_margin_or_velocity"
        self._start_step(self._motion_direction(), time_s)

    def _start_step(self, direction_xy: np.ndarray, time_s: float) -> None:
        foot_names = ("left_foot", "right_foot")
        positions = np.vstack([self.model.body_pose(name)[:2, 3] for name in foot_names])
        trailing = int(np.argmin(positions @ direction_xy))
        swing_foot = foot_names[trailing]
        support_foot = foot_names[1 - trailing]
        start_pose = self.model.body_pose(swing_foot)
        target_pose = start_pose.copy()
        target_pose[:2, 3] += direction_xy * float(self.hybrid_config.get("step_length_m", 0.16))
        self.step_event = StepEvent(swing_foot, support_foot, direction_xy.copy(), time_s, target_pose)
        self._step_direction_xy = direction_xy.copy()
        self.step_phase = "transfer"
        self.mode = "transfer"
        self._transfer_start_s = time_s
        support_index = 0 if support_foot == "left_foot" else 1
        support_vertices = np.asarray(self.model.foot_support_vertices_world(), dtype=float)[support_index]
        support_center = np.nanmean(support_vertices[:, :2], axis=0)
        transfer_target = self.com_des.copy()
        transfer_target[:2] = np.array([self.model.center_of_mass()[0], support_center[1]])
        self._transfer_target_com = transfer_target
        self.com_des = transfer_target.copy()
        self.com_task_weight_override = float(self.hybrid_config.get("transfer_com_weight", 20.0))
        self.com_task_kp_override = float(self.hybrid_config.get("transfer_com_kp", 140.0))
        self.com_task_kd_override = float(self.hybrid_config.get("transfer_com_kd", 24.0))
        self.set_active_contacts(("left_foot", "right_foot"))
        self.set_swing_target(None)

    def _update_transfer(self) -> None:
        if self.step_event is None or self.step_phase != "transfer" or self._transfer_target_com is None:
            return
        now = float(self.model.data.time)
        elapsed = now - float(self._transfer_start_s if self._transfer_start_s is not None else now)
        support_index = 0 if self.step_event.support_foot == "left_foot" else 1
        vertices = np.asarray(self.model.foot_support_vertices_world(), dtype=float)[support_index]
        margin = signed_support_margin(self.model.center_of_mass()[:2], convex_hull_2d(vertices[:, :2]))
        com_velocity = com_jacobian(self.model) @ self.model.data.qvel
        transfer_duration = float(self.hybrid_config.get("transfer_duration_s", 0.24))
        max_transfer_duration = float(self.hybrid_config.get("max_transfer_duration_s", 0.48))
        transfer_margin = float(self.hybrid_config.get("transfer_margin_m", 0.004))
        lateral_speed = abs(float(com_velocity[1]))
        ready = elapsed >= transfer_duration and margin >= transfer_margin and lateral_speed <= float(self.hybrid_config.get("transfer_speed_m_s", 0.15))
        if not ready and elapsed < max_transfer_duration:
            return
        self.mode = "single_support"
        self.step_phase = "swing"
        self.step_event.start_time_s = now
        self._swing_start_pose = self.model.body_pose(self.step_event.swing_foot).copy()
        self.set_active_contacts((self.step_event.support_foot,))
        self.set_swing_target(self.step_event.swing_foot, self._swing_start_pose)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)

    def _update_swing_target(self) -> None:
        if self.step_event is None:
            return
        # The immutable start pose is reconstructed from the target and event
        # only once; storing it avoids target drift across control cycles.
        if not hasattr(self, "_swing_start_pose") or self._swing_start_pose is None:
            self._swing_start_pose = self.model.body_pose(self.step_event.swing_foot).copy()
        start = self._swing_start_pose
        target = self.step_event.target_pose
        duration = max(float(self.hybrid_config.get("swing_duration_s", 0.34)), 1e-3)
        phase_time = float(self.model.data.time) - self.step_event.start_time_s
        u = float(np.clip(phase_time / duration, 0.0, 1.0))
        s = self._smoothstep(u)
        pose = start.copy()
        pose[:3, 3] = (1.0 - s) * start[:3, 3] + s * target[:3, 3]
        pose[2, 3] += float(self.hybrid_config.get("step_clearance_m", 0.06)) * 4.0 * u * (1.0 - u)
        self.set_swing_target(self.step_event.swing_foot, pose)
        if u >= 1.0 and self.step_phase == "swing":
            self.step_phase = "landing"
            self.mode = "landing"
            self._landing_start_s = float(self.model.data.time)
            self.set_active_contacts((self.step_event.support_foot, self.step_event.swing_foot))
            # Once touchdown is expected, the foot returns to the fixed-contact
            # set.  The preceding swing solve already brought it to ``target``;
            # retaining a swing task while it is an active contact would make
            # the QP semantics contradictory.
            self.set_swing_target(None)

    def _update_landing(self) -> None:
        if self.step_event is None or self.step_phase != "landing":
            return
        left, right = self.model.contact_flags()
        landing_duration = float(self.hybrid_config.get("landing_duration_s", 0.18))
        if left and right and self._landing_start_s is not None and float(self.model.data.time) - self._landing_start_s >= landing_duration:
            self.mode = "double_support"
            self.step_phase = "recovered_step"
            self.step_count += 1
            self._cooldown_until_s = float(self.model.data.time) + float(self.hybrid_config.get("step_cooldown_s", 0.35))
            self._settle_until_s = float(self.model.data.time) + float(self.hybrid_config.get("post_landing_duration_s", 0.50))
            self.set_active_contacts(("left_foot", "right_foot"))
            self.set_swing_target(None)
            if self._nominal_com_des is not None:
                self.com_des = self._nominal_com_des.copy()

    def solve(self) -> QPResult:
        if self.mode == "double_support" and self.step_phase == "recovered_step" and self._settle_until_s is not None:
            if float(self.model.data.time) >= self._settle_until_s:
                self.com_task_weight_override = None
                self.com_task_kp_override = None
                self.com_task_kd_override = None
                self._settle_until_s = None
        if self.mode == "double_support":
            self._maybe_trigger_step()
        if self.mode == "transfer":
            self._update_transfer()
        if self.mode == "single_support":
            self._update_swing_target()
        elif self.mode == "landing":
            self._update_landing()
        result = super().solve()
        self._last_result = result
        result.diagnostics.update({
            "control_mode": self.mode,
            "step_phase": self.step_phase,
            "swing_foot": self.step_event.swing_foot if self.step_event else None,
            "support_margin_m": self._support_margin_m,
            "step_count": self.step_count,
        })
        return result

    def reset_trial(self) -> None:
        self.set_active_contacts(("left_foot", "right_foot"))
        self.set_swing_target(None)
        self.mode = "double_support"
        self.step_phase = "stance"
        self.step_event = None
        self.step_count = 0
        self._landing_start_s = None
        self._cooldown_until_s = 0.0
        self._support_margin_m = float("nan")
        self._step_direction_xy = np.zeros(2, dtype=float)
        self._last_result = None
        self._swing_start_pose = None
        self._nominal_com_des = None
        self._transfer_target_com = None
        self._transfer_start_s = None
        self._settle_until_s = None
        self._step_trigger_reason = None
        self.com_task_weight_override = None
        self.com_task_kp_override = None
        self.com_task_kd_override = None

    def summary(self) -> dict:
        return {
            "controller": "hybrid_se3_wbc",
            "step_count": int(self.step_count),
            "step_triggered": bool(self.step_event is not None),
            "swing_foot": self.step_event.swing_foot if self.step_event else None,
            "support_foot": self.step_event.support_foot if self.step_event else None,
            "step_direction_xy": self._step_direction_xy.tolist(),
            "final_mode": self.mode,
            "final_phase": self.step_phase,
            "trigger_reason": self._step_trigger_reason,
        }
