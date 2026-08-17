"""Adaptive contact-mode recovery built on the production whole-body QP.

The controller starts in the validated fixed-foot double-support mode.  It
does not receive the configured push as an oracle.  When measured state leaves
the double-support margin with sufficient horizontal motion, it selects the
trailing foot, switches the QP to the remaining support foot, tracks a smooth
clearance/placement target, and returns to double support after touchdown.

The fixed-foot mode remains the default.  The supervisor can complete a
bounded second step, but it is intentionally not a general walking planner.
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
    trigger_time_s: float
    start_time_s: float
    target_pose: np.ndarray
    step_index: int


class HybridRecoveryController(WholeBodyQPController):
    """Fixed-foot WBC with measured-state adaptive stepping."""

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

    def _support_margin_for(self, contact_names: tuple[str, ...] | list[str]) -> tuple[float, np.ndarray]:
        """Compute a support margin for the contacts active in the mode."""
        vertices = np.asarray(self.model.foot_support_vertices_world(), dtype=float)
        indices = [0 if name == "left_foot" else 1 for name in contact_names]
        selected = vertices[indices] if indices else np.empty((0, 4, 2))
        finite = selected[np.all(np.isfinite(selected), axis=2)]
        if len(finite) < 3:
            return float("nan"), np.empty((0, 2), dtype=float)
        hull = convex_hull_2d(finite)
        return signed_support_margin(self.model.center_of_mass()[:2], hull), hull

    def _current_support_margin(self) -> float:
        if self.mode in {"single_support", "landing", "failed_recovery"} and self.step_event is not None:
            margin, _ = self._support_margin_for((self.step_event.support_foot,))
        else:
            margin, _ = self._support_geometry()
        return float(margin)

    def _emit_event(self, label: str, reason: str = "", **payload) -> None:
        """Record a replayable event and expose it on the next QP result."""
        record = {"time_s": float(self.model.data.time), "label": str(label), "reason": str(reason)}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                record[key] = value.astype(float).reshape(-1).tolist()
            elif isinstance(value, (np.floating, np.integer)):
                record[key] = value.item()
            else:
                record[key] = value
        self.events.append(record)
        self._pending_event_label = str(label)
        self._pending_event_reason = str(reason)

    def _update_step_history(self, status: str) -> None:
        if self.step_history:
            self.step_history[-1]["status"] = str(status)
            self.step_history[-1]["status_time_s"] = float(self.model.data.time)

    def _planned_step_length(self) -> float:
        velocity = com_jacobian(self.model) @ self.model.data.qvel
        speed = float(np.linalg.norm(velocity[:2]))
        displacement = float(np.linalg.norm(self.model.center_of_mass()[:2] - self.com_des[:2]))
        severity = max(0.0, speed - float(self.hybrid_config.get("trigger_speed_m_s", 0.15)))
        severity += max(0.0, displacement - float(self.hybrid_config.get("trigger_com_displacement_m", 0.10)))
        nominal = float(self.hybrid_config.get("step_length_m", 0.30))
        gain = float(self.hybrid_config.get("step_length_per_severity", 0.35))
        minimum = float(self.hybrid_config.get("min_step_length_m", 0.22))
        maximum = float(self.hybrid_config.get("max_step_length_m", 0.40))
        if self.step_count > 0:
            maximum = min(maximum, float(self.hybrid_config.get("second_step_max_length_m", maximum)))
            minimum = min(minimum, maximum)
        return float(np.clip(
            nominal + gain * severity,
            minimum,
            maximum,
        ))

    def _make_step_target(self, swing_foot: str, support_foot: str, direction_xy: np.ndarray) -> np.ndarray:
        start_pose = self.model.body_pose(swing_foot)
        support_pose = self.model.body_pose(support_foot)
        target_pose = start_pose.copy()
        target_pose[:2, 3] += direction_xy * self._planned_step_length()
        if self.step_count > 0:
            # A bounded second step is a capture action, not another copy of
            # the nominal stance offset. Use the measured CoM and short-horizon
            # horizontal velocity, then apply the same reach clamp below.
            velocity = com_jacobian(self.model) @ self.model.data.qvel
            horizon = float(self.hybrid_config.get("second_step_capture_time_s", 0.05))
            target_pose[:2, 3] = self.model.center_of_mass()[:2] + velocity[:2] * horizon
        # Reference the support foot's height so the target is on the actual
        # MuJoCo ground plane; clearance is added only to the swing trajectory.
        target_pose[2, 3] = support_pose[2, 3]
        support_center = support_pose[:2, 3]
        relative = target_pose[:2, 3] - support_center
        max_reach = float(self.hybrid_config.get("max_step_reach_m", 0.46))
        if np.linalg.norm(relative) > max_reach:
            target_pose[:2, 3] = support_center + relative / max(np.linalg.norm(relative), 1e-12) * max_reach
        return target_pose

    def _double_support_com_target(self) -> np.ndarray:
        """Return a conservative CoM target centered between measured feet."""
        centers = np.vstack([
            self.model.body_pose("left_foot")[:2, 3],
            self.model.body_pose("right_foot")[:2, 3],
        ])
        target = self.com_des.copy()
        target[:2] = np.mean(centers, axis=0)
        if self._nominal_com_des is not None:
            target[2] = self._nominal_com_des[2]
        return target

    def _maybe_trigger_step(self) -> None:
        if self.mode != "double_support" or self.step_count >= int(self.hybrid_config.get("max_steps", 2)):
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
        predictive_trigger = (
            speed >= float(self.hybrid_config.get("predictive_trigger_speed_m_s", np.inf))
            and margin <= float(self.hybrid_config.get("predictive_trigger_margin_m", -np.inf))
        )
        if not com_trigger and not predictive_trigger and (margin > margin_limit or (speed < speed_limit and torso_error < orientation_limit and not slack_trigger)):
            return
        if com_trigger:
            self._step_trigger_reason = "com_displacement"
        elif predictive_trigger:
            self._step_trigger_reason = "predicted_support_exhaustion"
        elif margin <= margin_limit:
            self._step_trigger_reason = "support_margin"
        elif speed >= speed_limit:
            self._step_trigger_reason = "horizontal_motion"
        else:
            self._step_trigger_reason = "contact_slack"
        self._start_step(self._motion_direction(), time_s)

    def _start_step(self, direction_xy: np.ndarray, time_s: float) -> None:
        foot_names = ("left_foot", "right_foot")
        # Once a step has landed, the landed foot is the known support for the
        # next bounded step. Re-selecting from instantaneous geometry can pick
        # the already-loaded foot again while the body is falling.
        if self.step_count > 0 and self._active_support_foot and self._next_swing_foot:
            swing_foot = self._next_swing_foot
            support_foot = self._active_support_foot
        else:
            positions = np.vstack([self.model.body_pose(name)[:2, 3] for name in foot_names])
            trailing = int(np.argmin(positions @ direction_xy))
            swing_foot = foot_names[trailing]
            support_foot = foot_names[1 - trailing]
        target_pose = self._make_step_target(swing_foot, support_foot, direction_xy)
        self.attempted_step_count += 1
        self.step_event = StepEvent(
            swing_foot, support_foot, direction_xy.copy(), time_s, time_s, target_pose, self.attempted_step_count,
        )
        self.step_history.append({
            "step_index": int(self.attempted_step_count),
            "swing_foot": swing_foot,
            "support_foot": support_foot,
            "direction_xy": direction_xy.astype(float).tolist(),
            "target_world": target_pose[:3, 3].astype(float).tolist(),
            "trigger_time_s": float(time_s),
            "status": "planned",
        })
        self._step_direction_xy = direction_xy.copy()
        self.step_phase = "transfer"
        self.mode = "transfer"
        self._transfer_start_s = time_s
        support_index = 0 if support_foot == "left_foot" else 1
        support_vertices = np.asarray(self.model.foot_support_vertices_world(), dtype=float)[support_index]
        support_center = np.nanmean(support_vertices[:, :2], axis=0)
        transfer_target = self.com_des.copy()
        com_forward_fraction = float(self.hybrid_config.get("step_com_target_fraction", 0.50))
        transfer_target[:2] = (
            (1.0 - com_forward_fraction) * support_center
            + com_forward_fraction * target_pose[:2, 3]
        )
        self._transfer_target_com = transfer_target
        self.com_des = transfer_target.copy()
        self.com_task_weight_override = float(self.hybrid_config.get("transfer_com_weight", 20.0))
        self.com_task_kp_override = float(self.hybrid_config.get("transfer_com_kp", 140.0))
        self.com_task_kd_override = float(self.hybrid_config.get("transfer_com_kd", 24.0))
        self.set_active_contacts(("left_foot", "right_foot"))
        self.set_swing_target(None)
        self._emit_event(
            "STEP_TRIGGER",
            self._step_trigger_reason or "support_margin",
            step_index=self.attempted_step_count,
            swing_foot=swing_foot,
            support_foot=support_foot,
            target_world=target_pose[:3, 3],
        )

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
        if self.step_event.step_index > 1:
            transfer_duration = float(self.hybrid_config.get("second_step_transfer_duration_s", transfer_duration))
            max_transfer_duration = float(self.hybrid_config.get("second_step_max_transfer_duration_s", max_transfer_duration))
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
        self._emit_event(
            "LIFTOFF",
            "transfer_complete",
            step_index=self.step_event.step_index,
            swing_foot=self.step_event.swing_foot,
        )

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
        duration = float(self.hybrid_config.get("swing_duration_s", 0.34))
        if self.step_event.step_index > 1:
            duration = float(self.hybrid_config.get("second_step_swing_duration_s", duration))
        duration = max(duration, 1e-3)
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
            self._landing_contact_start_s = None
            # Keep only the known support foot constrained until MuJoCo reports
            # sustained load on the landing foot.  Promoting both contacts at
            # the end of the reference trajectory made transient touch points
            # look like a successful recovery.
            self.set_active_contacts((self.step_event.support_foot,))
            self.set_swing_target(self.step_event.swing_foot, self.step_event.target_pose)
            self._emit_event(
                "SWING_COMPLETE",
                "target_reached",
                step_index=self.step_event.step_index,
                swing_foot=self.step_event.swing_foot,
            )

    def _update_landing(self) -> None:
        if self.step_event is None or self.step_phase != "landing":
            return
        now = float(self.model.data.time)
        contact = self.model.actual_contact_data()
        load_threshold = float(self.hybrid_config.get("contact_force_threshold_N", 25.0))
        loaded = contact.contact_flags & (contact.normal_force_N >= load_threshold)
        swing_index = 0 if self.step_event.swing_foot == "left_foot" else 1
        target_error = float(np.linalg.norm(
            self.model.body_pose(self.step_event.swing_foot)[:2, 3] - self.step_event.target_pose[:2, 3],
        ))
        ready = bool(
            loaded[swing_index]
            and contact.tangent_velocity_m_s[swing_index] <= float(self.hybrid_config.get("landing_tangent_speed_m_s", 0.12))
            and target_error <= float(self.hybrid_config.get("landing_position_tolerance_m", 0.08))
        )
        if ready:
            if not self._landing_candidate_active:
                self._landing_contact_start_s = now
                self._landing_candidate_active = True
                self._emit_event(
                    "TOUCHDOWN_CANDIDATE",
                    "load_bearing_contact",
                    step_index=self.step_event.step_index,
                    swing_foot=self.step_event.swing_foot,
                    normal_force_N=float(contact.normal_force_N[swing_index]),
                    support_contact=bool(contact.contact_flags[1 - swing_index]),
                )
            elif now - self._landing_contact_start_s >= float(self.hybrid_config.get("landing_duration_s", 0.18)):
                self.mode = "double_support"
                self.step_phase = "recovered_step"
                self.step_count += 1
                self._update_step_history("landed")
                self._active_support_foot = self.step_event.swing_foot
                self._next_swing_foot = self.step_event.support_foot
                self._cooldown_until_s = now + float(self.hybrid_config.get("step_cooldown_s", 0.35))
                self._settle_until_s = now + float(self.hybrid_config.get("post_landing_duration_s", 0.50))
                self.set_active_contacts(("left_foot", "right_foot"))
                self.set_swing_target(None)
                # Hold the measured double-support center while touchdown
                # transients settle. Returning to the original target before
                # the new support polygon captures the CoM causes an immediate
                # loss of the newly established contact.
                self.com_des = self._double_support_com_target()
                self._emit_event(
                    "TOUCHDOWN",
                    "sustained_load",
                    step_index=self.step_event.step_index,
                    swing_foot=self.step_event.swing_foot,
                )
            return
        self._landing_contact_start_s = None
        self._landing_candidate_active = False
        landing_timeout = float(self.hybrid_config.get("landing_timeout_s", 0.90))
        if self._landing_start_s is not None and now - self._landing_start_s >= landing_timeout:
            self.mode = "failed_recovery"
            self.step_phase = "failed"
            self._failure_reason = "landing_timeout"
            self._update_step_history("failed_landing")
            self._emit_event(
                "RECOVERY_FAIL",
                self._failure_reason,
                step_index=self.step_event.step_index,
                swing_foot=self.step_event.swing_foot,
            )

    def solve(self) -> QPResult:
        if self.mode == "double_support" and self.step_phase == "recovered_step" and self._settle_until_s is not None:
            if float(self.model.data.time) >= self._settle_until_s:
                self.com_task_weight_override = None
                self.com_task_kp_override = None
                self.com_task_kd_override = None
                self._settle_until_s = None
                if self._nominal_com_des is not None:
                    self.com_des = self._nominal_com_des.copy()
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
        self._support_margin_m = self._current_support_margin()
        planned_target = self.step_event.target_pose[:3, 3] if self.step_event is not None else np.full(3, np.nan)
        result.diagnostics.update({
            "control_mode": self.mode,
            "step_phase": self.step_phase,
            "swing_foot": self.step_event.swing_foot if self.step_event else None,
            "support_margin_m": self._support_margin_m,
            "step_count": self.step_count,
            "attempted_step_count": self.attempted_step_count,
            "planned_foot_target_world": planned_target.copy(),
            "event_label": self._pending_event_label,
            "event_reason": self._pending_event_reason,
            "recovery_failure_reason": self._failure_reason,
        })
        self._pending_event_label = ""
        self._pending_event_reason = ""
        return result

    def reset_trial(self) -> None:
        self.set_active_contacts(("left_foot", "right_foot"))
        self.set_swing_target(None)
        self.mode = "double_support"
        self.step_phase = "stance"
        self.step_event = None
        self.step_count = 0
        self.attempted_step_count = 0
        self.events: list[dict] = []
        self.step_history: list[dict] = []
        self._landing_start_s = None
        self._landing_contact_start_s = None
        self._landing_candidate_active = False
        self._cooldown_until_s = 0.0
        self._support_margin_m = float("nan")
        self._step_direction_xy = np.zeros(2, dtype=float)
        self._active_support_foot = None
        self._next_swing_foot = None
        self._last_result = None
        self._swing_start_pose = None
        self._nominal_com_des = None
        self._transfer_target_com = None
        self._transfer_start_s = None
        self._settle_until_s = None
        self._step_trigger_reason = None
        self._failure_reason = None
        self._pending_event_label = ""
        self._pending_event_reason = ""
        self.com_task_weight_override = None
        self.com_task_kp_override = None
        self.com_task_kd_override = None

    def summary(self) -> dict:
        return {
            "controller": "adaptive_hybrid_se3_wbc",
            "step_count": int(self.step_count),
            "attempted_step_count": int(self.attempted_step_count),
            "step_triggered": bool(self.attempted_step_count > 0),
            "swing_foot": self.step_event.swing_foot if self.step_event else None,
            "support_foot": self.step_event.support_foot if self.step_event else None,
            "step_direction_xy": self._step_direction_xy.tolist(),
            "planned_foot_target_world": self.step_event.target_pose[:3, 3].tolist() if self.step_event else None,
            "final_mode": self.mode,
            "final_phase": self.step_phase,
            "trigger_reason": self._step_trigger_reason,
            "failure_reason": self._failure_reason,
            "step_history": self.step_history,
            "events": self.events,
        }
