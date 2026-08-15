"""Objective recovery classification with explicit failure precedence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FailureReason = Literal[
    "FALL", "SLIP", "CONTACT_LOSS", "TORQUE_LIMIT", "JOINT_LIMIT", "QP_FAILURE", "TIMEOUT", "NUMERICAL_FAILURE",
]


@dataclass(frozen=True)
class RecoveryConfig:
    orientation_threshold_rad: float = np.deg2rad(5.0)
    angular_velocity_threshold_rad_s: float = 0.15
    com_displacement_threshold_m: float = 0.10
    stable_duration_s: float = 0.25
    timeout_s: float = 6.0
    startup_grace_period_s: float = 0.25
    contact_loss_duration_s: float = 0.08
    friction_utilization_threshold: float = 1.05
    slip_tangent_velocity_threshold_m_s: float = 0.08
    slip_displacement_threshold_m: float = 0.025
    slip_duration_s: float = 0.04
    foot_contact_required: bool = True
    torso_ground_height_m: float = 0.35


@dataclass(frozen=True)
class RecoveryResult:
    success: bool
    failure_reason: str | None
    recovered_at_s: float | None
    recovery_latency_s: float | None
    max_torso_error_rad: float
    max_com_displacement_m: float
    max_joint_torque_Nm: float
    min_friction_margin: float


def classify_recovery(
    time_s: np.ndarray,
    orientation_error_rad: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
    com_displacement_m: np.ndarray,
    contact_left: np.ndarray,
    contact_right: np.ndarray,
    torso_height_m: np.ndarray,
    torque_abs_max_Nm: np.ndarray,
    qp_ok: np.ndarray,
    friction_margin: np.ndarray,
    config: RecoveryConfig | None = None,
    actual_friction_utilization: np.ndarray | None = None,
    foot_tangent_velocity: np.ndarray | None = None,
    foot_xy_displacement: np.ndarray | None = None,
    torque_utilization: np.ndarray | None = None,
    joint_limit_violation: np.ndarray | None = None,
    numerical_valid: np.ndarray | None = None,
    push_end_s: float = 0.0,
    recovery_start_s: float | None = None,
) -> RecoveryResult:
    cfg = config or RecoveryConfig()
    time_s = np.asarray(time_s, dtype=float)
    ori = np.asarray(orientation_error_rad, dtype=float)
    ang = np.asarray(angular_velocity_rad_s, dtype=float)
    com = np.asarray(com_displacement_m, dtype=float)
    if recovery_start_s is not None:
        push_end_s = float(recovery_start_s)
    if len(time_s) == 0 or any(not np.all(np.isfinite(a)) for a in (time_s, ori, ang, com)):
        return RecoveryResult(False, "NUMERICAL_FAILURE", None, None, np.inf, np.inf, np.inf, -np.inf)
    max_ori = float(np.max(ori))
    max_com = float(np.max(com))
    max_tau = float(np.max(torque_abs_max_Nm)) if len(torque_abs_max_Nm) else 0.0
    finite_margin = np.asarray(friction_margin, dtype=float)
    min_margin = float(np.nanmin(finite_margin)) if np.any(np.isfinite(finite_margin)) else float("nan")
    eval_start = max(cfg.startup_grace_period_s, float(push_end_s))
    eval_mask = time_s >= eval_start
    if numerical_valid is not None and not np.all(np.asarray(numerical_valid, dtype=bool)[eval_mask]):
        reason = "NUMERICAL_FAILURE"
    elif np.any(np.asarray(torso_height_m)[eval_mask] < cfg.torso_ground_height_m):
        reason = "FALL"
    elif cfg.foot_contact_required and _sustained_contact_loss(time_s, contact_left, contact_right, cfg, eval_start):
        reason = "CONTACT_LOSS"
    elif _slip_event(time_s, eval_mask, actual_friction_utilization, foot_tangent_velocity, foot_xy_displacement, cfg):
        reason = "SLIP"
    elif torque_utilization is not None and np.any(np.asarray(torque_utilization, dtype=float)[eval_mask] > 1.0005):
        reason = "TORQUE_LIMIT"
    elif joint_limit_violation is not None and np.any(np.asarray(joint_limit_violation, dtype=bool)[eval_mask]):
        reason = "JOINT_LIMIT"
    elif not np.all(np.asarray(qp_ok, dtype=bool)[eval_mask]):
        reason = "QP_FAILURE"
    else:
        reason = None
    stable = (
        (ori <= cfg.orientation_threshold_rad)
        & (ang <= cfg.angular_velocity_threshold_rad_s)
        & (com <= cfg.com_displacement_threshold_m)
    )
    recovered_at = None
    recovery_start = eval_start
    if len(time_s) > 1:
        for i in range(len(time_s)):
            if time_s[i] < recovery_start:
                continue
            if not stable[i]:
                continue
            j = i
            while j < len(time_s) and stable[j]:
                if time_s[j] - time_s[i] >= cfg.stable_duration_s:
                    recovered_at = float(time_s[i])
                    break
                j += 1
            if recovered_at is not None:
                break
    if recovered_at is None and reason is None:
        reason = "TIMEOUT"
    success = reason is None and recovered_at is not None
    latency = None if recovered_at is None else max(0.0, recovered_at - float(push_end_s))
    return RecoveryResult(success, reason, recovered_at, latency, max_ori, max_com, max_tau, min_margin)


def _slip_event(time_s, eval_mask, utilization, tangent_velocity, displacement, cfg) -> bool:
    if utilization is None and tangent_velocity is None and displacement is None:
        return False
    mask = np.zeros(len(time_s), dtype=bool)
    if utilization is not None:
        mask |= np.any(np.nan_to_num(utilization, nan=0.0)[..., :] > cfg.friction_utilization_threshold, axis=1)
    if tangent_velocity is not None:
        mask |= np.any(np.nan_to_num(tangent_velocity, nan=0.0)[..., :] > cfg.slip_tangent_velocity_threshold_m_s, axis=1)
    if displacement is not None:
        mask |= np.any(np.nan_to_num(displacement, nan=0.0)[..., :] > cfg.slip_displacement_threshold_m, axis=1)
    return _sustained_event(time_s, mask & eval_mask, cfg.slip_duration_s)


def _sustained_contact_loss(time_s: np.ndarray, left: np.ndarray, right: np.ndarray, cfg: RecoveryConfig, start_s: float | None = None) -> bool:
    bad = ~(np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool))
    bad &= time_s >= (cfg.startup_grace_period_s if start_s is None else start_s)
    if not np.any(bad):
        return False
    starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
    ends = np.flatnonzero(bad & ~np.r_[bad[1:], False])
    return any(time_s[end] - time_s[start] >= cfg.contact_loss_duration_s for start, end in zip(starts, ends))


def _sustained_event(time_s: np.ndarray, event: np.ndarray, duration_s: float) -> bool:
    if not np.any(event):
        return False
    starts = np.flatnonzero(event & ~np.r_[False, event[:-1]])
    ends = np.flatnonzero(event & ~np.r_[event[1:], False])
    return any(time_s[end] - time_s[start] >= duration_s for start, end in zip(starts, ends))
