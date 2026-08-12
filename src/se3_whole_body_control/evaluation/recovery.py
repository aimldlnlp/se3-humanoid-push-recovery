"""Objective recovery classification with explicit failure precedence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FailureReason = Literal[
    "FALL", "SLIP", "CONTACT_LOSS", "TORQUE_LIMIT", "JOINT_LIMIT",
    "QP_FAILURE", "TIMEOUT", "NUMERICAL_FAILURE",
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
    friction_margin_tolerance_N: float = 0.01
    foot_contact_required: bool = True
    torso_ground_height_m: float = 0.35


@dataclass(frozen=True)
class RecoveryResult:
    success: bool
    failure_reason: str | None
    recovery_time_s: float | None
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
    recovery_start_s: float = 0.0,
) -> RecoveryResult:
    cfg = config or RecoveryConfig()
    time_s = np.asarray(time_s, dtype=float)
    ori = np.asarray(orientation_error_rad, dtype=float)
    ang = np.asarray(angular_velocity_rad_s, dtype=float)
    com = np.asarray(com_displacement_m, dtype=float)
    if len(time_s) == 0 or any(not np.all(np.isfinite(a)) for a in (time_s, ori, ang, com)):
        return RecoveryResult(False, "NUMERICAL_FAILURE", None, np.inf, np.inf, np.inf, -np.inf)
    max_ori = float(np.max(ori))
    max_com = float(np.max(com))
    max_tau = float(np.max(torque_abs_max_Nm)) if len(torque_abs_max_Nm) else 0.0
    min_margin = float(np.min(friction_margin)) if len(friction_margin) else float("nan")
    eval_mask = time_s >= max(cfg.startup_grace_period_s, float(recovery_start_s))
    if np.any(np.asarray(torso_height_m)[eval_mask] < cfg.torso_ground_height_m):
        reason = "FALL"
    elif cfg.foot_contact_required and _sustained_contact_loss(time_s, contact_left, contact_right, cfg):
        reason = "CONTACT_LOSS"
    elif np.any(np.asarray(friction_margin)[eval_mask] < -cfg.friction_margin_tolerance_N):
        reason = "SLIP"
    elif not np.all(np.asarray(qp_ok, dtype=bool)[eval_mask]):
        reason = "QP_FAILURE"
    elif time_s[-1] + 1e-9 >= cfg.timeout_s:
        reason = "TIMEOUT"
    else:
        reason = None
    stable = (
        (ori <= cfg.orientation_threshold_rad)
        & (ang <= cfg.angular_velocity_threshold_rad_s)
        & (com <= cfg.com_displacement_threshold_m)
    )
    recovery_time = None
    recovery_start = max(cfg.startup_grace_period_s, float(recovery_start_s))
    if len(time_s) > 1:
        for i in range(len(time_s)):
            if time_s[i] < recovery_start:
                continue
            if not stable[i]:
                continue
            j = i
            while j < len(time_s) and stable[j]:
                if time_s[j] - time_s[i] >= cfg.stable_duration_s:
                    recovery_time = float(time_s[i])
                    break
                j += 1
            if recovery_time is not None:
                break
    success = reason is None and recovery_time is not None
    if not success and reason is None:
        reason = "TIMEOUT"
    return RecoveryResult(success, reason, recovery_time, max_ori, max_com, max_tau, min_margin)


def _sustained_contact_loss(time_s: np.ndarray, left: np.ndarray, right: np.ndarray, cfg: RecoveryConfig) -> bool:
    bad = ~(np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool))
    bad &= time_s >= cfg.startup_grace_period_s
    if not np.any(bad):
        return False
    starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
    ends = np.flatnonzero(bad & ~np.r_[bad[1:], False])
    return any(time_s[end] - time_s[start] >= cfg.contact_loss_duration_s for start, end in zip(starts, ends))
