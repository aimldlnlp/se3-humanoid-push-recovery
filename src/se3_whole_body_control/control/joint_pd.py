"""Joint-space PD reference controller with gravity compensation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PDOutput:
    control: np.ndarray
    generalized_torque: np.ndarray


class JointPDController:
    def __init__(
        self,
        model,
        kp: float = 120.0,
        kd: float = 18.0,
        q_des: np.ndarray | None = None,
        feedforward_control: np.ndarray | None = None,
    ):
        self.model = model
        self.kp = float(kp)
        self.kd = float(kd)
        self.q_des = model.joint_positions() if q_des is None else np.asarray(q_des, dtype=float).copy()
        self.feedforward_control = (
            np.zeros(model.nu, dtype=float)
            if feedforward_control is None
            else np.asarray(feedforward_control, dtype=float).copy()
        )

    def compute(self) -> PDOutput:
        q = self.model.joint_positions()
        qdot = self.model.joint_velocities()
        generalized = np.zeros(self.model.nv)
        generalized[self.model.joint_qvel_indices] = self.kp * (self.q_des - q) - self.kd * qdot
        generalized += self.model.data.qfrc_bias
        control = self.model.actuator_matrix.T @ generalized + self.feedforward_control
        control = np.clip(control, self.model.actuator_limits[:, 0], self.model.actuator_limits[:, 1])
        # The actuator matrix maps control to generalized torque; this is the
        # commanded actuator-space vector that MuJoCo consumes.
        applied = self.model.actuator_matrix @ control
        return PDOutput(control=control, generalized_torque=applied)
