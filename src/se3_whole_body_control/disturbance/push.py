"""Deterministic horizontal push disturbances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Push:
    magnitude_N: float
    direction_rad: float
    duration_s: float
    start_time_s: float
    application_body: str
    application_point_local: np.ndarray | None = None

    @property
    def impulse_Ns(self) -> float:
        return float(self.magnitude_N * self.duration_s)

    @property
    def direction_deg(self) -> float:
        return float(np.rad2deg(self.direction_rad))

    def normalized(self, mass_kg: float, gravity_m_s2: float = 9.81) -> tuple[float, float]:
        """Return the measured dimensionless force and impulse for a model mass."""
        mass = max(float(mass_kg), 1e-12)
        return float(self.magnitude_N / (mass * gravity_m_s2)), float(self.impulse_Ns / mass)


def active_push(push: Push | None, time_s: float) -> bool:
    return push is not None and push.start_time_s <= time_s < push.start_time_s + push.duration_s


def push_force(push: Push | None, time_s: float) -> np.ndarray:
    if not active_push(push, time_s):
        return np.zeros(3)
    return float(push.magnitude_N) * np.array([
        np.cos(push.direction_rad), np.sin(push.direction_rad), 0.0
    ])
