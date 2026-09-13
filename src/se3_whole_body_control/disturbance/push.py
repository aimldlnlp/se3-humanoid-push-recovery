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
    if push is None:
        return False
    # MuJoCo advances time by repeated floating-point additions. Compare in
    # elapsed-time coordinates with a tiny boundary tolerance so a pulse whose
    # duration is an exact multiple of the physics step cannot acquire an extra
    # substep at its nominal end time.
    elapsed_s = float(time_s) - float(push.start_time_s)
    boundary_tolerance_s = 1e-12
    return -boundary_tolerance_s <= elapsed_s < float(push.duration_s) - boundary_tolerance_s


def push_force(push: Push | None, time_s: float) -> np.ndarray:
    if not active_push(push, time_s):
        return np.zeros(3)
    return float(push.magnitude_N) * np.array([
        np.cos(push.direction_rad), np.sin(push.direction_rad), 0.0
    ])
