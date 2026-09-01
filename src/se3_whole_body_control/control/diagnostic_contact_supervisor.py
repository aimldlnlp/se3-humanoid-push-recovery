"""Diagnostic-only measured-contact mode supervision.

This module provides the smallest contact-mode intervention used by the
post-touchdown mechanism study. It does not alter the production hybrid
controller. The supervisor consumes measured MuJoCo contact data and emits a
non-empty, canonically ordered foot set for WholeBodyQPController
set_active_contacts after deterministic hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CANONICAL_FOOT_NAMES = ("left_foot", "right_foot")


def _canonical_names(names) -> tuple[str, ...]:
    values = tuple(str(name) for name in names)
    if not values or any(name not in CANONICAL_FOOT_NAMES for name in values):
        raise ValueError(f"contact names must be a non-empty subset of {CANONICAL_FOOT_NAMES}")
    if len(set(values)) != len(values):
        raise ValueError("contact names must be unique")
    return tuple(name for name in CANONICAL_FOOT_NAMES if name in values)


@dataclass(frozen=True)
class ContactModeDecision:
    """One measured-contact supervisor decision at a pre-step timestamp."""

    time_s: float
    active_contacts: tuple[str, ...]
    previous_active_contacts: tuple[str, ...]
    loaded_contacts: tuple[str, ...]
    contact_flags: tuple[bool, bool]
    normal_force_N: tuple[float, float]
    changed: bool
    event: str
    loaded_streak: tuple[int, int]
    unloaded_streak: tuple[int, int]


class MeasuredContactModeSupervisor:
    """Debounced contact handoff for a diagnostic replay.

    A foot is acquired only when MuJoCo reports a contact flag and at least
    load_threshold_N normal force for debounce_samples consecutive controller
    observations. An active foot is released only after its measured load is
    below release_threshold_N (or contact is absent) for the same number of
    observations. The hysteresis band prevents a 4 N contact from chattering
    across a 5 N acquisition threshold.

    The supervisor never emits an empty active-contact set. If all measured
    support disappears, it retains the previous non-empty WBC mode and records
    hold_no_loaded_support for diagnosis.
    """

    def __init__(
        self,
        initial_active_contacts: tuple[str, ...] | list[str] = ("right_foot",),
        *,
        load_threshold_N: float = 5.0,
        release_threshold_N: float = 3.0,
        debounce_samples: int = 2,
    ):
        self.initial_active_contacts = _canonical_names(initial_active_contacts)
        self.load_threshold_N = float(load_threshold_N)
        self.release_threshold_N = float(release_threshold_N)
        self.debounce_samples = int(debounce_samples)
        if not np.isfinite(self.load_threshold_N) or self.load_threshold_N <= 0.0:
            raise ValueError("load_threshold_N must be finite and positive")
        if not np.isfinite(self.release_threshold_N) or self.release_threshold_N < 0.0:
            raise ValueError("release_threshold_N must be finite and non-negative")
        if self.release_threshold_N >= self.load_threshold_N:
            raise ValueError("release_threshold_N must be below load_threshold_N")
        if self.debounce_samples < 1:
            raise ValueError("debounce_samples must be at least one")
        self.reset()

    def reset(self) -> None:
        self.active_contacts = self.initial_active_contacts
        self.loaded_streak = np.zeros(2, dtype=int)
        self.unloaded_streak = np.zeros(2, dtype=int)
        self.last_decision: ContactModeDecision | None = None

    @staticmethod
    def _contact_arrays(contact_data) -> tuple[np.ndarray, np.ndarray]:
        flags = np.asarray(contact_data.contact_flags, dtype=bool).reshape(-1)
        normal = np.asarray(contact_data.normal_force_N, dtype=float).reshape(-1)
        if flags.shape != (2,) or normal.shape != (2,):
            raise ValueError(
                f"contact data must contain two feet: flags={flags.shape}, normal_force_N={normal.shape}"
            )
        return flags, normal

    def update(self, contact_data, time_s: float) -> ContactModeDecision:
        """Consume one pre-step measured contact observation."""

        flags, normal = self._contact_arrays(contact_data)
        loaded = flags & np.isfinite(normal) & (normal >= self.load_threshold_N)
        unloaded = (~flags) | ~np.isfinite(normal) | (normal < self.release_threshold_N)
        self.loaded_streak = np.where(loaded, self.loaded_streak + 1, 0)
        self.unloaded_streak = np.where(unloaded, self.unloaded_streak + 1, 0)

        previous = self.active_contacts
        candidate = set(previous)
        for index, name in enumerate(CANONICAL_FOOT_NAMES):
            if name not in candidate and self.loaded_streak[index] >= self.debounce_samples:
                candidate.add(name)
            if name in candidate and self.unloaded_streak[index] >= self.debounce_samples:
                candidate.discard(name)

        if not candidate:
            active = previous
            event = "hold_no_loaded_support"
        else:
            active = _canonical_names(candidate)
            event = "mode_change:" + "+".join(active) if active != previous else "hold"
        self.active_contacts = active
        decision = ContactModeDecision(
            time_s=float(time_s),
            active_contacts=active,
            previous_active_contacts=previous,
            loaded_contacts=tuple(
                name for index, name in enumerate(CANONICAL_FOOT_NAMES) if bool(loaded[index])
            ),
            contact_flags=(bool(flags[0]), bool(flags[1])),
            normal_force_N=(float(normal[0]), float(normal[1])),
            changed=bool(active != previous),
            event=event,
            loaded_streak=(int(self.loaded_streak[0]), int(self.loaded_streak[1])),
            unloaded_streak=(int(self.unloaded_streak[0]), int(self.unloaded_streak[1])),
        )
        self.last_decision = decision
        return decision
