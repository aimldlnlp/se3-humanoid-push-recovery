"""Diagnostic-only WBC wrapper with measured-contact mode handoff."""

from __future__ import annotations

import json

import numpy as np

from .diagnostic_contact_supervisor import MeasuredContactModeSupervisor
from .diagnostic_telemetry import CorrectedDiagnosticWholeBodyQPController


class ContactAwareDiagnosticWholeBodyQPController(CorrectedDiagnosticWholeBodyQPController):
    """Corrected diagnostic WBC with a debounced measured-contact supervisor.

    The wrapper changes only the diagnostic replay's active contact set. It
    reuses the production controller's set_active_contacts implementation and
    leaves all task weights, physical limits, friction assumptions, and QP
    constraints unchanged.
    """

    def __init__(
        self,
        model,
        controller_config: dict,
        recovery_config: dict | None = None,
        *,
        supervisor: MeasuredContactModeSupervisor,
        control_timestep_s: float = 0.004,
    ):
        if not isinstance(supervisor, MeasuredContactModeSupervisor):
            raise TypeError("supervisor must be a MeasuredContactModeSupervisor")
        self.contact_supervisor = supervisor
        super().__init__(
            model,
            controller_config,
            recovery_config,
            profile="baseline_landed_support",
            control_timestep_s=control_timestep_s,
        )
        self.set_active_contacts(self.contact_supervisor.active_contacts)
        self.allows_single_support = True
        self.requires_final_double_support = False
        self.contact_mode_history: list[dict] = []

    def reset_trial(self) -> None:
        super().reset_trial()
        self.contact_supervisor.reset()
        self.set_active_contacts(self.contact_supervisor.active_contacts)
        self.allows_single_support = True
        self.requires_final_double_support = False
        self.contact_mode_history.clear()

    def solve(self):
        model = self.internal_model
        measured = model.actual_contact_data()
        time_s = float(model.data.time)
        decision = self.contact_supervisor.update(measured, time_s)
        if tuple(self.contact_names) != tuple(decision.active_contacts):
            self.set_active_contacts(decision.active_contacts)
        result = super().solve()
        record = {
            "time_s": decision.time_s,
            "active_contacts": list(decision.active_contacts),
            "previous_active_contacts": list(decision.previous_active_contacts),
            "loaded_contacts": list(decision.loaded_contacts),
            "contact_flags": np.asarray(decision.contact_flags, dtype=bool),
            "normal_force_N": np.asarray(decision.normal_force_N, dtype=float),
            "changed": decision.changed,
            "event": decision.event,
            "loaded_streak": np.asarray(decision.loaded_streak, dtype=int),
            "unloaded_streak": np.asarray(decision.unloaded_streak, dtype=int),
        }
        self.contact_mode_history.append(record)
        result.diagnostics = dict(result.diagnostics or {})
        result.diagnostics.update({
            "active_contacts": list(decision.active_contacts),
            "contact_supervisor_loaded_contacts": list(decision.loaded_contacts),
            "contact_supervisor_contact_flags": list(decision.contact_flags),
            "contact_supervisor_changed": bool(decision.changed),
            "contact_supervisor_event": decision.event,
        })
        if self.diagnostic_history:
            self.diagnostic_history[-1].update({
                "contact_supervisor_active_contacts": list(decision.active_contacts),
                "contact_supervisor_previous_active_contacts": list(decision.previous_active_contacts),
                "contact_supervisor_loaded_contacts": list(decision.loaded_contacts),
                "contact_supervisor_contact_flags": np.asarray(decision.contact_flags, dtype=bool),
                "contact_supervisor_normal_force_N": np.asarray(decision.normal_force_N, dtype=float),
                "contact_supervisor_changed": bool(decision.changed),
                "contact_supervisor_event": decision.event,
                "contact_supervisor_loaded_streak": np.asarray(decision.loaded_streak, dtype=int),
                "contact_supervisor_unloaded_streak": np.asarray(decision.unloaded_streak, dtype=int),
            })
        return result

    def history_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().history_arrays()
        history = self.contact_mode_history
        arrays.update({
            "contact_supervisor_active_contacts_json": np.asarray(
                [json.dumps(item["active_contacts"], separators=(",", ":")) for item in history],
                dtype="U64",
            ),
            "contact_supervisor_previous_active_contacts_json": np.asarray(
                [json.dumps(item["previous_active_contacts"], separators=(",", ":")) for item in history],
                dtype="U64",
            ),
            "contact_supervisor_loaded_contacts_json": np.asarray(
                [json.dumps(item["loaded_contacts"], separators=(",", ":")) for item in history],
                dtype="U64",
            ),
            "contact_supervisor_contact_flags": np.stack(
                [np.asarray(item["contact_flags"], dtype=bool) for item in history],
            ) if history else np.zeros((0, 2), dtype=bool),
            "contact_supervisor_normal_force_N": np.stack(
                [np.asarray(item["normal_force_N"], dtype=float) for item in history],
            ) if history else np.zeros((0, 2), dtype=float),
            "contact_supervisor_changed": np.asarray(
                [bool(item["changed"]) for item in history], dtype=bool,
            ),
            "contact_supervisor_event": np.asarray(
                [str(item["event"]) for item in history], dtype="U96",
            ),
            "contact_supervisor_loaded_streak": np.stack(
                [np.asarray(item["loaded_streak"], dtype=int) for item in history],
            ) if history else np.zeros((0, 2), dtype=int),
            "contact_supervisor_unloaded_streak": np.stack(
                [np.asarray(item["unloaded_streak"], dtype=int) for item in history],
            ) if history else np.zeros((0, 2), dtype=int),
        })
        return arrays

    def summary(self) -> dict:
        summary = super().summary()
        summary.update({
            "contact_mode_supervisor": {
                "load_threshold_N": self.contact_supervisor.load_threshold_N,
                "release_threshold_N": self.contact_supervisor.release_threshold_N,
                "debounce_samples": self.contact_supervisor.debounce_samples,
            },
            "contact_mode_changes": int(sum(bool(item["changed"]) for item in self.contact_mode_history)),
            "active_contacts": list(self.contact_names),
        })
        return summary
