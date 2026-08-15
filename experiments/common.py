"""Shared experiment setup and serialization helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from se3_whole_body_control.config import load_configs, resolve_model_path
from se3_whole_body_control.control.joint_pd import JointPDController
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
from se3_whole_body_control.disturbance.push import Push
from se3_whole_body_control.dynamics.humanoid import HumanoidModel
from se3_whole_body_control.evaluation.metrics import save_trial_npz, summarize_trial
from se3_whole_body_control.evaluation.recovery import RecoveryConfig
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner
from se3_whole_body_control.geometry.so3 import exp_so3


def output_dirs() -> dict[str, Path]:
    dirs = {
        "data": ROOT / "results" / "data",
        "png": ROOT / "results" / "figures" / "png",
        "pdf": ROOT / "results" / "figures" / "pdf",
        "videos": ROOT / "results" / "videos",
        "logs": ROOT / "results" / "logs",
        "frames": ROOT / "results" / "frames",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_model(configs: dict | None = None, mass_scale: float = 1.0, friction_coefficient: float | None = None) -> HumanoidModel:
    cfg = configs or load_configs(ROOT)
    friction = float(friction_coefficient if friction_coefficient is not None else cfg["controller"].get("friction_coefficient", cfg["robot"].get("friction_coefficient", 0.7)))
    return HumanoidModel(
        resolve_model_path(cfg),
        mass_scale=mass_scale,
        friction_coefficient=friction,
        robot_config=cfg["robot"],
    )


def recovery_config(configs: dict) -> RecoveryConfig:
    return RecoveryConfig(**configs["experiments"]["recovery"])


def make_push(configs: dict, magnitude: float | None = None, direction_deg: float | None = None, duration: float | None = None, start: float | None = None) -> Push:
    p = configs["experiments"]["push"]
    return Push(
        magnitude_N=float(p["magnitude_N"] if magnitude is None else magnitude),
        direction_rad=float(np.deg2rad(p["direction_deg"] if direction_deg is None else direction_deg)),
        duration_s=float(p["duration_s"] if duration is None else duration),
        start_time_s=float(p["start_time_s"] if start is None else start),
        application_body=str(p["application_body"]),
    )


def run_controller(controller_name: str, model, configs: dict):
    c = configs["controller"]
    if controller_name in {"pd", "joint_pd"}:
        controller = JointPDController(model, kp=c["posture_kp"], kd=c["posture_kd"])
        if bool(c.get("use_nominal_contact_feedforward", False)):
            # A fixed, disturbance-unaware contact-equilibrium bias makes the
            # PD reference physically valid on the G1's multi-point foot
            # contacts. It is computed once at nominal standing and never
            # updated from the push or measured GRF during the trial.
            bootstrap = WholeBodyQPController(model, c, configs["experiments"].get("recovery"))
            equilibrium = bootstrap.solve().control
            gravity_pd = model.actuator_matrix.T @ model.data.qfrc_bias
            controller.feedforward_control = np.asarray(equilibrium - gravity_pd, dtype=float)
        return controller
    return WholeBodyQPController(model, c, configs["experiments"]["recovery"])


def run_trial(controller_name: str, configs: dict, push: Push | None = None, duration: float | None = None, mass_scale: float = 1.0, seed: int = 0, classify: bool = False, frame_callback=None, initial_qpos: np.ndarray | None = None, initial_qvel: np.ndarray | None = None, friction_coefficient: float | None = None):
    model = make_model(configs, mass_scale=mass_scale, friction_coefficient=friction_coefficient)
    nominal_qpos = model.qpos0.copy()
    nominal_torso = model.body_pose("torso").copy()
    nominal_pelvis = model.body_pose("pelvis").copy()
    nominal_com = model.center_of_mass().copy()
    controller = run_controller(controller_name, model, configs)
    perturbed = initial_qpos is not None or initial_qvel is not None
    runner = SimulationRunner(
        model, controller,
        duration_s=duration or configs["robot"]["duration_s"],
        control_timestep_s=configs["robot"]["control_timestep"],
        # Perturbed trials start directly from the requested qpos/qvel. A PD
        # warmup here would silently erase part of the disturbance before the
        # measured WBC/PD response begins.
        warmup_duration_s=(0.0 if perturbed else configs["robot"].get("warmup_duration_s", 0.4)),
        warmup_reanchor=not perturbed,
    )
    return model, runner.run(
        push=push, recovery_config=recovery_config(configs), classify=classify, seed=seed,
        frame_callback=frame_callback, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
        desired_torso=nominal_torso if perturbed else None,
        desired_pelvis=nominal_pelvis if perturbed else None,
        com_reference=nominal_com if perturbed else None,
    )


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a near-identity rotation matrix to MuJoCo's wxyz quaternion."""
    trace = float(np.trace(R))
    w = 0.5 * np.sqrt(max(1.0 + trace, 1e-12))
    xyz = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / max(4.0 * w, 1e-12)
    quat = np.r_[w, xyz]
    return quat / max(np.linalg.norm(quat), 1e-12)


def randomized_initial_state(configs: dict, seed: int, mass_scale: float = 1.0, friction_coefficient: float | None = None, randomization: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a reproducible, physically meaningful initial perturbation."""
    rng = np.random.default_rng(seed)
    random_cfg = randomization or configs["experiments"].get("robustness_randomization", {})
    model = make_model(configs, mass_scale=mass_scale, friction_coefficient=friction_coefficient)
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv, dtype=float)
    free_joints = [j for j in range(model.model.njnt) if model.model.jnt_type[j] == 0]
    if not free_joints:
        return qpos, qvel, {"tilt_rad": 0.0, "com_offset_m": 0.0, "angular_velocity_rad_s": 0.0}
    joint_id = free_joints[0]
    qpos_adr = int(model.model.jnt_qposadr[joint_id])
    qvel_adr = int(model.model.jnt_dofadr[joint_id])
    tilt_limit = float(random_cfg.get("initial_torso_tilt_rad", 0.0))
    tilt = rng.normal(size=2)
    tilt /= max(np.linalg.norm(tilt), 1e-12)
    tilt *= rng.uniform(0.0, tilt_limit)
    qpos[qpos_adr + 3 : qpos_adr + 7] = _rotation_to_quaternion(exp_so3(np.r_[tilt, 0.0]))
    offset_limit = float(random_cfg.get("initial_com_offset_m", 0.0))
    offset = rng.normal(size=2)
    offset /= max(np.linalg.norm(offset), 1e-12)
    offset *= rng.uniform(0.0, offset_limit)
    qpos[qpos_adr : qpos_adr + 2] += offset
    angular_limit = float(random_cfg.get("initial_angular_velocity_rad_s", 0.0))
    angular_velocity = rng.normal(size=3)
    angular_velocity /= max(np.linalg.norm(angular_velocity), 1e-12)
    angular_velocity *= rng.uniform(0.0, angular_limit)
    qvel[qvel_adr + 3 : qvel_adr + 6] = angular_velocity
    return qpos, qvel, {
        "tilt_rad": float(np.linalg.norm(tilt)),
        "com_offset_m": float(np.linalg.norm(offset)),
        "angular_velocity_rad_s": float(np.linalg.norm(angular_velocity)),
    }


def save_run(run, path: Path, metadata: dict | None = None) -> None:
    meta = dict(metadata or {})
    # Carry the physical run identity into every artifact.  Keep paths
    # portable so committed results never expose an operator's local checkout.
    for key, value in run.metadata.items():
        if key == "model_path":
            value = Path(value)
        meta.setdefault(key, value)
    meta.setdefault("seed", int(run.metadata.get("seed", 0)))
    meta["manifest"] = execution_manifest(meta, seed=meta["seed"])
    meta = _json_safe(meta)
    meta["summary"] = summarize_trial(run.log)
    if run.recovery is not None:
        meta["recovery"] = asdict(run.recovery)
    extra_arrays = {"qpos_history": np.asarray(run.qpos_history, dtype=float)} if run.qpos_history else None
    save_trial_npz(run.log, path, meta, extra_arrays=extra_arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, Path):
        try:
            return value.relative_to(ROOT).as_posix()
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(key): ("." if key == "root" else _json_safe(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _source_version() -> str:
    frozen = os.environ.get("SE3_SOURCE_VERSION")
    if frozen:
        return frozen
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        digest = hashlib.sha256()
        roots = (ROOT / "configs", ROOT / "models", ROOT / "experiments", ROOT / "scripts", ROOT / "src")
        for base in roots:
            if not base.exists():
                continue
            for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
        return f"source-sha256:{digest.hexdigest()[:16]}"


def execution_manifest(metadata: dict | None = None, seed: int = 0) -> dict:
    """Return portable provenance for a trial or a sweep invocation."""
    metadata = metadata or {}
    config = metadata.get("config")
    portable_config = _json_safe(config) if config is not None else None
    config_sha256 = None
    if config is not None:
        config_sha256 = hashlib.sha256(json.dumps(portable_config, sort_keys=True, default=str).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = metadata.get("run_id") or os.environ.get("SE3_RUN_ID")
    return {
        "run_id": run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "source_version": _source_version(),
        "seed": int(seed),
        "hostname": socket.gethostname(),
        "timestamp_utc": timestamp,
        "config_sha256": config_sha256,
    }


def write_execution_manifest(path: Path, configs: dict, seed: int = 0, extra: dict | None = None) -> None:
    portable_config = _json_safe(configs)
    payload = {"config": portable_config, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(execution_manifest(payload, seed=seed) | {"config": portable_config, **(extra or {})}, indent=2, default=str), encoding="utf-8")


def flatten_result(run, controller: str, push: Push, trial_id: str, seed: int = 0, extra: dict | None = None) -> dict:
    rec = run.recovery
    a = run.log.arrays()
    row = {
        "trial_id": trial_id,
        "controller": controller,
        "push_magnitude_N": push.magnitude_N,
        "push_direction_deg": push.direction_deg,
        "push_duration_s": push.duration_s,
        "impulse_Ns": push.impulse_Ns,
        "success": bool(rec.success) if rec else False,
        "failure_reason": rec.failure_reason if rec else "NOT_CLASSIFIED",
        "recovered_at_s": rec.recovered_at_s if rec and rec.recovered_at_s is not None else "",
        "recovery_latency_s": rec.recovery_latency_s if rec and rec.recovery_latency_s is not None else "",
        "max_torso_error_rad": rec.max_torso_error_rad if rec else float(np.max(a["torso_rotation_error_rad"])),
        "max_com_displacement_m": rec.max_com_displacement_m if rec else float(np.max(np.linalg.norm(a["com_world"] - a["com_world"][0], axis=1))),
        "max_joint_torque_Nm": rec.max_joint_torque_Nm if rec else float(np.max(a["torque_abs_max_Nm"])),
        "max_contact_force_N": (
            float(np.max(np.abs(a["actual_contact_wrench"][:, [0, 1, 2, 6, 7, 8]])))
            if len(a["actual_contact_wrench"]) else 0.0
        ),
        "min_friction_margin": rec.min_friction_margin if rec else float(np.nanmin(a["actual_friction_margin"])),
        "seed": seed,
    }
    mass_kg = float(run.metadata.get("mass_kg", float("nan")))
    force_over_mg, impulse_over_mass = push.normalized(mass_kg) if np.isfinite(mass_kg) else (float("nan"), float("nan"))
    row.update({
        "mass_kg": mass_kg,
        "force_over_mg": force_over_mg,
        "impulse_over_mass_m_s": impulse_over_mass,
    })
    row.update(extra or {})
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
