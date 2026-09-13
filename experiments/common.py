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
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from se3_whole_body_control.config import load_configs, resolve_model_path
from se3_whole_body_control.control.joint_pd import JointPDController
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController
from se3_whole_body_control.control.hybrid_recovery import HybridRecoveryController
from se3_whole_body_control.disturbance.push import Push
from se3_whole_body_control.dynamics.humanoid import HumanoidModel
from se3_whole_body_control.evaluation.metrics import save_trial_npz, summarize_trial
from se3_whole_body_control.evaluation.recovery import RecoveryConfig
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner
from se3_whole_body_control.geometry.so3 import exp_so3


@dataclass(frozen=True)
class PairedInitialCondition:
    qpos: np.ndarray
    qvel: np.ndarray
    desired_torso: np.ndarray
    desired_pelvis: np.ndarray
    com_reference: np.ndarray
    joint_reference: np.ndarray
    seed: int
    setup_policy: str
    setup_duration_s: float

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        values = (self.qpos, self.qvel, self.desired_torso, self.desired_pelvis, self.com_reference, self.joint_reference)
        for value in values:
            array = np.ascontiguousarray(value, dtype='<f8')
            digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
            digest.update(array.tobytes())
        digest.update(str(int(self.seed)).encode('ascii'))
        digest.update(self.setup_policy.encode('utf-8'))
        digest.update(np.asarray([self.setup_duration_s], dtype='<f8').tobytes())
        return digest.hexdigest()

    @staticmethod
    def _array_sha256(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value, dtype='<f8')
        return hashlib.sha256(array.tobytes()).hexdigest()

    def metadata(self) -> dict:
        return {
            'sha256': self.sha256,
            'qpos_sha256': self._array_sha256(self.qpos),
            'qvel_sha256': self._array_sha256(self.qvel),
            'torso_reference_sha256': self._array_sha256(self.desired_torso),
            'pelvis_reference_sha256': self._array_sha256(self.desired_pelvis),
            'com_reference_sha256': self._array_sha256(self.com_reference),
            'joint_reference_sha256': self._array_sha256(self.joint_reference),
            'seed': int(self.seed),
            'setup_policy': self.setup_policy,
            'setup_duration_s': float(self.setup_duration_s),
        }

    def json_payload(self) -> dict:
        return {
            **self.metadata(),
            'qpos': np.asarray(self.qpos, dtype=float).tolist(),
            'qvel': np.asarray(self.qvel, dtype=float).tolist(),
            'desired_torso': np.asarray(self.desired_torso, dtype=float).tolist(),
            'desired_pelvis': np.asarray(self.desired_pelvis, dtype=float).tolist(),
            'com_reference': np.asarray(self.com_reference, dtype=float).tolist(),
            'joint_reference': np.asarray(self.joint_reference, dtype=float).tolist(),
        }


def save_paired_initial_condition(condition: PairedInitialCondition, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=condition.qpos,
        qvel=condition.qvel,
        desired_torso=condition.desired_torso,
        desired_pelvis=condition.desired_pelvis,
        com_reference=condition.com_reference,
        joint_reference=condition.joint_reference,
        metadata_json=np.asarray(json.dumps(condition.metadata(), sort_keys=True)),
    )
    path.with_suffix('.json').write_text(
        json.dumps(condition.json_payload(), indent=2, sort_keys=True), encoding='utf-8',
    )


def load_paired_initial_condition(path: Path) -> PairedInitialCondition:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload['metadata_json']))
        condition = PairedInitialCondition(
            qpos=np.asarray(payload['qpos'], dtype=float),
            qvel=np.asarray(payload['qvel'], dtype=float),
            desired_torso=np.asarray(payload['desired_torso'], dtype=float),
            desired_pelvis=np.asarray(payload['desired_pelvis'], dtype=float),
            com_reference=np.asarray(payload['com_reference'], dtype=float),
            joint_reference=np.asarray(payload['joint_reference'], dtype=float),
            seed=int(metadata['seed']),
            setup_policy=str(metadata['setup_policy']),
            setup_duration_s=float(metadata['setup_duration_s']),
        )
    if condition.sha256 != metadata['sha256']:
        raise ValueError(f'paired initial condition hash mismatch: {path}')
    return condition


def output_dirs(root: Path | None = None) -> dict[str, Path]:
    """Return an isolated result tree, optionally selected by the caller.

    ``SE3_RESULTS_ROOT`` is used by worker/staging runs so experiments never
    overwrite the checked-in or previously archived result set implicitly.
    """
    results_root = Path(os.environ.get("SE3_RESULTS_ROOT", str(root or ROOT / "results"))).expanduser()
    dirs = {
        "data": results_root / "data",
        "png": results_root / "figures" / "png",
        "pdf": results_root / "figures" / "pdf",
        "videos": results_root / "videos",
        "logs": results_root / "logs",
        "frames": results_root / "frames",
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
    if controller_name == 'pd_nominal_ff':
        controller_name = 'pd'
    if controller_name == 'pure_pd':
        c = configs['controller']
        return JointPDController(model, kp=c['posture_kp'], kd=c['posture_kd'])
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
    if controller_name in {"hybrid_wbc", "hybrid_se3_wbc"}:
        return HybridRecoveryController(
            model,
            c,
            configs["experiments"]["recovery"],
            configs["experiments"].get("hybrid_recovery", {}),
        )
    return WholeBodyQPController(model, c, configs["experiments"]["recovery"])


def prepare_paired_initial_condition(
    configs: dict,
    seed: int = 0,
    setup_duration_s: float = 0.6,
    mass_scale: float = 1.0,
    friction_coefficient: float | None = None,
) -> PairedInitialCondition:
    model = make_model(configs, mass_scale=mass_scale, friction_coefficient=friction_coefficient)
    controller = run_controller('pd_nominal_ff', model, configs)
    runner = SimulationRunner(
        model, controller, duration_s=0.0,
        control_timestep_s=float(configs['robot']['control_timestep']),
        warmup_duration_s=float(setup_duration_s), warmup_reanchor=True,
    )
    runner.run(seed=seed)
    return PairedInitialCondition(
        qpos=model.data.qpos.copy(), qvel=model.data.qvel.copy(),
        desired_torso=model.body_pose('torso').copy(),
        desired_pelvis=model.body_pose('pelvis').copy(),
        com_reference=model.center_of_mass().copy(),
        joint_reference=model.joint_positions().copy(), seed=int(seed),
        setup_policy='pd_nominal_ff', setup_duration_s=float(setup_duration_s),
    )


def run_trial(controller_name: str, configs: dict, push: Push | None = None, duration: float | None = None, mass_scale: float = 1.0, seed: int = 0, classify: bool = False, frame_callback=None, initial_qpos: np.ndarray | None = None, initial_qvel: np.ndarray | None = None, friction_coefficient: float | None = None, initial_condition: PairedInitialCondition | None = None):
    if initial_condition is not None and (initial_qpos is not None or initial_qvel is not None):
        raise ValueError('initial_condition cannot be combined with initial_qpos/initial_qvel')
    if initial_condition is not None:
        initial_qpos = initial_condition.qpos
        initial_qvel = initial_condition.qvel
        seed = initial_condition.seed
    model = make_model(configs, mass_scale=mass_scale, friction_coefficient=friction_coefficient)
    nominal_qpos = model.qpos0.copy()
    nominal_torso = model.body_pose("torso").copy()
    nominal_pelvis = model.body_pose("pelvis").copy()
    nominal_com = model.center_of_mass().copy()
    if initial_condition is not None:
        nominal_torso = initial_condition.desired_torso
        nominal_pelvis = initial_condition.desired_pelvis
        nominal_com = initial_condition.com_reference
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
        joint_reference=(initial_condition.joint_reference if initial_condition is not None else None),
        initial_condition_metadata=(initial_condition.metadata() if initial_condition is not None else None),
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
    extra_arrays = None
    if run.qpos_history:
        extra_arrays = {"qpos_history": np.asarray(run.qpos_history, dtype=float)}
        if getattr(run, "qvel_history", None):
            extra_arrays["qvel_history"] = np.asarray(run.qvel_history, dtype=float)
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
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
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
    model_path = metadata.get("model_path")
    if not model_path and isinstance(config, dict):
        model_path = config.get("robot", {}).get("model_path")
    model_sha256 = metadata.get("model_sha256")
    if model_path and not model_sha256:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            model_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    dependency_versions = {}
    for module_name, import_name in (("numpy", "numpy"), ("mujoco", "mujoco"), ("osqp", "osqp")):
        try:
            module = __import__(import_name)
            dependency_versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            dependency_versions[module_name] = "unavailable"
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = metadata.get("run_id") or os.environ.get("SE3_RUN_ID")
    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        git_dirty = None
    return {
        "run_id": run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "source_version": _source_version(),
        "git_dirty": git_dirty,
        "seed": int(seed),
        "hostname": socket.gethostname(),
        "timestamp_utc": timestamp,
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "python_version": sys.version.split()[0],
        "dependency_versions": dependency_versions,
        "command": metadata.get("command") or " ".join(sys.argv),
    }


def write_execution_manifest(path: Path, configs: dict, seed: int = 0, extra: dict | None = None) -> None:
    portable_config = _json_safe(configs)
    payload = {"config": portable_config, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(execution_manifest(payload, seed=seed) | {"config": portable_config, **(extra or {})}, indent=2, default=str), encoding="utf-8")


def flatten_result(run, controller: str, push: Push, trial_id: str, seed: int = 0, extra: dict | None = None) -> dict:
    rec = run.recovery
    a = run.log.arrays()
    qp_status_counts = Counter(str(status) for status in a['qp_status'])
    push_end_s = float(push.start_time_s + push.duration_s)
    eval_end_s = push_end_s + 6.0
    if rec is None or rec.recovered_at_s is None:
        recovery_window_label = 'NOT_OBSERVED_IN_WINDOW'
    elif rec.recovered_at_s < push_end_s:
        recovery_window_label = 'BEFORE_WINDOW'
    elif rec.recovered_at_s <= eval_end_s:
        recovery_window_label = 'WITHIN_WINDOW'
    else:
        recovery_window_label = 'AFTER_WINDOW'
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
        "max_com_displacement_m": rec.max_com_displacement_m if rec else float(np.max(np.linalg.norm(a["com_world"][:, :2] - a["com_world"][0, :2], axis=1))),
        "max_com_3d_displacement_m": float(np.max(np.linalg.norm(a["com_world"] - a["com_world"][0], axis=1))),
        "max_joint_torque_Nm": rec.max_joint_torque_Nm if rec else float(np.max(a["torque_abs_max_Nm"])),
        "max_contact_force_N": (
            float(np.max(np.linalg.norm(
                a.get("actual_contact_wrench_post_step", a["actual_contact_wrench"])[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3),
                axis=2,
            )))
            if len(a.get("actual_contact_wrench_post_step", a["actual_contact_wrench"])) else 0.0
        ),
        "min_friction_margin": rec.min_friction_margin if rec else float(np.nanmin(a["actual_friction_margin"])),
        "seed": seed,
        "initial_condition_sha256": run.metadata.get("initial_condition_sha256", ""),
        "planned_push_duration_s": run.metadata.get("planned_push_duration_s", push.duration_s),
        "realized_push_duration_s": run.metadata.get("realized_push_duration_s", ""),
        "planned_impulse_Ns": run.metadata.get("planned_impulse_Ns", push.impulse_Ns),
        "realized_impulse_Ns": run.metadata.get("realized_impulse_Ns", ""),
        "active_push_substeps": run.metadata.get("active_push_substeps", ""),
        "force_trace_sha256": run.metadata.get("force_trace_sha256", ""),
    }
    mass_kg = float(run.metadata.get("mass_kg", float("nan")))
    force_over_mg, impulse_over_mass = push.normalized(mass_kg) if np.isfinite(mass_kg) else (float("nan"), float("nan"))
    row.update({
        "mass_kg": mass_kg,
        "force_over_mg": force_over_mg,
        "impulse_over_mass_m_s": impulse_over_mass,
    })
    row.update({
        'initial_qpos_sha256': run.metadata.get('initial_condition', {}).get('qpos_sha256', ''),
        'initial_qvel_sha256': run.metadata.get('initial_condition', {}).get('qvel_sha256', ''),
        'torso_reference_sha256': run.metadata.get('initial_condition', {}).get('torso_reference_sha256', ''),
        'pelvis_reference_sha256': run.metadata.get('initial_condition', {}).get('pelvis_reference_sha256', ''),
        'com_reference_sha256': run.metadata.get('initial_condition', {}).get('com_reference_sha256', ''),
        'joint_reference_sha256': run.metadata.get('initial_condition', {}).get('joint_reference_sha256', ''),
        'evaluation_window_start_s': push_end_s,
        'evaluation_window_end_s': eval_end_s,
        'recovery_window_label': recovery_window_label,
        'qp_status_counts_json': json.dumps(dict(sorted(qp_status_counts.items())), sort_keys=True),
        'fallback_count': int(qp_status_counts.get('fallback_pd', 0)),
        'qp_failure_count': int(np.count_nonzero(~np.asarray(a['qp_success'], dtype=bool))),
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
