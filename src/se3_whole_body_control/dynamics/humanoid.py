"""MuJoCo model loading and explicit state/kinematics extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # Keep geometry-only tooling usable without MuJoCo installed.
    import mujoco
except ImportError:  # pragma: no cover - exercised on dependency-light hosts
    mujoco = None


@dataclass
class HumanoidState:
    qpos: np.ndarray
    qvel: np.ndarray
    com_world: np.ndarray
    T_world_torso: np.ndarray
    T_world_pelvis: np.ndarray
    T_world_left_foot: np.ndarray
    T_world_right_foot: np.ndarray
    torso_jacobian: np.ndarray
    pelvis_jacobian: np.ndarray
    left_foot_jacobian: np.ndarray
    right_foot_jacobian: np.ndarray
    contact_left: bool
    contact_right: bool


@dataclass(frozen=True)
class RobotAdapterConfig:
    """Name-based robot interface shared by the legacy and G1 models."""

    name: str
    floating_base_body: str
    floating_base_joint: str
    torso_body: str
    pelvis_body: str
    left_foot_body: str
    right_foot_body: str
    ground_geom: str
    left_foot_contact_geoms: tuple[str, ...]
    right_foot_contact_geoms: tuple[str, ...]
    nominal_base_qpos: np.ndarray
    nominal_joint_positions: np.ndarray
    foot_support_vertices_local: np.ndarray
    actuated_joint_names: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None, model_path: Path) -> "RobotAdapterConfig":
        """Build a model adapter from YAML, with a compatibility fallback.

        The fallback exists only for direct library users constructing the
        legacy model by path. Production experiments always pass the selected
        profile from ``configs/robots``.
        """
        raw = dict(mapping or {})
        is_mini = "mini_humanoid" in model_path.name
        defaults = {
            "name": "mini_humanoid" if is_mini else "humanoid",
            "floating_base_body": "pelvis",
            "floating_base_joint": "root" if is_mini else "floating_base_joint",
            "torso_body": "torso" if is_mini else "torso_link",
            "pelvis_body": "pelvis",
            "left_foot_body": "left_foot" if is_mini else "left_ankle_roll_link",
            "right_foot_body": "right_foot" if is_mini else "right_ankle_roll_link",
            "ground_geom": "ground",
            "left_foot_contact_geoms": ["left_foot_geom"] if is_mini else [],
            "right_foot_contact_geoms": ["right_foot_geom"] if is_mini else [],
            "nominal_base_qpos": [0.0, 0.0, 1.04999 if is_mini else 0.793, 1.0, 0.0, 0.0, 0.0],
            "nominal_joint_positions": [],
            "foot_support_vertices_local": [],
            "actuated_joint_names": [],
        }
        for key, value in defaults.items():
            raw.setdefault(key, value)
        vertices = np.asarray(raw["foot_support_vertices_local"], dtype=float)
        if vertices.size == 0:
            if is_mini:
                vertices = np.asarray([
                    [[-0.115, -0.12, -0.09], [0.225, -0.12, -0.09],
                     [0.225, 0.12, -0.09], [-0.115, 0.12, -0.09]],
                    [[-0.115, -0.12, -0.09], [0.225, -0.12, -0.09],
                     [0.225, 0.12, -0.09], [-0.115, 0.12, -0.09]],
                ], dtype=float)
            else:
                vertices = np.zeros((2, 0, 3), dtype=float)
        return cls(
            name=str(raw["name"]),
            floating_base_body=str(raw["floating_base_body"]),
            floating_base_joint=str(raw["floating_base_joint"]),
            torso_body=str(raw["torso_body"]),
            pelvis_body=str(raw["pelvis_body"]),
            left_foot_body=str(raw["left_foot_body"]),
            right_foot_body=str(raw["right_foot_body"]),
            ground_geom=str(raw["ground_geom"]),
            left_foot_contact_geoms=tuple(str(v) for v in raw["left_foot_contact_geoms"]),
            right_foot_contact_geoms=tuple(str(v) for v in raw["right_foot_contact_geoms"]),
            nominal_base_qpos=np.asarray(raw["nominal_base_qpos"], dtype=float),
            nominal_joint_positions=np.asarray(raw["nominal_joint_positions"], dtype=float),
            foot_support_vertices_local=vertices,
            actuated_joint_names=tuple(str(v) for v in raw["actuated_joint_names"]),
        )


@dataclass
class ActualContactData:
    """Physical ground reactions extracted from MuJoCo contacts.

    Wrenches are ordered per foot as ``[Fx, Fy, Fz, Mx, My, Mz]`` in world
    coordinates, with moments about the corresponding foot body-frame origin.  These
    values are measured from MuJoCo's contact solver and are intentionally
    separate from any wrench predicted by the controller QP.
    """

    wrench_world: np.ndarray
    contact_flags: np.ndarray
    tangent_velocity_m_s: np.ndarray
    xy_points_m: np.ndarray
    friction_utilization: np.ndarray
    cop_world: np.ndarray
    normal_force_N: np.ndarray


def _transform(position: np.ndarray, rotation_flat: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation_flat, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


class HumanoidModel:
    """Thin, testable wrapper around ``mujoco.MjModel`` and ``MjData``."""

    def __init__(
        self,
        model_path: str | Path,
        mass_scale: float = 1.0,
        friction_coefficient: float | None = None,
        robot_config: Mapping[str, Any] | None = None,
    ):
        if mujoco is None:
            raise ImportError("MuJoCo is required for HumanoidModel")
        self.model_path = Path(model_path)
        self.adapter = RobotAdapterConfig.from_mapping(robot_config, self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if mass_scale != 1.0:
            self.model.body_mass[:] *= float(mass_scale)
            self.model.body_inertia[:] *= float(mass_scale)
            # MuJoCo caches subtree masses, dof_M0, and other constants used by
            # forward dynamics.  Mutating body mass/inertia without refreshing
            # those derived fields makes robustness trials silently use a
            # hybrid of the nominal and perturbed models.
            mujoco.mj_setConst(self.model, self.data)
        if friction_coefficient is not None:
            self.model.geom_friction[:, 0] = float(friction_coefficient)
        body_names = {
            "floating_base": self.adapter.floating_base_body,
            "pelvis": self.adapter.pelvis_body,
            "torso": self.adapter.torso_body,
            "left_foot": self.adapter.left_foot_body,
            "right_foot": self.adapter.right_foot_body,
        }
        self.body_ids = {
            alias: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for alias, name in body_names.items()
        }
        if any(v < 0 for v in self.body_ids.values()):
            raise ValueError(f"required body missing: {self.body_ids}")
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.adapter.ground_geom)
        if ground_id < 0:
            # A scene wrapper may use ``floor`` while the adapter keeps the
            # semantic alias ``ground``.
            ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if ground_id < 0:
            raise ValueError(f"required ground geom missing: {self.adapter.ground_geom}")
        self.geom_ids = {"ground": int(ground_id)}
        self.foot_contact_geom_ids = (
            self._discover_contact_geoms(self.adapter.left_foot_body, self.adapter.left_foot_contact_geoms),
            self._discover_contact_geoms(self.adapter.right_foot_body, self.adapter.right_foot_contact_geoms),
        )
        self.geom_ids.update({
            "left_foot_geom": int(self.foot_contact_geom_ids[0][0]) if self.foot_contact_geom_ids[0] else -1,
            "right_foot_geom": int(self.foot_contact_geom_ids[1][0]) if self.foot_contact_geom_ids[1] else -1,
        })
        if not all(self.foot_contact_geom_ids):
            raise ValueError(f"no contact geoms found for both feet: {self.foot_contact_geom_ids}")
        actuator_joint_ids = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            actuator_joint_ids.append(joint_id)
        all_joint_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j): j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        }
        if self.adapter.actuated_joint_names:
            missing = [name for name in self.adapter.actuated_joint_names if name not in all_joint_names]
            if missing:
                raise ValueError(f"configured actuated joints missing from model: {missing}")
            self.joint_names = list(self.adapter.actuated_joint_names)
        else:
            self.joint_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                for joint_id in actuator_joint_ids
                if joint_id >= 0 and joint_id in all_joint_names.values()
            ]
        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        }
        self.joint_qpos_indices = np.array(
            [self.model.jnt_qposadr[j] for j in self.joint_ids.values()], dtype=int
        )
        self.joint_qvel_indices = np.array(
            [self.model.jnt_dofadr[j] for j in self.joint_ids.values()], dtype=int
        )
        self.actuator_matrix = self._build_actuator_matrix()
        self.actuator_limits = self._actuator_limits()
        self.support_vertices_local = np.asarray(self.adapter.foot_support_vertices_local, dtype=float)
        self.qpos0 = self._standing_qpos()
        self.reset()

    def _discover_contact_geoms(self, foot_body_name: str, explicit_names: tuple[str, ...]) -> tuple[int, ...]:
        """Return all physical contact geoms attached to a configured foot body."""
        if explicit_names:
            ids = tuple(
                int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
                for name in explicit_names
            )
            if any(geom_id < 0 for geom_id in ids):
                raise ValueError(f"configured foot contact geom missing: {explicit_names}")
            return ids
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_body_name))
        ids = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(self.model.geom_contype[geom_id]) == 0 or int(self.model.geom_conaffinity[geom_id]) == 0:
                continue
            ids.append(int(geom_id))
        return tuple(ids)

    def _standing_qpos(self) -> np.ndarray:
        q = np.zeros(self.model.nq, dtype=float)
        q[:] = self.model.qpos0
        free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.adapter.floating_base_joint,
        )
        if free_joint_id < 0:
            free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            free_joint_id = free[0] if free else -1
        if free_joint_id >= 0:
            adr = int(self.model.jnt_qposadr[free_joint_id])
            base = np.asarray(self.adapter.nominal_base_qpos, dtype=float).reshape(-1)
            if base.size != 7:
                raise ValueError("nominal_base_qpos must contain 7 values for a free joint")
            q[adr : adr + 7] = base
        nominal = np.asarray(self.adapter.nominal_joint_positions, dtype=float).reshape(-1)
        if nominal.size not in (0, len(self.joint_names)):
            raise ValueError(
                f"nominal_joint_positions has {nominal.size} values, expected {len(self.joint_names)}"
            )
        if nominal.size:
            for name, value in zip(self.joint_names, nominal):
                q[int(self.model.jnt_qposadr[self.joint_ids[name]])] = float(value)
        return q

    def _build_actuator_matrix(self) -> np.ndarray:
        B = np.zeros((self.model.nv, self.model.nu), dtype=float)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            dof = int(self.model.jnt_dofadr[joint_id])
            B[dof, actuator_id] = float(self.model.actuator_gear[actuator_id, 0])
        return B

    def _actuator_limits(self) -> np.ndarray:
        if self.model.nu == 0:
            return np.zeros((0, 2))
        limits = np.tile(np.array([-180.0, 180.0]), (self.model.nu, 1))
        for actuator_id in range(self.model.nu):
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                limits[actuator_id] = self.model.actuator_ctrlrange[actuator_id]
                continue
            if bool(getattr(self.model, "actuator_forcelimited", np.zeros(self.model.nu, dtype=bool))[actuator_id]):
                limits[actuator_id] = self.model.actuator_forcerange[actuator_id]
                continue
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_limited = getattr(self.model, "jnt_actfrclimited", None)
            joint_range = getattr(self.model, "jnt_actfrcrange", None)
            if joint_id >= 0 and joint_limited is not None and bool(joint_limited[joint_id]):
                limits[actuator_id] = joint_range[joint_id]
        return limits

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    def reset(self, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None) -> None:
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.qpos0 if qpos is None else np.asarray(qpos, dtype=float)
        self.data.qvel[:] = np.zeros(self.nv) if qvel is None else np.asarray(qvel, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray | None = None) -> None:
        if ctrl is not None:
            self.data.ctrl[:] = np.asarray(ctrl, dtype=float)
        mujoco.mj_step(self.model, self.data)

    def body_pose(self, body_name: str) -> np.ndarray:
        body_id = self.body_ids[body_name]
        return _transform(self.data.xpos[body_id], self.data.xmat[body_id])

    def body_jacobian(self, body_name: str) -> np.ndarray:
        body_id = self.body_ids[body_name]
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return np.vstack([jacp, jacr])

    def body_velocity(self, body_name: str) -> np.ndarray:
        return self.body_jacobian(body_name) @ self.data.qvel

    def mass_matrix(self) -> np.ndarray:
        M = np.zeros((self.nv, self.nv), dtype=float)
        # Recent MuJoCo Python bindings do not expose data.qM directly. The
        # public matrix-vector multiply API reconstructs the same dense matrix
        # without relying on private struct fields.
        for i in range(self.nv):
            basis = np.zeros(self.nv, dtype=float)
            basis[i] = 1.0
            column = np.zeros(self.nv, dtype=float)
            mujoco.mj_mulM(self.model, self.data, column, basis)
            M[:, i] = column
        return M

    def center_of_mass(self) -> np.ndarray:
        masses = self.model.body_mass
        total = float(np.sum(masses))
        return np.sum(self.data.xipos * masses[:, None], axis=0) / max(total, 1e-12)

    def point_jacobian(self, body_name: str, point_world: np.ndarray) -> np.ndarray:
        """World/spatial Jacobian at an arbitrary point on a named body."""
        body_id = self.body_ids[body_name]
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, np.asarray(point_world, dtype=float), body_id)
        return np.vstack([jacp, jacr])

    def contact_flags(self, min_normal_force_N: float = 0.0) -> tuple[bool, bool]:
        """Return physical foot contacts, optionally requiring load support.

        A geometric MuJoCo contact can be a grazing or numerically transient
        touch.  The default preserves the low-level geometric signal used by
        existing diagnostics; recovery-mode logic can request a positive
        normal-force threshold when it needs to establish load-bearing contact.
        """
        if float(min_normal_force_N) > 0.0:
            measured = self.actual_contact_data()
            loaded = measured.contact_flags & (measured.normal_force_N >= float(min_normal_force_N))
            return bool(loaded[0]), bool(loaded[1])
        ground_id = self.geom_ids["ground"]
        flags = [False, False]
        for i in range(self.data.ncon):
            a = int(self.data.contact[i].geom1)
            b = int(self.data.contact[i].geom2)
            if ground_id not in (a, b):
                continue
            foot_geom = b if a == ground_id else a
            for foot_index, geom_ids in enumerate(self.foot_contact_geom_ids):
                if foot_geom in geom_ids:
                    flags[foot_index] = True
        return bool(flags[0]), bool(flags[1])

    def contact_jacobian(self, foot_names: tuple[str, ...] | list[str] | None = None) -> np.ndarray:
        """Return stacked spatial Jacobians for the selected fixed contacts.

        The original controller always used both feet.  Keeping the selection
        here, at the robot-adapter boundary, lets higher-level contact-mode
        controllers change the active support set without scattering G1 body
        indices through the QP implementation.
        """
        names = tuple(foot_names or ("left_foot", "right_foot"))
        if not names:
            return np.zeros((0, self.nv), dtype=float)
        return np.vstack([self.body_jacobian(name) for name in names])

    def actual_contact_data(self) -> ActualContactData:
        """Extract actual MuJoCo ground-reaction force and slip quantities."""
        wrench = np.zeros(12, dtype=float)
        flags = np.zeros(2, dtype=bool)
        tangent_velocity = np.zeros(2, dtype=float)
        points = np.full((2, 3), np.nan, dtype=float)
        cop_world = np.full((2, 2), np.nan, dtype=float)
        normal_force = np.zeros(2, dtype=float)
        tangent_force = np.zeros(2, dtype=float)
        foot_geom_ids = self.foot_contact_geom_ids
        foot_names = ["left_foot", "right_foot"]
        point_weight = np.zeros(2, dtype=float)
        point_sum = np.zeros((2, 3), dtype=float)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            foot_index = None
            for i, geom_ids in enumerate(foot_geom_ids):
                if any(geom_id in (geom1, geom2) for geom_id in geom_ids) and self.geom_ids["ground"] in (geom1, geom2):
                    foot_index = i
                    break
            if foot_index is None:
                continue
            flags[foot_index] = True
            force_contact = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, contact_id, force_contact)
            frame_world = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_world = frame_world.T @ force_contact[:3]
            moment_world = frame_world.T @ force_contact[3:]
            # MuJoCo's contact-frame convention is oriented from the first
            # geom toward the second geom.  For the ground/foot pair this
            # makes the transformed normal point downward even though the
            # physical reaction on the foot is upward.  The ground plane is
            # horizontal in this model, so orient the measured reaction by
            # its world-z normal rather than by geom ordering.  Tangential
            # utilization is sign invariant; the wrench itself is now the
            # physically meaningful reaction on the foot.
            if force_world[2] < 0.0:
                force_world *= -1.0
                moment_world *= -1.0
            body_name = foot_names[foot_index]
            body_id = self.body_ids[body_name]
            # The production contact Jacobian is ``mj_jacBody``, whose
            # translational reference is the body-frame origin ``xpos``. Keep
            # the measured wrench at that same point so QP and MuJoCo signals
            # can be compared without an inertial-offset moment bias.
            moment_world += np.cross(np.asarray(contact.pos) - self.data.xpos[body_id], force_world)
            offset = 6 * foot_index
            wrench[offset : offset + 3] += force_world
            wrench[offset + 3 : offset + 6] += moment_world
            points[foot_index] = np.asarray(contact.pos)
            jacp = np.zeros((3, self.nv))
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, contact.pos, body_id)
            velocity = jacp @ self.data.qvel
            tangent_velocity[foot_index] = max(tangent_velocity[foot_index], float(np.linalg.norm(velocity[:2])))
            normal = max(0.0, float(force_world[2]))
            normal_force[foot_index] += normal
            tangent_force[foot_index] += float(np.linalg.norm(force_world[:2]))
            weight = max(normal, 1e-12)
            point_sum[foot_index] += weight * np.asarray(contact.pos)
            point_weight[foot_index] += weight
        for foot_index in range(2):
            if point_weight[foot_index] > 0:
                points[foot_index] = point_sum[foot_index] / point_weight[foot_index]
        mu_values = []
        for geom_ids in foot_geom_ids:
            values = [float(self.model.geom_friction[geom_id, 0]) for geom_id in geom_ids]
            mu_values.append(max(values) if values else 1.0)
        mu = np.maximum(np.asarray(mu_values, dtype=float), 1e-9)
        utilization = np.divide(tangent_force, mu * normal_force, out=np.zeros(2), where=normal_force > 1e-9)
        for foot_index, body_name in enumerate(foot_names):
            fz = wrench[6 * foot_index + 2]
            if flags[foot_index] and fz > 1e-9 and np.all(np.isfinite(wrench[6 * foot_index : 6 * foot_index + 6])):
                # The wrench is about the body-frame origin. For a horizontal
                # contact plane, Mx = y Fz and My = -x Fz.
                relative_xy = np.array([-wrench[6 * foot_index + 4] / fz, wrench[6 * foot_index + 3] / fz])
                cop_world[foot_index] = self.data.xpos[self.body_ids[body_name]][:2] + relative_xy
        return ActualContactData(wrench, flags, tangent_velocity, points, utilization, cop_world, normal_force)

    def foot_support_vertices_world(self) -> np.ndarray:
        """Return the four ground-facing vertices of each foot geom in world XY."""
        vertices = np.full((2, 4, 2), np.nan, dtype=float)
        if self.support_vertices_local.shape == (2, 4, 3):
            for foot_index, body_name in enumerate(("left_foot", "right_foot")):
                T = self.body_pose(body_name)
                local_h = np.c_[self.support_vertices_local[foot_index], np.ones(4)]
                vertices[foot_index] = (local_h @ T.T)[:, :2]
            return vertices
        # Compatibility fallback for an unprofiled box-foot model.
        for foot_index, geom_name in enumerate(("left_foot_geom", "right_foot_geom")):
            geom_id = self.geom_ids[geom_name]
            if geom_id < 0 or int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            center = np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            half_size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            local = np.array(
                [[-half_size[0], -half_size[1], -half_size[2]],
                 [ half_size[0], -half_size[1], -half_size[2]],
                 [ half_size[0],  half_size[1], -half_size[2]],
                 [-half_size[0],  half_size[1], -half_size[2]]],
                dtype=float,
            )
            vertices[foot_index] = (center + local @ rotation.T)[:, :2]
        return vertices

    def contact_bias_acceleration(
        self,
        finite_difference_dt: float = 1e-6,
        foot_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Return ``Jdot(q, qdot) qdot`` for the selected foot Jacobians.

        MuJoCo exposes the body Jacobian directly but not this bias term. A
        directional finite difference along the current generalized velocity
        is frame-consistent and keeps the contact constraint explicit.
        """
        qpos_saved = self.data.qpos.copy()
        qvel_saved = self.data.qvel.copy()
        names = tuple(foot_names or ("left_foot", "right_foot"))
        J0 = self.contact_jacobian(names)
        qpos_next = qpos_saved.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel_saved, float(finite_difference_dt))
        self.data.qpos[:] = qpos_next
        mujoco.mj_forward(self.model, self.data)
        J1 = self.contact_jacobian(names)
        self.data.qpos[:] = qpos_saved
        self.data.qvel[:] = qvel_saved
        mujoco.mj_forward(self.model, self.data)
        return ((J1 - J0) / float(finite_difference_dt)) @ qvel_saved

    def state(self) -> HumanoidState:
        left, right = self.contact_flags()
        return HumanoidState(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            com_world=self.center_of_mass(),
            T_world_torso=self.body_pose("torso"),
            T_world_pelvis=self.body_pose("pelvis"),
            T_world_left_foot=self.body_pose("left_foot"),
            T_world_right_foot=self.body_pose("right_foot"),
            torso_jacobian=self.body_jacobian("torso"),
            pelvis_jacobian=self.body_jacobian("pelvis"),
            left_foot_jacobian=self.body_jacobian("left_foot"),
            right_foot_jacobian=self.body_jacobian("right_foot"),
            contact_left=left,
            contact_right=right,
        )

    def set_external_force(self, body_name: str, force_world: np.ndarray, point_local: np.ndarray | None = None) -> None:
        body_id = self.body_ids[body_name]
        force = np.asarray(force_world, dtype=float).reshape(3)
        point = np.zeros(3) if point_local is None else np.asarray(point_local, dtype=float).reshape(3)
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = 0.0
        if np.linalg.norm(point) > 0:
            # xfrc_applied stores a world-frame wrench about the body COM.
            # Convert the user-specified body-local application offset before
            # forming the moment; the previous implementation treated the
            # local vector as if it were already world aligned.
            R_world_body = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
            point_world = R_world_body @ point
            self.data.xfrc_applied[body_id, 3:] = np.cross(point_world, force)

    def clear_external_force(self) -> None:
        self.data.xfrc_applied[:] = 0.0

    def external_generalized_force(self) -> np.ndarray:
        """Map applied world-frame body wrenches into generalized coordinates."""
        out = np.zeros(self.nv, dtype=float)
        for body_id in range(1, self.model.nbody):
            wrench = self.data.xfrc_applied[body_id]
            if np.linalg.norm(wrench) < 1e-14:
                continue
            jacp = np.zeros((3, self.nv))
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            out += jacp.T @ wrench[:3] + jacr.T @ wrench[3:]
        return out

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_indices].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.joint_qvel_indices].copy()

    def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.model.jnt_range[j] [0] for j in self.joint_ids.values()])
        hi = np.array([self.model.jnt_range[j] [1] for j in self.joint_ids.values()])
        return lo, hi

    def joint_position_limit_violation(self, margin_rad: float = 1e-5) -> bool:
        """Return whether an actuated joint is outside its configured range."""
        lo, hi = self.joint_position_limits()
        if not len(lo):
            return False
        q = self.joint_positions()
        limited = np.asarray([bool(self.model.jnt_limited[j]) for j in self.joint_ids.values()], dtype=bool)
        return bool(np.any(limited & ((q < lo - margin_rad) | (q > hi + margin_rad))))
