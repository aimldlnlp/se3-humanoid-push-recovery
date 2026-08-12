"""MuJoCo model loading and explicit state/kinematics extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class ActualContactData:
    """Physical ground reactions extracted from MuJoCo contacts.

    Wrenches are ordered per foot as ``[Fx, Fy, Fz, Mx, My, Mz]`` in world
    coordinates, with moments about the corresponding foot body COM.  These
    values are measured from MuJoCo's contact solver and are intentionally
    separate from any wrench predicted by the controller QP.
    """

    wrench_world: np.ndarray
    contact_flags: np.ndarray
    tangent_velocity_m_s: np.ndarray
    xy_points_m: np.ndarray
    friction_utilization: np.ndarray


def _transform(position: np.ndarray, rotation_flat: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation_flat, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


class HumanoidModel:
    """Thin, testable wrapper around ``mujoco.MjModel`` and ``MjData``."""

    def __init__(self, model_path: str | Path, mass_scale: float = 1.0, friction_coefficient: float | None = None):
        if mujoco is None:
            raise ImportError("MuJoCo is required for HumanoidModel")
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if mass_scale != 1.0:
            self.model.body_mass[:] *= float(mass_scale)
            self.model.body_inertia[:] *= float(mass_scale)
        if friction_coefficient is not None:
            self.model.geom_friction[:, 0] = float(friction_coefficient)
        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["pelvis", "torso", "left_foot", "right_foot"]
        }
        self.geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["ground", "left_foot_geom", "right_foot_geom"]
        }
        if any(v < 0 for v in self.body_ids.values()):
            raise ValueError(f"required body missing: {self.body_ids}")
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
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
        self.qpos0 = self._standing_qpos()
        self.reset()

    def _standing_qpos(self) -> np.ndarray:
        q = np.zeros(self.model.nq, dtype=float)
        q[:] = self.model.qpos0
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if free:
            adr = self.model.jnt_qposadr[free[0]]
            q[adr : adr + 7] = np.array([0.0, 0.0, 1.04999, 1.0, 0.0, 0.0, 0.0])
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
        if np.all(self.model.actuator_ctrllimited):
            return self.model.actuator_ctrlrange.copy()
        return np.tile(np.array([-180.0, 180.0]), (self.model.nu, 1))

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

    def contact_flags(self) -> tuple[bool, bool]:
        left_id = self.geom_ids["left_foot_geom"]
        right_id = self.geom_ids["right_foot_geom"]
        ground_id = self.geom_ids["ground"]
        left = False
        right = False
        for i in range(self.data.ncon):
            a = int(self.data.contact[i].geom1)
            b = int(self.data.contact[i].geom2)
            pair = {a, b}
            if {left_id, ground_id}.issubset(pair):
                left = True
            if {right_id, ground_id}.issubset(pair):
                right = True
        return left, right

    def contact_jacobian(self) -> np.ndarray:
        return np.vstack([self.body_jacobian("left_foot"), self.body_jacobian("right_foot")])

    def actual_contact_data(self) -> ActualContactData:
        """Extract actual MuJoCo ground-reaction force and slip quantities."""
        wrench = np.zeros(12, dtype=float)
        flags = np.zeros(2, dtype=bool)
        tangent_velocity = np.zeros(2, dtype=float)
        points = np.full((2, 3), np.nan, dtype=float)
        normal_force = np.zeros(2, dtype=float)
        tangent_force = np.zeros(2, dtype=float)
        foot_geom_ids = [self.geom_ids["left_foot_geom"], self.geom_ids["right_foot_geom"]]
        foot_names = ["left_foot", "right_foot"]
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            foot_index = None
            for i, geom_id in enumerate(foot_geom_ids):
                if geom_id in (geom1, geom2) and self.geom_ids["ground"] in (geom1, geom2):
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
            moment_world += np.cross(np.asarray(contact.pos) - self.data.xipos[body_id], force_world)
            offset = 6 * foot_index
            wrench[offset : offset + 3] += force_world
            wrench[offset + 3 : offset + 6] += moment_world
            points[foot_index] = np.asarray(contact.pos)
            jacp = np.zeros((3, self.nv))
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, contact.pos, body_id)
            velocity = jacp @ self.data.qvel
            tangent_velocity[foot_index] = max(tangent_velocity[foot_index], float(np.linalg.norm(velocity[:2])))
            normal_force[foot_index] += max(0.0, float(force_world[2]))
            tangent_force[foot_index] += float(np.linalg.norm(force_world[:2]))
        mu = np.maximum(self.model.geom_friction[foot_geom_ids, 0], 1e-9)
        utilization = np.divide(tangent_force, mu * normal_force, out=np.zeros(2), where=normal_force > 1e-9)
        return ActualContactData(wrench, flags, tangent_velocity, points, utilization)

    def contact_bias_acceleration(self, finite_difference_dt: float = 1e-6) -> np.ndarray:
        """Return ``Jdot(q, qdot) qdot`` for both foot Jacobians.

        MuJoCo exposes the body Jacobian directly but not this bias term. A
        directional finite difference along the current generalized velocity
        is frame-consistent and keeps the contact constraint explicit.
        """
        qpos_saved = self.data.qpos.copy()
        qvel_saved = self.data.qvel.copy()
        J0 = self.contact_jacobian()
        qpos_next = qpos_saved.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel_saved, float(finite_difference_dt))
        self.data.qpos[:] = qpos_next
        mujoco.mj_forward(self.model, self.data)
        J1 = self.contact_jacobian()
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
