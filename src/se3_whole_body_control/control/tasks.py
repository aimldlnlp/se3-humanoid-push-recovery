"""Task-space acceleration targets used by the whole-body controller."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from se3_whole_body_control.geometry.se3 import adjoint_se3, inverse_se3, log_se3


def quintic_reference(start, goal, time_s: float, duration_s: float):
    """Rest-to-rest Cartesian trajectory, holding each endpoint outside [0, D]."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.shape != (3,) or goal.shape != (3,) or not np.all(np.isfinite([start, goal])):
        raise ValueError("trajectory endpoints must be finite 3-vectors")
    if not np.isfinite(time_s) or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("trajectory duration must be positive and time finite")
    u = np.clip(time_s / duration_s, 0, 1)
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / duration_s
    dds = (60*u - 180*u**2 + 120*u**3) / duration_s**2
    delta = goal - start
    return start + s*delta, ds*delta, dds*delta


@dataclass
class ReachTask:
    """Optional world-frame position task; orientation remains unconstrained."""
    body_name: str
    point_local: np.ndarray
    position_world: np.ndarray
    velocity_world: np.ndarray
    acceleration_world: np.ndarray
    kp: float = 150.0
    kd: float = 25.0
    weight: float = 100.0

    def acceleration_target(self, model):
        vectors = (self.position_world, self.velocity_world, self.acceleration_world)
        if any(np.asarray(v).shape != (3,) or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError("reach references must be finite world 3-vectors")
        if any(not np.isfinite(v) or v < 0 for v in (self.kp, self.kd, self.weight)):
            raise ValueError("reach gains and weight must be finite and nonnegative")
        position, J, bias = model.attached_point_kinematics(self.body_name, self.point_local)
        target = (np.asarray(self.acceleration_world) + self.kp*(self.position_world-position)
                  + self.kd*(self.velocity_world-J @ model.data.qvel) - bias)
        return J, target, position


def pose_task_acceleration(
    current_T: np.ndarray,
    desired_T: np.ndarray,
    jacobian_world: np.ndarray,
    qvel: np.ndarray,
    kp_position: float,
    kd_position: float,
    kp_rotation: float,
    kd_rotation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a world/spatial SE(3) task in ``[linear, angular]`` order.

    Convention used throughout the controller:

    * ``T_world_body`` maps body coordinates into the world frame;
    * the error is the *right-invariant spatial* error
      ``E = T_world_current @ inv(T_world_desired)``;
    * ``Log(E)^vee`` is therefore expressed in the world tangent frame;
    * MuJoCo's body Jacobian is also interpreted as a world/spatial twist
      Jacobian, ``V_world = J_world @ qdot``.

    The proportional and derivative gains are specified in the desired-body
    tangent frame and transported to the world tangent frame with the desired
    pose adjoint.  This makes the complete task output equivariant under an
    arbitrary constant change of world frame, including a change of origin.
    The returned target is a resolved-acceleration approximation, not a claim
    of globally exact nonlinear SE(3) tracking.
    """

    error = log_se3(current_T @ inverse_se3(desired_T))
    velocity_world = jacobian_world @ qvel
    gain_adjoint = adjoint_se3(desired_T)
    gain_adjoint_inverse = inverse_se3(desired_T)
    kp_body = np.diag(np.r_[np.full(3, kp_position), np.full(3, kp_rotation)])
    kd_body = np.diag(np.r_[np.full(3, kd_position), np.full(3, kd_rotation)])
    kp_world = gain_adjoint @ kp_body @ adjoint_se3(gain_adjoint_inverse)
    kd_world = gain_adjoint @ kd_body @ adjoint_se3(gain_adjoint_inverse)
    return jacobian_world, -kp_world @ error - kd_world @ velocity_world, error


def posture_task(
    joint_q: np.ndarray,
    joint_q_des: np.ndarray,
    joint_qdot: np.ndarray,
    joint_dof_indices: np.ndarray,
    nv: int,
    kp: float,
    kd: float,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((len(joint_dof_indices), nv), dtype=float)
    A[np.arange(len(joint_dof_indices)), joint_dof_indices] = 1.0
    b = kp * (joint_q_des - joint_q) - kd * joint_qdot
    return A, b


def com_jacobian(model) -> np.ndarray:
    """Mass-weighted world linear Jacobian of the actual MuJoCo body COMs."""

    masses = model.model.body_mass
    total = max(float(np.sum(masses)), 1e-12)
    J = np.zeros((3, model.nv), dtype=float)
    for body_id in range(1, model.model.nbody):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        import mujoco

        # ``mj_jacBody`` is tied to the body reference point in the C API.
        # The CoM used by ``data.xipos`` is unambiguous, so use the explicit
        # COM routine whenever the installed MuJoCo binding provides it.
        jac_body_com = getattr(mujoco, "mj_jacBodyCom", None)
        if jac_body_com is not None:
            jac_body_com(model.model, model.data, jacp, jacr, body_id)
        else:  # compatibility fallback for older MuJoCo Python bindings
            mujoco.mj_jacBody(model.model, model.data, jacp, jacr, body_id)
        J += masses[body_id] * jacp
    return J / total
