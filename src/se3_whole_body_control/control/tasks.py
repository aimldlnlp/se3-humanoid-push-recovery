"""Task-space acceleration targets used by the whole-body controller."""

from __future__ import annotations

import numpy as np

from se3_whole_body_control.geometry.se3 import inverse_se3, log_se3


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
    """Return ``(J, desired_acceleration, se3_error)`` in [linear, angular] order."""

    error = log_se3(inverse_se3(desired_T) @ current_T)
    velocity = jacobian_world @ qvel
    kp = np.r_[np.full(3, kp_position), np.full(3, kp_rotation)]
    kd = np.r_[np.full(3, kd_position), np.full(3, kd_rotation)]
    return jacobian_world, -kp * error - kd * velocity, error


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
    """Mass-weighted world linear Jacobian of the model CoM."""

    masses = model.model.body_mass
    total = max(float(np.sum(masses)), 1e-12)
    J = np.zeros((3, model.nv), dtype=float)
    for body_id in range(1, model.model.nbody):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        import mujoco

        mujoco.mj_jacBody(model.model, model.data, jacp, jacr, body_id)
        J += masses[body_id] * jacp
    return J / total
