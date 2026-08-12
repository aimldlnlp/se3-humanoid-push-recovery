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
    """Return a world/spatial SE(3) task in ``[linear, angular]`` order.

    Convention used throughout the controller:

    * ``T_world_body`` maps body coordinates into the world frame;
    * the error is the *left/spatial* error
      ``E = T_world_current @ inv(T_world_desired)``;
    * ``Log(E)^vee`` is therefore expressed in the world tangent frame;
    * MuJoCo's body Jacobian is also interpreted as a world/spatial twist
      Jacobian, ``V_world = J_world @ qdot``.

    The proportional-derivative acceleration target is a resolved
    acceleration approximation in this common tangent frame.  Using the
    right/body error here would require an adjoint transformation of both the
    error and velocity, so it is deliberately not mixed with the world-frame
    Jacobian.
    """

    error = log_se3(current_T @ inverse_se3(desired_T))
    velocity_world = jacobian_world @ qvel
    kp = np.r_[np.full(3, kp_position), np.full(3, kp_rotation)]
    kd = np.r_[np.full(3, kd_position), np.full(3, kd_rotation)]
    return jacobian_world, -kp * error - kd * velocity_world, error


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
