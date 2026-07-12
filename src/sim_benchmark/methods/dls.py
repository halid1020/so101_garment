"""Damped-least-squares differential IK on the MuJoCo Jacobian.

The strategy used by Dream-Machines' vr-teleop-kit: per control tick,
compute the 6D task-space error at the end-effector, solve
``dq = J^T (J J^T + lambda^2 I)^-1 e`` per arm, and integrate. Cheap,
dependency-free (no QP solver), and runs directly on the simulator's
kinematic model, so it also cross-checks the Pinocchio-based methods
against MuJoCo's kinematics.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim_benchmark.constants import ARM_JOINT_SUFFIXES, ARM_JOINTS, SIDES
from sim_benchmark.methods.base import Targets, TeleopMethod


def _rot_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Axis-angle rotation error (world frame) via the matrix log."""
    r_err = r_target @ r_current.T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, r_err.reshape(9))
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


class DampedLeastSquares(TeleopMethod):
    """Per-arm 6D DLS differential IK (vr-teleop-kit style)."""

    name = "dls"

    # Task weights: position dominates; orientation softly tracked, matching
    # the spirit of the production cost ratio.
    ORI_WEIGHT = 0.5
    DAMPING = 0.05
    GAIN = 0.4  # error feedback gain per tick, like the Pink frame-task gain
    MAX_JOINT_VEL = 3.0  # rad/s clamp

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        super().__init__(sim_model)
        self.model = sim_model
        self.data = mujoco.MjData(sim_model)  # private kinematic scratchpad
        self.q = np.zeros(len(ARM_JOINTS))

        self._qpos_idx = np.array([sim_model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self._dof_idx = {
            side: np.array(
                [
                    sim_model.joint(f"{side}_{sfx}").dofadr[0]
                    for sfx in ARM_JOINT_SUFFIXES
                ]
            )
            for side in SIDES
        }
        self._site_id = {side: sim_model.site(f"{side}_eef_site").id for side in SIDES}
        self._q_low = np.array([sim_model.joint(j).range[0] for j in ARM_JOINTS])
        self._q_high = np.array([sim_model.joint(j).range[1] for j in ARM_JOINTS])

    def reset(self, q0: np.ndarray) -> None:
        self.q = q0.copy()

    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        # Forward kinematics at the current *commanded* configuration.
        self.data.qpos[self._qpos_idx] = self.q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        dq_parts = []

        for side in SIDES:
            pos_t, rot_t = targets[side]
            sid = self._site_id[side]
            pos_c = self.data.site_xpos[sid]
            rot_c = self.data.site_xmat[sid].reshape(3, 3)

            err = np.concatenate(
                [pos_t - pos_c, self.ORI_WEIGHT * _rot_error(rot_t, rot_c)]
            )

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            cols = self._dof_idx[side]
            jac = np.vstack([jacp[:, cols], self.ORI_WEIGHT * jacr[:, cols]])

            jjt = jac @ jac.T + self.DAMPING**2 * np.eye(6)
            dq_arm = jac.T @ np.linalg.solve(jjt, err)

            vel = np.clip(
                dq_arm * self.GAIN / dt, -self.MAX_JOINT_VEL, self.MAX_JOINT_VEL
            )
            dq_parts.append(vel * dt)

        self.q = np.clip(self.q + np.concatenate(dq_parts), self._q_low, self._q_high)
        return self.q.copy()
