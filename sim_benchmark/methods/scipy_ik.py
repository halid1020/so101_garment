"""Numerical position-first IK via scipy least-squares (telegrip style).

telegrip and several hobbyist SO-10x teleop stacks skip differential IK
entirely: each tick they run a small bounded nonlinear least-squares solve
for the absolute joint configuration that best reaches the target position,
with a weak orientation regularizer. Warm-started from the previous
solution it converges in a few iterations, and unlike velocity-integration
methods it cannot accumulate drift — at the cost of a slower, less smooth
solve.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from sim_benchmark.constants import (
    ARM_JOINT_SUFFIXES,
    ARM_JOINTS,
    DUAL_URDF_PATH,
    EE_FRAMES,
    SIDES,
)
from sim_benchmark.methods.base import Targets, TeleopMethod

ORI_WEIGHT = 0.1


class ScipyLeastSquares(TeleopMethod):
    """Per-arm bounded least-squares pose IK, warm-started each tick."""

    name = "scipy_ls"

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        super().__init__(sim_model)
        full = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
        gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
        self.model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))
        self.data = self.model.createData()

        self._frame_id = {
            side: self.model.getFrameId(EE_FRAMES[side]) for side in SIDES
        }
        self._q_idx = {
            side: np.array(
                [
                    self.model.joints[self.model.getJointId(f"{side}_{sfx}")].idx_q
                    for sfx in ARM_JOINT_SUFFIXES
                ]
            )
            for side in SIDES
        }
        self._arm_idx = np.array(
            [self.model.joints[self.model.getJointId(j)].idx_q for j in ARM_JOINTS]
        )
        self.q_full = pin.neutral(self.model)

    def _residual(
        self, q_arm: np.ndarray, side: str, pos_t: np.ndarray, rot_t: np.ndarray
    ) -> np.ndarray:
        q = self.q_full.copy()
        q[self._q_idx[side]] = q_arm
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self._frame_id[side]]
        pos_err = oMf.translation - pos_t
        rot_err = pin.log3(rot_t.T @ oMf.rotation)
        return np.concatenate([pos_err, ORI_WEIGHT * rot_err])

    def reset(self, q0: np.ndarray) -> None:
        self.q_full = pin.neutral(self.model)
        self.q_full[self._arm_idx] = q0

    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        for side in SIDES:
            pos_t, rot_t = targets[side]
            idx = self._q_idx[side]
            result = least_squares(
                self._residual,
                self.q_full[idx],
                args=(side, pos_t, rot_t),
                bounds=(
                    self.model.lowerPositionLimit[idx],
                    self.model.upperPositionLimit[idx],
                ),
                method="trf",
                max_nfev=25,
                ftol=1e-6,
                xtol=1e-6,
            )
            self.q_full[idx] = result.x
        return self.q_full[self._arm_idx].copy()
