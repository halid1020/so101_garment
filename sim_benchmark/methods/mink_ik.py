"""mink (MuJoCo-native QP IK) teleop method.

mink is the IK library used by several LeRobot/SO-101 community teleop
stacks; it solves the same weighted-task QP as Pink but directly on the
MuJoCo model, avoiding a second (Pinocchio) kinematic description of the
robot. Benchmarking it against pink_full quantifies what, if anything, the
URDF/MJCF model mismatch costs.
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from sim_benchmark.constants import ARM_JOINTS, EE_FRAMES, SIDES
from sim_benchmark.methods.base import Targets, TeleopMethod


class MinkQP(TeleopMethod):
    """Dual-arm weighted-task QP IK on the MuJoCo model via mink."""

    name = "mink"

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        super().__init__(sim_model)
        self.model = sim_model
        self.configuration = mink.Configuration(sim_model)

        self.tasks: dict[str, mink.FrameTask] = {
            side: mink.FrameTask(
                frame_name=EE_FRAMES[side],
                frame_type="body",
                position_cost=1.0,
                orientation_cost=0.75,
                gain=0.4,
                lm_damping=1e-3,
            )
            for side in SIDES
        }
        self.posture_task = mink.PostureTask(sim_model, cost=1e-3)
        # Velocity cap matching the DLS method's clamp; without it mink
        # commands unbounded joint speeds near workspace-edge singularities.
        velocities = {
            sim_model.joint(j).name: 3.0
            for j in range(sim_model.njnt)
            if sim_model.joint(j).type == mujoco.mjtJoint.mjJNT_HINGE
        }
        self.limits = [
            mink.ConfigurationLimit(sim_model),
            mink.VelocityLimit(sim_model, velocities),
        ]

        self._qpos_idx = np.array([sim_model.joint(j).qposadr[0] for j in ARM_JOINTS])

    def reset(self, q0: np.ndarray) -> None:
        q = np.zeros(self.model.nq)
        q[self._qpos_idx] = q0
        self.configuration.update(q)
        self.posture_task.set_target(q)

    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        for side, (pos, rot) in targets.items():
            self.tasks[side].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos)
            )
        vel = mink.solve_ik(
            self.configuration,
            [*self.tasks.values(), self.posture_task],
            dt,
            solver="quadprog",
            damping=1e-12,
            limits=self.limits,
        )
        self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q[self._qpos_idx].copy()
