"""Pink (Pinocchio QP) teleop methods.

Two variants of the differential-IK QP used by the production pipeline
(src/common/pink_ik_solver.py):

- ``pink_full``    — full 6D FrameTask with the production cost weights;
  what tool/meta_quest_teleopration.py runs on the real robot today.
- ``pink_relaxed`` — near position-only tracking (orientation cost lowered
  an order of magnitude), the "relaxed 5-DoF IK" strategy used by
  TeleopXR's SO-101 model. On a 5-DoF arm full 6D orientation tracking is
  over-constrained; relaxing it trades wrist attitude for position accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
for _p in (str(_repo_root), str(_repo_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.pink_ik_solver import PinkIKSolver  # noqa: E402
from sim_benchmark.constants import ARM_JOINTS, DUAL_URDF_PATH, EE_FRAMES  # noqa: E402
from sim_benchmark.methods.base import Targets, TeleopMethod  # noqa: E402


class _PinkBase(TeleopMethod):
    """Shared wrapper around the production PinkIKSolver."""

    position_cost: float
    orientation_cost: float

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        super().__init__(sim_model)
        self.solver = PinkIKSolver(
            urdf_path=str(DUAL_URDF_PATH),
            end_effector_frames=[EE_FRAMES[s] for s in ("left", "right")],
            solver_name="quadprog",
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            # Production values from src/common/configs.py.
            frame_task_gain=0.4,
            lm_damping=0.0,
            damping_cost=0.25,
            solver_damping_value=1e-12,
        )
        model = self.solver.urdf_model
        # Map ARM_JOINTS order -> Pinocchio q indices (grippers are locked
        # out of the reduced model, so nq == 10).
        self._pin_idx = np.array(
            [model.joints[model.getJointId(j)].idx_q for j in ARM_JOINTS]
        )

    def reset(self, q0: np.ndarray) -> None:
        q_pin = np.zeros(self.solver.urdf_model.nq)
        q_pin[self._pin_idx] = q0
        self.solver.set_configuration(q_pin)

    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        self.solver.set_target_poses(
            {EE_FRAMES[side]: (pos, rot) for side, (pos, rot) in targets.items()}
        )
        self.solver.solve_ik(dt)
        return self.solver.get_current_configuration()[self._pin_idx]


class PinkFull(_PinkBase):
    """Production QP IK: full 6D pose task (position 1.0 / orientation 0.75)."""

    name = "pink_full"
    position_cost = 1.0
    orientation_cost = 0.75


class PinkRelaxed(_PinkBase):
    """Relaxed-orientation QP IK (position 1.0 / orientation 0.05)."""

    name = "pink_relaxed"
    position_cost = 1.0
    orientation_cost = 0.05
