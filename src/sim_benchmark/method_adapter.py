"""PinkIKSolver-compatible facade over the benchmark teleop methods.

The armplane pipeline (the production teleop thread,
`common.threads.dual_ik_solver`) talks to its
IK solver through a narrow interface: anchor to measured joints, report EE
poses, accept per-frame pose targets, solve, expose the configuration. This
adapter implements exactly that interface on top of any registered
benchmark method (`sim_benchmark.methods.METHODS`), so all five candidate
IK strategies can be driven by the real Meta Quest device — on the real
arms (tool/meta_quest_teleopration.py --method ...) and in the MuJoCo sim
(tool/quest_sim_teleop.py) — without touching the armplane pipeline
(the production teleop thread).

A joint-space rate limiter (per benchmark finding: mandatory for scipy_ls
and mink near the workspace edge) clamps every solve.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin

_repo_root = Path(__file__).resolve().parent.parent.parent
for _p in (str(_repo_root), str(_repo_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim_benchmark.constants import DUAL_URDF_PATH, EE_FRAMES, SIDES  # noqa: E402
from sim_benchmark.methods import METHODS  # noqa: E402
from sim_benchmark.scene import DualArmSim  # noqa: E402

_FRAME_TO_SIDE = {frame: side for side, frame in EE_FRAMES.items()}


class MethodIKAdapter:
    """Drives a benchmark TeleopMethod behind the PinkIKSolver interface."""

    def __init__(
        self,
        method_name: str,
        dt: float = 0.01,
        max_joint_vel: float = 3.0,
        initial_configuration: np.ndarray | None = None,
    ) -> None:
        if method_name not in METHODS:
            raise ValueError(
                f"Unknown method {method_name!r} (choose from {sorted(METHODS)})"
            )
        # Kinematic-only sim instance: provides the MuJoCo model required by
        # the dls/mink methods and FK for EE poses. Never stepped.
        self._kin = DualArmSim()
        self.method = METHODS[method_name](self._kin.model)
        self.method_name = method_name
        self.dt = dt
        self.max_joint_vel = max_joint_vel

        # The armplane pipeline reads .urdf_model (Pinocchio) for arm base
        # positions; build the same reduced model PinkIKSolver uses.
        full = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
        gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
        self.urdf_model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))

        self._q = (
            np.asarray(initial_configuration, dtype=float).copy()
            if initial_configuration is not None
            else self._kin.neutral_q()
        )
        self.method.reset(self._q)
        self._targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_solve_time = 0.0

    # ------------------------------------------------------------------
    # PinkIKSolver interface used by common.threads.dual_ik_solver
    # ------------------------------------------------------------------

    def set_configuration_no_task_update(self, joint_config: np.ndarray) -> None:
        """Anchor the method state to measured joints (radians, 10-dof)."""
        self._q = np.asarray(joint_config, dtype=float).copy()
        self.method.reset(self._q)

    def set_configuration(self, joint_config: np.ndarray) -> None:
        self.set_configuration_no_task_update(joint_config)

    def get_current_configuration(self) -> np.ndarray:
        """Current commanded configuration (radians, 10-dof)."""
        return self._q.copy()

    def get_current_end_effector_poses(self) -> dict[str, np.ndarray]:
        """4x4 EE poses at the current configuration, keyed by URDF frame."""
        poses = self._kin.fk_eef_pose(self._q)
        return {EE_FRAMES[side]: poses[side] for side in SIDES}

    def set_target_poses(
        self, targets: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for frame, (position, rotation) in targets.items():
            side = _FRAME_TO_SIDE.get(frame)
            if side is None:
                raise ValueError(f"Unknown end-effector frame {frame!r}")
            self._targets[side] = (
                np.asarray(position, dtype=float).copy(),
                np.asarray(rotation, dtype=float).copy(),
            )

    def solve_ik(self, dt: float | None = None) -> bool:
        if len(self._targets) < len(SIDES):
            return False
        step_dt = dt if dt is not None else self.dt
        start = time.perf_counter()
        q_new = self.method.solve(self._targets, step_dt)
        # Safety: joint-space rate limit (benchmark: scipy_ls/mink can jump
        # between IK solutions near the workspace edge).
        max_step = self.max_joint_vel * step_dt
        self._q = self._q + np.clip(q_new - self._q, -max_step, max_step)
        self.last_solve_time = (time.perf_counter() - start) * 1e3
        return True

    def get_statistics(self) -> dict[str, float]:
        return {"last_solve_time_ms": self.last_solve_time}

    def update_task_parameters(self, **_kwargs: Any) -> None:
        """No-op: benchmark methods have fixed parameters."""

    def reset_to_neutral(self) -> None:
        self.set_configuration(self._kin.neutral_q())
