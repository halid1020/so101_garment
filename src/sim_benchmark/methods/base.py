"""Common interface for benchmarked teleoperation IK methods.

Every method consumes absolute end-effector targets (robot base frame) and
produces arm joint commands in ARM_JOINTS order. The Quest-side clutch
mapping (controller delta -> EE target) happens upstream in the mock, so
methods are compared purely on target-following quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np

Targets = dict[str, tuple[np.ndarray, np.ndarray]]  # side -> (pos(3), rot(3,3))


class TeleopMethod(ABC):
    """A teleop IK pipeline mapping EE pose targets to joint commands."""

    #: Registry key and display name, set by subclasses.
    name: str = ""

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        """sim_model is the compiled benchmark scene (methods that solve on
        their own URDF-based model may ignore it)."""

    @abstractmethod
    def reset(self, q0: np.ndarray) -> None:
        """Re-initialize internal state at arm configuration q0 (radians,
        ARM_JOINTS order)."""

    @abstractmethod
    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        """Return the next joint command (radians, ARM_JOINTS order)."""
