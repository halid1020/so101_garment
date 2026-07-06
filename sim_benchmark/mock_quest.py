"""Mock Meta-Quest hand trajectories for the teleop benchmark.

Mimics the real pipeline's clutch behavior: at grip-press the controller
pose and the robot EE pose are latched, and subsequent controller *deltas*
are applied to the latched EE pose. Here the mock emits those deltas
directly, so a trajectory is defined purely by its delta curve
``delta(t) -> (dpos(3), drot(3,3))`` applied to each arm's initial EE pose.

Two families are provided, per the pre-hardware verification plan:
- circles of different radii drawn in the horizontal (table) plane;
- straight lines on the table along different compass directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim_benchmark.constants import SIDES

DeltaFn = Callable[[float], np.ndarray]  # t -> dpos(3), orientation held


@dataclass(frozen=True)
class MockTrajectory:
    """A named hand-motion pattern with a duration in seconds."""

    name: str
    duration: float
    delta_fn: DeltaFn
    # Mirror the y-axis of the left hand's delta so hands move symmetrically,
    # as a human naturally would when both arms draw the same figure.
    mirror_left_y: bool = False

    def targets(
        self,
        t: float,
        initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Absolute EE targets per side at time t (orientation held fixed)."""
        dpos = self.delta_fn(t)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in SIDES:
            p0, r0 = initial_poses[side]
            d = dpos.copy()
            if self.mirror_left_y and side == "left":
                d[1] = -d[1]
            out[side] = (p0 + d, r0.copy())
        return out


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    """C1 ramp used to ease trajectories in from zero velocity."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def circle(radius: float, period: float = 6.0, cycles: int = 2) -> MockTrajectory:
    """Circle of given radius (m) in the horizontal plane at fixed height.

    Starts and ends at the latched pose, with a smooth ramp-in so the mock
    hand does not teleport.
    """
    duration = period * cycles

    def delta(t: float) -> np.ndarray:
        ramp = _smoothstep(t / (0.25 * period))
        phase = 2 * np.pi * t / period
        # Circle through the start point: offset center by -radius in x.
        return ramp * np.array(
            [radius * (np.cos(phase) - 1.0), radius * np.sin(phase), 0.0]
        )

    return MockTrajectory(
        name=f"circle_r{int(radius * 100)}cm",
        duration=duration,
        delta_fn=delta,
        mirror_left_y=True,
    )


def line(
    direction_deg: float,
    length: float = 0.12,
    period: float = 4.0,
    cycles: int = 2,
) -> MockTrajectory:
    """Back-and-forth line stroke on the table plane.

    direction_deg is the compass direction in the horizontal plane:
    0 = +x (away from the robot bases), 90 = +y (to the robots' left).
    """
    theta = np.deg2rad(direction_deg)
    axis = np.array([np.cos(theta), np.sin(theta), 0.0])
    duration = period * cycles

    def delta(t: float) -> np.ndarray:
        # Smooth 0 -> length -> 0 stroke each period (cosine profile).
        s = 0.5 * (1 - np.cos(2 * np.pi * t / period))
        ramp = _smoothstep(t / (0.25 * period))
        return ramp * s * length * axis

    return MockTrajectory(
        name=f"line_{int(direction_deg)}deg",
        duration=duration,
        delta_fn=delta,
        mirror_left_y=True,
    )


def default_suite() -> list[MockTrajectory]:
    """The benchmark suite: 3 circle radii + 4 line directions."""
    return [
        circle(radius=0.03),
        circle(radius=0.05),
        circle(radius=0.08),
        line(direction_deg=0),
        line(direction_deg=45),
        line(direction_deg=90),
        line(direction_deg=135),
    ]
