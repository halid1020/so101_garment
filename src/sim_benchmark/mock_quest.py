"""Mock Meta-Quest hand trajectories for the teleop benchmark.

Mimics the real pipeline's clutch behavior: at grip-press the controller
pose and the robot EE pose are latched, and subsequent controller *deltas*
are applied to the latched EE pose. Here the mock emits those deltas
directly, so a trajectory is defined purely by its delta curve
``delta(t) -> (dpos(3), drot(3,3))`` applied to each arm's initial EE pose.

Four families are provided:
- circles of different radii drawn in the horizontal (table) plane;
- straight lines on the table along different compass directions;
- wrist-agility oscillations (position held, orientation oscillating) at
  several frequencies — quantifies how agile each method's wrist is;
- envelope excursions that deliberately drag the target outside the
  reachable workspace (``envelope_suite``), for the out-of-envelope
  policy benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim_benchmark.constants import SIDES

DeltaFn = Callable[[float], np.ndarray]  # t -> dpos(3)
RotFn = Callable[[float], np.ndarray]  # t -> drot(3,3) in the EE LOCAL frame

# Conjugating a local delta rotation with this reflection negates the roll
# component (rotation about local x/tip) while preserving flex (about local
# y): mirror-symmetric bimanual wrist motion for the left hand.
_MIRROR_Y = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class MockTrajectory:
    """A named hand-motion pattern with a duration in seconds."""

    name: str
    duration: float
    delta_fn: DeltaFn
    # Mirror the y-axis of the left hand's delta so hands move symmetrically,
    # as a human naturally would when both arms draw the same figure.
    mirror_left_y: bool = False
    # Optional orientation curve: target rotation = r0 @ rot_fn(t) (applied
    # in the latched EE's local frame). None keeps the orientation fixed.
    rot_fn: RotFn | None = None
    # Mirror the roll component of rot_fn for the left hand (bimanual
    # symmetry), keeping flex intact.
    mirror_left_roll: bool = False

    def targets(
        self,
        t: float,
        initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Absolute EE targets per side at time t."""
        dpos = self.delta_fn(t)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in SIDES:
            p0, r0 = initial_poses[side]
            d = dpos.copy()
            if self.mirror_left_y and side == "left":
                d[1] = -d[1]
            if self.rot_fn is not None:
                drot = self.rot_fn(t)
                if self.mirror_left_roll and side == "left":
                    drot = _MIRROR_Y @ drot @ _MIRROR_Y
                r = r0 @ drot
            else:
                r = r0.copy()
            out[side] = (p0 + d, r)
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


def _freq_tag(freq_hz: float) -> str:
    """0.5 -> 'f0p5hz', 1.0 -> 'f1hz', 2.0 -> 'f2hz'."""
    s = f"{freq_hz:g}".replace(".", "p")
    return f"f{s}hz"


def _rot_about(axis: int, angle: float) -> np.ndarray:
    """Rotation matrix about a local coordinate axis (0=x/roll, 1=y/flex)."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:  # about x (EE tip axis -> roll)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:  # about y (flex / tip elevation)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrist_roll_osc(
    freq_hz: float, amp_deg: float = 45.0, duration: float = 8.0
) -> MockTrajectory:
    """Position held; roll (about the EE tip axis) oscillates at freq_hz."""
    amp = np.deg2rad(amp_deg)

    def rot(t: float) -> np.ndarray:
        ramp = _smoothstep(t / 1.0)
        return _rot_about(0, ramp * amp * np.sin(2 * np.pi * freq_hz * t))

    return MockTrajectory(
        name=f"wrist_roll_{_freq_tag(freq_hz)}",
        duration=duration,
        delta_fn=lambda t: np.zeros(3),
        rot_fn=rot,
        mirror_left_roll=True,
    )


def wrist_flex_osc(
    freq_hz: float, amp_deg: float = 30.0, duration: float = 8.0
) -> MockTrajectory:
    """Position held; tip elevation (rotation about local y) oscillates."""
    amp = np.deg2rad(amp_deg)

    def rot(t: float) -> np.ndarray:
        ramp = _smoothstep(t / 1.0)
        return _rot_about(1, ramp * amp * np.sin(2 * np.pi * freq_hz * t))

    return MockTrajectory(
        name=f"wrist_flex_{_freq_tag(freq_hz)}",
        duration=duration,
        delta_fn=lambda t: np.zeros(3),
        rot_fn=rot,
        mirror_left_roll=True,
    )


def wrist_combo(freq_hz: float = 1.0, duration: float = 12.0) -> MockTrajectory:
    """Slow 3 cm circle + roll oscillation — 'move and articulate'."""
    amp = np.deg2rad(35.0)
    period = 6.0

    def delta(t: float) -> np.ndarray:
        ramp = _smoothstep(t / (0.25 * period))
        phase = 2 * np.pi * t / period
        return ramp * np.array(
            [0.03 * (np.cos(phase) - 1.0), 0.03 * np.sin(phase), 0.0]
        )

    def rot(t: float) -> np.ndarray:
        ramp = _smoothstep(t / 1.0)
        return _rot_about(0, ramp * amp * np.sin(2 * np.pi * freq_hz * t))

    return MockTrajectory(
        name=f"wrist_combo_{_freq_tag(freq_hz)}",
        duration=duration,
        delta_fn=delta,
        mirror_left_y=True,
        rot_fn=rot,
        mirror_left_roll=True,
    )


def _seg(t: float, t0: float, t1: float) -> float:
    """Smoothstep progress of the segment [t0, t1] at time t (0..1)."""
    return float(_smoothstep((t - t0) / (t1 - t0)))


def envelope_radial(push: float = 0.30, duration: float = 9.0) -> MockTrajectory:
    """Forward stroke far past r_max, 2 s dwell outside, smooth return."""

    def delta(t: float) -> np.ndarray:
        return np.array([push * (_seg(t, 0.0, 3.0) - _seg(t, 5.0, 8.0)), 0.0, 0.0])

    return MockTrajectory(name="envelope_radial", duration=duration, delta_fn=delta)


def envelope_swoop(dip: float = 0.20, duration: float = 9.0) -> MockTrajectory:
    """Vertical swoop well below the z-floor and back up."""

    def delta(t: float) -> np.ndarray:
        return np.array([0.0, 0.0, -dip * (_seg(t, 0.0, 3.0) - _seg(t, 5.0, 8.0))])

    return MockTrajectory(name="envelope_swoop", duration=duration, delta_fn=delta)


def envelope_slide(
    push: float = 0.28, slide: float = 0.15, duration: float = 9.0
) -> MockTrajectory:
    """Exit radially, slide sideways while outside, return.

    Distinguishes the OOE policies: 'project' keeps tracking the lateral
    slide along the boundary, 'freeze' does not.
    """

    def delta(t: float) -> np.ndarray:
        dx = push * (_seg(t, 0.0, 2.5) - _seg(t, 6.0, 8.5))
        dy = slide * (_seg(t, 2.5, 4.0) - _seg(t, 4.0, 5.5))
        return np.array([dx, dy, 0.0])

    return MockTrajectory(
        name="envelope_slide",
        duration=duration,
        delta_fn=delta,
        mirror_left_y=True,
    )


def default_suite() -> list[MockTrajectory]:
    """The benchmark suite: circles + lines + wrist-agility oscillations."""
    return [
        circle(radius=0.03),
        circle(radius=0.05),
        circle(radius=0.08),
        line(direction_deg=0),
        line(direction_deg=45),
        line(direction_deg=90),
        line(direction_deg=135),
        wrist_roll_osc(freq_hz=0.5),
        wrist_roll_osc(freq_hz=1.0),
        wrist_roll_osc(freq_hz=2.0),
        wrist_flex_osc(freq_hz=0.5),
        wrist_flex_osc(freq_hz=1.0),
        wrist_flex_osc(freq_hz=2.0),
        wrist_combo(freq_hz=1.0),
    ]


def envelope_suite() -> list[MockTrajectory]:
    """Out-of-envelope excursions (used by run_envelope.py only)."""
    return [envelope_radial(), envelope_swoop(), envelope_slide()]
