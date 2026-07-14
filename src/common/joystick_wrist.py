"""Clutched, joint-space wrist trims driven by the Meta-Quest thumbsticks.

The ``mymethod`` teleop stack lets the operator adjust each gripper's wrist
directly from that controller's thumbstick, independently of where the handle
is pointing. Stick x (left/right) trims the arm's ``wrist_roll``; stick y
(forward/back) trims its ``wrist_flex``. Both are integrated in JOINT space —
the stick commands a joint RATE, not a target pose — so the mapping is
immune to IK singularities and never fights the reach envelope.

The trim is *clutched*: while a stick is deflected past the deadzone the
consuming thread latches the rest of that arm's joints and ignores the
handle, so only wrist_roll/wrist_flex move; on release the wrist stays where
it was left and handle teleoperation resumes from the new pose. This class is
deliberately free of any robot model (numpy only): it owns only the axis
shaping, the rate integration, and the per-side engage/release edge
detection. The consuming thread (``common.threads.dual_ik_solver``) owns the
joint latch, the floor guard, and the re-anchoring.

Sign conventions live in the constructor (bound from the YAML) so a
real-robot direction flip is a config edit, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WristTrim:
    """One update's worth of wrist trim for a single arm.

    ``engaged`` is True whenever either raw axis sits beyond the deadzone;
    ``just_engaged`` / ``just_released`` are the rising/falling edges of that
    state versus the previous update for the same side. ``d_roll`` / ``d_flex``
    are the signed joint increments (radians) to apply THIS tick.
    """

    engaged: bool
    just_engaged: bool
    just_released: bool
    d_roll: float
    d_flex: float


class JoystickWristTrim:
    """Shapes and integrates thumbstick deflection into wrist-joint trims.

    Per-axis shaping applies a deadzone and an expo curve: ``0`` inside the
    deadzone, otherwise ``sign(v) * ((|v| - deadzone) / (1 - deadzone)) **
    expo``. The rescaling makes the response continuous at the deadzone edge
    (it starts from zero there), and ``expo > 1`` gives fine control near the
    centre while still reaching full rate at full deflection.
    """

    def __init__(
        self,
        deadzone: float,
        expo: float,
        roll_rate_rad_s: float,
        flex_rate_rad_s: float,
        roll_sign: float,
        flex_sign: float,
    ) -> None:
        self.deadzone = float(deadzone)
        self.expo = float(expo)
        self.roll_rate_rad_s = float(roll_rate_rad_s)
        self.flex_rate_rad_s = float(flex_rate_rad_s)
        self.roll_sign = float(roll_sign)
        self.flex_sign = float(flex_sign)
        # Per-side engagement state from the previous update, for edge detection.
        self._engaged_prev: dict[str, bool] = {"left": False, "right": False}

    def _shape(self, v: float) -> float:
        """Deadzone + expo shaping of one raw axis value in [-1, 1]."""
        av = abs(v)
        if av <= self.deadzone:
            return 0.0
        # Rescale so the useful range starts at 0 right at the deadzone edge.
        s = (av - self.deadzone) / (1.0 - self.deadzone)
        return float(np.sign(v)) * (s**self.expo)

    def update(self, side: str, x: float, y: float, dt: float) -> WristTrim:
        """Shape/integrate one tick of stick input for ``side`` ('left'|'right').

        ``x`` (left/right) drives wrist_roll, ``y`` (forward/back) drives
        wrist_flex; both raw in [-1, 1]. ``dt`` is the tick duration (s).
        """
        engaged = abs(x) > self.deadzone or abs(y) > self.deadzone
        prev = self._engaged_prev[side]
        just_engaged = engaged and not prev
        just_released = prev and not engaged
        self._engaged_prev[side] = engaged
        d_roll = self.roll_sign * self.roll_rate_rad_s * self._shape(x) * dt
        d_flex = self.flex_sign * self.flex_rate_rad_s * self._shape(y) * dt
        return WristTrim(
            engaged=engaged,
            just_engaged=just_engaged,
            just_released=just_released,
            d_roll=d_roll,
            d_flex=d_flex,
        )

    def reset(self) -> None:
        """Clear the per-side edge state (call when teleop recalibrates)."""
        self._engaged_prev = {"left": False, "right": False}
