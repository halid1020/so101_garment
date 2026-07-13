"""Wrist-roll ratcheting: unbounded gripper roll on a finite-range joint.

The armplane pipeline anchors the roll target to the gripper's *current*
roll at every clutch engagement (the "knuckle" capture in the IK thread),
so repeated twist -> release -> untwist -> re-grip cycles already
accumulate roll — an implicit ratchet. What stops it is the joint itself:
``wrist_roll`` spans only ~320 degrees (about -157 to +163), so a few
cycles in one direction park the joint at its limit, where further hand
twist does nothing and the operator perceives the roll as "stuck".

This module makes the ratchet explicit and removes the ceiling:

* **Pi-equivalence rewrap.** A parallel-jaw gripper is functionally
  identical under a half-turn of roll (the jaws swap). When a clutch
  engagement finds the solved wrist-roll within a guard band of a joint
  limit, the engagement re-anchors to the jaw-equivalent orientation half
  a turn away — restoring headroom with zero change to what the gripper
  can do. The existing one-second orientation blend glides the wrist
  there, and the rewrap is suppressed while the trigger is held (an
  object in the jaws should never be spun half a turn uninvited).
* **Mid-hold hint.** While clutched, entering the guard band prints a
  throttled reminder to release, untwist, and re-grip.
* **Reset to neutral.** A per-hand joystick click queues a reset: the
  next engagement anchors the roll to the joint's zero instead of the
  current roll, and the blend glides the gripper back.

The decision logic lives here, dependency-light, so it is unit-testable
without MuJoCo or hardware; the IK thread consults it at each clutch
engagement and each tick.
"""

from __future__ import annotations

import numpy as np

# Cosine threshold beyond which the tip is treated as vertical and the
# roll reference is undefined (matches the telegrip split-IK guard).
_DEG_TIP_Z = 0.995

# Actions decide_at_grip can return.
KEEP = "keep"
REWRAP = "rewrap"
NEUTRAL = "neutral"

# Seconds between repeated mid-hold "near roll limit" hints.
_WARN_PERIOD_S = 2.0


def gripper_roll_about_tip(rot: np.ndarray) -> float | None:
    """Signed angle of the EE local z about the tip axis vs horizontal.

    The reference is the horizontal direction obtained by projecting
    world-z out of the tip axis; returns None when the tip is
    near-vertical (reference undefined) — callers hold the previous roll
    in that case.
    """
    tip = rot[:, 0]
    if abs(tip[2]) >= _DEG_TIP_Z:
        return None
    u = np.array([0.0, 0.0, 1.0]) - tip[2] * tip
    u /= np.linalg.norm(u)
    z_ax = rot[:, 2]
    return float(np.arctan2(np.cross(u, z_ax) @ tip, u @ z_ax))


def wrist_roll_margin_rad(q_roll: float, lo: float, hi: float) -> float:
    """Distance (rad) from a wrist-roll angle to its nearest joint limit."""
    return float(min(q_roll - lo, hi - q_roll))


class RollRatchet:
    """Per-side roll-ratchet decisions for the IK thread.

    lo/hi are the wrist_roll joint limits (rad); guard_rad is the margin
    below which an engagement rewraps or a hold warns.
    """

    def __init__(self, lo: float, hi: float, guard_rad: float) -> None:
        self.lo = lo
        self.hi = hi
        self.guard_rad = guard_rad
        self._last_warn_t: dict[str, float] = {}
        self._in_band: dict[str, bool] = {}

    def decide_at_grip(
        self,
        side: str,
        q_roll: float,
        reset_requested: bool,
        trigger_held: bool,
    ) -> str:
        """Choose the roll anchor for a clutch engagement.

        Priority: an operator-requested reset wins (it is explicit, and
        the blend makes it gentle even mid-grasp); otherwise rewrap when
        parked near a limit — unless the trigger holds an object, in
        which case the anchor is kept and the operator keeps control.
        """
        if reset_requested:
            return NEUTRAL
        near_limit = wrist_roll_margin_rad(q_roll, self.lo, self.hi) < self.guard_rad
        if near_limit and not trigger_held:
            return REWRAP
        return KEEP

    def should_warn_mid_hold(self, side: str, q_roll: float, t: float) -> bool:
        """Throttled True when the solved roll sits in the guard band.

        Re-arms once the joint leaves the band, so each approach to the
        limit produces a fresh (but rate-limited) reminder.
        """
        in_band = wrist_roll_margin_rad(q_roll, self.lo, self.hi) < self.guard_rad
        was_in_band = self._in_band.get(side, False)
        self._in_band[side] = in_band
        if not in_band:
            return False
        last = self._last_warn_t.get(side)
        if not was_in_band or last is None or t - last >= _WARN_PERIOD_S:
            self._last_warn_t[side] = t
            return True
        return False

    def reset(self) -> None:
        """Clear per-episode state (teleop deactivation)."""
        self._last_warn_t.clear()
        self._in_band.clear()
