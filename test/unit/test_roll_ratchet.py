"""Unit tests for the wrist-roll ratchet decision logic.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_roll_ratchet

Pure maths — no MuJoCo, no hardware. The joint limits used here are the
SO-101 wrist_roll limits from the arm model.
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.roll_ratchet import (
    KEEP,
    NEUTRAL,
    REWRAP,
    RollRatchet,
    gripper_roll_about_tip,
    wrist_roll_margin_rad,
)

LO = -2.74385  # rad, SO-101 wrist_roll lower limit
HI = 2.84121  # rad, upper limit
GUARD = np.deg2rad(8.0)


class TestRollMaths(unittest.TestCase):
    def test_margin_is_distance_to_nearest_limit(self) -> None:
        self.assertAlmostEqual(wrist_roll_margin_rad(0.0, LO, HI), 2.74385)
        self.assertAlmostEqual(wrist_roll_margin_rad(HI - 0.01, LO, HI), 0.01)
        self.assertAlmostEqual(wrist_roll_margin_rad(LO + 0.02, LO, HI), 0.02)
        self.assertLess(wrist_roll_margin_rad(HI + 0.1, LO, HI), 0.0)

    def test_tip_roll_zero_at_identity(self) -> None:
        self.assertAlmostEqual(gripper_roll_about_tip(np.eye(3)), 0.0)

    def test_tip_roll_tracks_rotation_about_tip(self) -> None:
        # Tip axis is local x; rolling about it must change the reported
        # angle by the same amount (sign fixed by the convention).
        base = np.eye(3)
        r30 = Rotation.from_rotvec([np.deg2rad(30), 0, 0]).as_matrix()
        a0 = gripper_roll_about_tip(base)
        a30 = gripper_roll_about_tip(base @ r30)
        assert a0 is not None and a30 is not None
        self.assertAlmostEqual(abs(a30 - a0), np.deg2rad(30), places=6)

    def test_tip_roll_degenerate_when_tip_vertical(self) -> None:
        # Rotate local x onto world z: the horizontal reference vanishes.
        rot = Rotation.from_rotvec([0, -np.pi / 2, 0]).as_matrix()
        self.assertIsNone(gripper_roll_about_tip(rot))

    def test_pi_flip_of_reference_shifts_roll_by_pi(self) -> None:
        # Negating the horizontal knuckle reference is the jaw-equivalent
        # half-turn the rewrap relies on: verify via the angle between a
        # reference and its negation as seen about the tip axis.
        rot = Rotation.from_rotvec([np.deg2rad(40), 0, 0]).as_matrix()
        flipped = rot @ Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
        a = gripper_roll_about_tip(rot)
        b = gripper_roll_about_tip(flipped)
        assert a is not None and b is not None
        diff = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
        self.assertAlmostEqual(diff, np.pi, places=6)


class TestRollRatchetDecisions(unittest.TestCase):
    def setUp(self) -> None:
        self.r = RollRatchet(lo=LO, hi=HI, guard_rad=GUARD)

    def test_keep_when_healthy(self) -> None:
        self.assertEqual(
            self.r.decide_at_grip(
                "left", 0.0, reset_requested=False, trigger_held=False
            ),
            KEEP,
        )

    def test_rewrap_near_either_limit(self) -> None:
        near_hi = HI - 0.5 * GUARD
        near_lo = LO + 0.5 * GUARD
        self.assertEqual(self.r.decide_at_grip("left", near_hi, False, False), REWRAP)
        self.assertEqual(self.r.decide_at_grip("left", near_lo, False, False), REWRAP)

    def test_trigger_blocks_rewrap_but_not_reset(self) -> None:
        near_hi = HI - 0.5 * GUARD
        self.assertEqual(self.r.decide_at_grip("left", near_hi, False, True), KEEP)
        self.assertEqual(self.r.decide_at_grip("left", near_hi, True, True), NEUTRAL)

    def test_reset_wins_even_when_healthy(self) -> None:
        self.assertEqual(self.r.decide_at_grip("left", 0.0, True, False), NEUTRAL)

    def test_mid_hold_warning_throttles_and_rearms(self) -> None:
        near = HI - 0.5 * GUARD
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=0.0))
        self.assertFalse(self.r.should_warn_mid_hold("left", near, t=0.5))
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=2.5))
        # Leaving the band re-arms the edge for the next approach.
        self.assertFalse(self.r.should_warn_mid_hold("left", 0.0, t=2.6))
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=2.7))

    def test_sides_warn_independently(self) -> None:
        near = HI - 0.5 * GUARD
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=0.0))
        self.assertTrue(self.r.should_warn_mid_hold("right", near, t=0.1))

    def test_reset_clears_warn_state(self) -> None:
        near = HI - 0.5 * GUARD
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=0.0))
        self.r.reset()
        self.assertTrue(self.r.should_warn_mid_hold("left", near, t=0.1))


if __name__ == "__main__":
    unittest.main()
