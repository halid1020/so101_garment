#!/usr/bin/env python3
"""Unit tests for the clutched thumbstick wrist trims (common/joystick_wrist.py).

Pure maths behind the ``mymethod`` clutch: deadzone + expo shaping, joint-rate
integration, and the per-side engage/release edge detection the IK thread uses
to latch and re-anchor. No robot model is involved.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_joystick_wrist
"""

import unittest

from common.joystick_wrist import JoystickWristTrim


def _make(**overrides) -> JoystickWristTrim:
    params = dict(
        deadzone=0.15,
        expo=2.0,
        roll_rate_rad_s=1.0,
        flex_rate_rad_s=1.0,
        roll_sign=1.0,
        flex_sign=1.0,
    )
    params.update(overrides)
    return JoystickWristTrim(**params)


class TestDeadzone(unittest.TestCase):
    def test_inside_deadzone_no_engage_no_trim(self) -> None:
        trim = _make()
        res = trim.update("left", 0.1, -0.14, 0.01)
        self.assertFalse(res.engaged)
        self.assertFalse(res.just_engaged)
        self.assertEqual(res.d_roll, 0.0)
        self.assertEqual(res.d_flex, 0.0)

    def test_beyond_deadzone_engages(self) -> None:
        trim = _make()
        res = trim.update("left", 0.5, 0.0, 0.01)
        self.assertTrue(res.engaged)
        self.assertNotEqual(res.d_roll, 0.0)


class TestEdges(unittest.TestCase):
    def test_engage_release_edges(self) -> None:
        trim = _make()
        # Centre: no engagement.
        r0 = trim.update("left", 0.0, 0.0, 0.01)
        self.assertFalse(r0.engaged)
        self.assertFalse(r0.just_engaged)
        # Deflect: rising edge exactly once.
        r1 = trim.update("left", 0.6, 0.0, 0.01)
        self.assertTrue(r1.engaged)
        self.assertTrue(r1.just_engaged)
        self.assertFalse(r1.just_released)
        # Hold: engaged but no new edge.
        r2 = trim.update("left", 0.6, 0.0, 0.01)
        self.assertTrue(r2.engaged)
        self.assertFalse(r2.just_engaged)
        # Release: falling edge exactly once.
        r3 = trim.update("left", 0.0, 0.0, 0.01)
        self.assertFalse(r3.engaged)
        self.assertTrue(r3.just_released)
        self.assertFalse(r3.just_engaged)
        # Stay centred: no edge.
        r4 = trim.update("left", 0.0, 0.0, 0.01)
        self.assertFalse(r4.just_released)

    def test_per_side_independence(self) -> None:
        trim = _make()
        trim.update("left", 0.6, 0.0, 0.01)  # left engages
        rl = trim.update("left", 0.6, 0.0, 0.01)
        rr = trim.update("right", 0.6, 0.0, 0.01)
        self.assertFalse(rl.just_engaged)  # left already engaged
        self.assertTrue(rr.just_engaged)  # right's first engagement

    def test_reset_clears_edges(self) -> None:
        trim = _make()
        trim.update("left", 0.6, 0.0, 0.01)  # engaged
        trim.reset()
        # After reset the next deflection is a fresh rising edge again.
        res = trim.update("left", 0.6, 0.0, 0.01)
        self.assertTrue(res.just_engaged)


class TestShaping(unittest.TestCase):
    def test_continuous_at_deadzone_edge(self) -> None:
        trim = _make()
        eps = 1e-4
        just_inside = trim._shape(0.15 - eps)
        just_outside = trim._shape(0.15 + eps)
        self.assertEqual(just_inside, 0.0)
        self.assertAlmostEqual(just_outside, 0.0, places=3)  # starts from zero

    def test_expo_softens_near_centre(self) -> None:
        # With expo > 1 the mid-throw response is below the linear (expo=1) one.
        expo = _make(expo=2.0)
        linear = _make(expo=1.0)
        self.assertLess(abs(expo._shape(0.5)), abs(linear._shape(0.5)))

    def test_full_deflection_reaches_unity(self) -> None:
        trim = _make()
        self.assertAlmostEqual(trim._shape(1.0), 1.0, places=9)
        self.assertAlmostEqual(trim._shape(-1.0), -1.0, places=9)

    def test_shape_is_odd(self) -> None:
        trim = _make()
        self.assertAlmostEqual(trim._shape(0.7), -trim._shape(-0.7), places=9)


class TestIntegration(unittest.TestCase):
    def test_full_deflection_one_second_matches_rate(self) -> None:
        # Integrating full deflection (shaped -> 1) for 1 s ~= the rate.
        trim = _make(roll_rate_rad_s=0.6, flex_rate_rad_s=0.45)
        dt = 0.001
        roll = 0.0
        flex = 0.0
        for _ in range(1000):
            res = trim.update("left", 1.0, 1.0, dt)
            roll += res.d_roll
            flex += res.d_flex
        self.assertAlmostEqual(roll, 0.6, places=6)
        self.assertAlmostEqual(flex, 0.45, places=6)

    def test_axis_mapping(self) -> None:
        # x drives roll, y drives flex, independently.
        trim = _make()
        res = trim.update("left", 0.6, 0.0, 0.01)
        self.assertNotEqual(res.d_roll, 0.0)
        self.assertEqual(res.d_flex, 0.0)
        trim.reset()
        res = trim.update("left", 0.0, 0.6, 0.01)
        self.assertEqual(res.d_roll, 0.0)
        self.assertNotEqual(res.d_flex, 0.0)


class TestSigns(unittest.TestCase):
    def test_sign_flips_honoured(self) -> None:
        pos = _make(roll_sign=1.0, flex_sign=1.0)
        neg = _make(roll_sign=-1.0, flex_sign=-1.0)
        rp = pos.update("left", 0.6, 0.6, 0.01)
        rn = neg.update("left", 0.6, 0.6, 0.01)
        self.assertAlmostEqual(rp.d_roll, -rn.d_roll, places=9)
        self.assertAlmostEqual(rp.d_flex, -rn.d_flex, places=9)
        self.assertGreater(rp.d_roll, 0.0)  # +x, +sign -> positive roll trim

    def test_stick_direction_sets_trim_sign(self) -> None:
        trim = _make()
        left_res = trim.update("left", -0.6, 0.0, 0.01)
        self.assertLess(left_res.d_roll, 0.0)  # -x -> negative roll trim


if __name__ == "__main__":
    unittest.main()
