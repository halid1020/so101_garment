"""Unit tests for the pure helpers of tool/set_wrist_roll_limits.py."""

import unittest

from tool.set_wrist_roll_limits import (
    FULL_MAX,
    FULL_MIN,
    MIN_BAND_DEG,
    apply_margin,
    deg_to_ticks,
    ticks_to_hw_deg,
    validate_band,
)


class TestConversions(unittest.TestCase):
    def test_deg_to_ticks_matches_lerobot_scale(self):
        # lerobot DEGREES: degrees = ticks * 360 / 4095 about the mid.
        self.assertEqual(deg_to_ticks(360.0), 4095)
        self.assertEqual(deg_to_ticks(5.0), round(5.0 * 4095 / 360.0))

    def test_ticks_to_hw_deg_zero_at_mid(self):
        self.assertAlmostEqual(ticks_to_hw_deg(2047.5), 0.0)
        self.assertAlmostEqual(ticks_to_hw_deg(FULL_MAX), 180.0, delta=0.1)
        self.assertAlmostEqual(ticks_to_hw_deg(FULL_MIN), -180.0, delta=0.1)

    def test_round_trip(self):
        for deg in (-90.0, -5.0, 0.0, 45.0, 120.0):
            ticks = deg_to_ticks(deg) + 2047.5
            self.assertAlmostEqual(ticks_to_hw_deg(ticks), deg, delta=0.05)


class TestApplyMargin(unittest.TestCase):
    def test_shrinks_both_sides(self):
        lo, hi = apply_margin(1000, 3000, 5.0)
        m = deg_to_ticks(5.0)
        self.assertEqual((lo, hi), (1000 + m, 3000 - m))

    def test_zero_margin_is_identity(self):
        self.assertEqual(apply_margin(1000, 3000, 0.0), (1000, 3000))


class TestValidateBand(unittest.TestCase):
    def test_good_band_passes(self):
        self.assertEqual(validate_band(1500, 2600, 2000), [])

    def test_wrap_hugging_band_rejected(self):
        problems = validate_band(10, 4090, 2000)
        self.assertTrue(any("wrap" in p for p in problems))

    def test_one_sided_extreme_is_fine(self):
        # Only one end near the wrap is a legitimate (non-wrapping) band.
        self.assertEqual(validate_band(60, 2600, 2000), [])

    def test_narrow_band_rejected(self):
        narrow = deg_to_ticks(MIN_BAND_DEG) - 20
        problems = validate_band(2000, 2000 + narrow, 2010)
        self.assertTrue(any("narrower" in p for p in problems))

    def test_present_outside_band_rejected(self):
        problems = validate_band(1500, 2600, 3000)
        self.assertTrue(any("OUTSIDE" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
