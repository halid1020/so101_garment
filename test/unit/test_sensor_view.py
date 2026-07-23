"""Unit tests for the pure helpers of src/common/sensor_view.py."""

import unittest

import numpy as np

from common.sensor_view import FrameRateCounter, build_arm_panel_lines


class TestFrameRateCounter(unittest.TestCase):
    def test_same_object_counts_once(self):
        c = FrameRateCounter(window_s=2.0)
        frame = np.zeros((2, 2, 3))
        c.tick(frame, now=0.0)
        c.tick(frame, now=0.1)  # same object: not a new frame
        self.assertAlmostEqual(c.hz(now=0.2), 0.5)  # 1 frame / 2 s window

    def test_distinct_objects_count(self):
        c = FrameRateCounter(window_s=2.0)
        for i in range(6):
            c.tick(np.zeros((2, 2, 3)), now=0.1 * i)
        self.assertAlmostEqual(c.hz(now=0.5), 3.0)  # 6 frames / 2 s

    def test_old_stamps_expire(self):
        c = FrameRateCounter(window_s=1.0)
        c.tick(np.zeros(1), now=0.0)
        c.tick(np.zeros(1), now=5.0)
        self.assertAlmostEqual(c.hz(now=5.0), 1.0)  # only the recent one

    def test_none_frame_ignored(self):
        c = FrameRateCounter()
        c.tick(None, now=0.0)
        self.assertEqual(c.hz(now=0.0), 0.0)


class TestBuildArmPanelLines(unittest.TestCase):
    def _vec(self):
        # left joints 0-4, right joints 5-9 (URDF degrees)
        return np.arange(10, dtype=float) * 10.0

    def test_right_side_uses_upper_slice(self):
        lines = build_arm_panel_lines(
            "right", self._vec(), self._vec(), 0.5, 1.0, "ENABLED", True
        )
        self.assertIn("right arm", lines[0])
        self.assertIn("teleop", lines[0])
        self.assertIn("+50.0", lines[2])  # shoulder_pan = index 5 -> 50.0
        self.assertIn("+90.0", lines[6])  # wrist_roll = index 9 -> 90.0

    def test_left_side_uses_lower_slice(self):
        lines = build_arm_panel_lines(
            "left", self._vec(), None, None, None, "DISABLED", False
        )
        self.assertNotIn("teleop", lines[0])
        self.assertIn("+0.0", lines[2])  # shoulder_pan = index 0
        self.assertIn("--", lines[2])  # no command vector yet

    def test_all_none_before_first_publish(self):
        lines = build_arm_panel_lines("left", None, None, None, None, "DISABLED", False)
        self.assertEqual(len(lines), 8)  # header + column row + 5 joints + gripper
        for row in lines[2:]:
            self.assertIn("--", row)

    def test_gripper_fractions_formatted(self):
        lines = build_arm_panel_lines(
            "left", self._vec(), self._vec(), 0.25, 1.0, "ENABLED", False
        )
        self.assertIn("0.25", lines[-1])
        self.assertIn("1.00", lines[-1])


if __name__ == "__main__":
    unittest.main()
