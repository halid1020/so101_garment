"""Unit tests for the pure helpers of src/common/sensor_view.py."""

import unittest

import numpy as np

from common.sensor_view import FrameRateCounter, _age_color, side_joint_dict


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

    def test_no_drops_without_expected_hz(self):
        c = FrameRateCounter()  # expected_hz unset → drop detection off
        c.tick(object(), now=0.0)
        c.tick(object(), now=5.0)  # huge gap, but no baseline to judge it
        self.assertEqual(c.drops, 0)

    def test_drops_counted_against_expected_period(self):
        c = FrameRateCounter(expected_hz=30.0)  # nominal 33.3 ms/frame
        c.tick(object(), now=0.0)
        c.tick(object(), now=1.0 / 30.0)  # on-time: no drop
        self.assertEqual(c.drops, 0)
        c.tick(object(), now=1.0 / 30.0 + 3.0 / 30.0)  # ~3 periods gap → 2 dropped
        self.assertEqual(c.drops, 2)


class TestAgeColor(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(_age_color(0.005), (0, 255, 0))  # green ≤ ½ frame
        self.assertEqual(_age_color(0.020), (0, 210, 255))  # amber ≤ 1 frame
        self.assertEqual(_age_color(0.050), (0, 0, 255))  # red beyond
        self.assertEqual(_age_color(None), (0, 0, 255))  # missing → red


class TestSideJointDict(unittest.TestCase):
    def _vec(self):
        # left joints 0-4, right joints 5-9 (URDF degrees)
        return np.arange(10, dtype=float) * 10.0

    def test_right_side_uses_upper_slice(self):
        d = side_joint_dict(self._vec(), "right", 0.5)
        self.assertEqual(d["shoulder_pan"], 50.0)  # index 5
        self.assertEqual(d["wrist_roll"], 90.0)  # index 9
        self.assertEqual(d["gripper"], 0.5)

    def test_left_side_uses_lower_slice(self):
        d = side_joint_dict(self._vec(), "left", None)
        self.assertEqual(d["shoulder_pan"], 0.0)  # index 0
        self.assertEqual(d["wrist_roll"], 40.0)  # index 4
        self.assertIsNone(d["gripper"])  # no gripper value yet

    def test_none_vector_gives_all_none_joints(self):
        d = side_joint_dict(None, "left", None)
        self.assertEqual(
            set(d),
            {
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            },
        )
        self.assertTrue(all(v is None for v in d.values()))

    def test_gripper_kept_when_vector_present(self):
        d = side_joint_dict(self._vec(), "left", 0.25)
        self.assertEqual(d["gripper"], 0.25)


if __name__ == "__main__":
    unittest.main()
