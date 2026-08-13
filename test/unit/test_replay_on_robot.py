"""Unit tests for the pure helpers of tool/replay_on_robot.py (no hardware)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool.replay_on_robot import (
    _OFFSETS,
    _SIGNS,
    action_to_goal,
    present_to_urdf,
    saved_episode_count,
    split_action,
    tracking_errors,
)


class TestSplitAction(unittest.TestCase):
    def test_splits_body_and_gripper_per_side(self):
        vec = np.arange(12, dtype=float)
        d = split_action(vec)
        np.testing.assert_array_equal(d["left"][0], [0, 1, 2, 3, 4])
        self.assertEqual(d["left"][1], 5.0)
        np.testing.assert_array_equal(d["right"][0], [6, 7, 8, 9, 10])
        self.assertEqual(d["right"][1], 11.0)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            split_action(np.zeros(10))


class TestActionToGoal(unittest.TestCase):
    def test_matches_signs_offset_conversion(self):
        # Hand-worked against the exact dual_joint_state conversion.
        body = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        goal = action_to_goal("left", body, 0.25)
        expected = _SIGNS["left"] * (body - _OFFSETS["left"])
        self.assertAlmostEqual(goal["shoulder_pan"], expected[0])
        self.assertAlmostEqual(goal["wrist_roll"], expected[4])
        self.assertAlmostEqual(goal["gripper"], 25.0)  # 0.25 → 0–100

    def test_round_trip_urdf_through_hardware(self):
        # action_to_goal then present_to_urdf must recover the URDF body + frac.
        body = np.array([5.0, -3.0, 12.0, 0.0, 90.0])
        goal = action_to_goal("right", body, 0.6)
        positions = {j: goal[j] for j in goal}
        urdf6 = present_to_urdf("right", positions)
        np.testing.assert_allclose(urdf6[:5], body, atol=1e-9)
        self.assertAlmostEqual(urdf6[5], 0.6)


class TestTrackingErrors(unittest.TestCase):
    def test_mean_and_max_per_channel(self):
        rec = np.zeros((4, 12))
        rep = np.zeros((4, 12))
        rep[:, 0] = [1.0, 3.0, 0.0, 2.0]  # channel 0: mean 1.5, max 3.0
        mean_err, max_err = tracking_errors(rec, rep)
        self.assertAlmostEqual(mean_err[0], 1.5)
        self.assertAlmostEqual(max_err[0], 3.0)
        self.assertAlmostEqual(mean_err[1], 0.0)


class TestSavedEpisodeCount(unittest.TestCase):
    def test_missing_info_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(saved_episode_count(Path(d)), 0)

    def test_reads_total_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d) / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(json.dumps({"total_episodes": 7}))
            self.assertEqual(saved_episode_count(Path(d)), 7)

    def test_corrupt_json_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d) / "meta"
            meta.mkdir()
            (meta / "info.json").write_text("{not json")
            self.assertEqual(saved_episode_count(Path(d)), 0)


if __name__ == "__main__":
    unittest.main()
