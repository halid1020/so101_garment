"""Unit tests for the pure helpers of tool/calibrate_arms.py."""

import unittest
from pathlib import Path

from tool.calibrate_arms import (
    build_calibration_plan,
    follower_cache_fpath,
    select_arms,
)

_ROBOT_CONF = {
    "ROBOT_NAME_0": "follower_0",
    "ROBOT_NAME_1": "follower_1",
    "LEADER_ID_LEFT": "leader_left",
    "LEADER_ID_RIGHT": "leader_right",
}


def _plan(sensor_map):
    return build_calibration_plan(
        sensor_map, _ROBOT_CONF, Path("/repo/src/calibration_files")
    )


class TestBuildCalibrationPlan(unittest.TestCase):
    def test_order_and_ids(self):
        plan = _plan(
            {
                "arms": {"right": "/dev/r", "left": "/dev/l"},
                "leaders": {"right": {"port": "/dev/lr"}, "left": {"port": "/dev/ll"}},
            }
        )
        self.assertEqual(
            [(a.label, a.calib_id, a.port) for a in plan],
            [
                ("follower right", "follower_0", "/dev/r"),
                ("follower left", "follower_1", "/dev/l"),
                ("leader right", "leader_right", "/dev/lr"),
                ("leader left", "leader_left", "/dev/ll"),
            ],
        )

    def test_follower_copy_targets(self):
        plan = _plan({"arms": {"right": "/dev/r"}})
        right = next(a for a in plan if a.label == "follower right")
        self.assertEqual(
            right.copy_to, Path("/repo/src/calibration_files/follower_0.json")
        )
        leader = next(a for a in plan if a.label == "leader left")
        self.assertIsNone(leader.copy_to)

    def test_device_types_and_args(self):
        plan = _plan({})
        followers = [a for a in plan if a.device_arg == "robot"]
        leaders = [a for a in plan if a.device_arg == "teleop"]
        self.assertTrue(all(a.device_type == "so101_follower" for a in followers))
        self.assertTrue(all(a.device_type == "so101_leader" for a in leaders))

    def test_unassigned_arms_are_present_but_flagged(self):
        plan = _plan({"arms": {"right": "/dev/r"}})  # no left, no leaders
        by_label = {a.label: a for a in plan}
        self.assertTrue(by_label["follower right"].assigned)
        self.assertFalse(by_label["follower left"].assigned)
        self.assertFalse(by_label["leader right"].assigned)
        self.assertEqual(len(plan), 4)

    def test_calibrate_argv_shape(self):
        arm = next(
            a for a in _plan({"arms": {"right": "/dev/ttyR"}}) if a.side == "right"
        )
        argv = arm.calibrate_argv()
        self.assertIn("-m", argv)
        self.assertIn("lerobot.scripts.lerobot_calibrate", argv)
        self.assertIn("--robot.type=so101_follower", argv)
        self.assertIn("--robot.port=/dev/ttyR", argv)
        self.assertIn("--robot.id=follower_0", argv)


class TestFollowerCachePath(unittest.TestCase):
    def test_path_layout(self):
        p = follower_cache_fpath("follower_0", Path("/home/u/.cache/calibration"))
        self.assertEqual(
            p, Path("/home/u/.cache/calibration/robots/so_follower/follower_0.json")
        )


class TestSelectArms(unittest.TestCase):
    def _full(self):
        return _plan(
            {
                "arms": {"right": "/dev/r", "left": "/dev/l"},
                "leaders": {"right": {"port": "/dev/lr"}, "left": {"port": "/dev/ll"}},
            }
        )

    def test_empty_selects_all(self):
        self.assertEqual(len(select_arms(self._full(), [])), 4)

    def test_group_followers(self):
        sel = select_arms(self._full(), ["followers"])
        self.assertEqual({a.device_arg for a in sel}, {"robot"})
        self.assertEqual(len(sel), 2)

    def test_group_leaders(self):
        sel = select_arms(self._full(), ["leaders"])
        self.assertEqual({a.device_arg for a in sel}, {"teleop"})

    def test_exact_id(self):
        sel = select_arms(self._full(), ["leader_left"])
        self.assertEqual([a.calib_id for a in sel], ["leader_left"])

    def test_mixed_tokens(self):
        sel = select_arms(self._full(), ["follower_0", "leader_right"])
        self.assertEqual({a.calib_id for a in sel}, {"follower_0", "leader_right"})


if __name__ == "__main__":
    unittest.main()
