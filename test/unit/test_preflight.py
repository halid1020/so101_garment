"""Unit tests for the pure classifiers in tool/collect_preflight."""

import unittest

from tool.collect_preflight import (
    FAIL,
    OK,
    WARN,
    classify_disk,
    classify_governors,
    classify_usb_autosuspend,
    pose_status,
    sensor_map_status,
)

_POSE = {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": 0,
    "gripper": 0,
}


class TestClassifiers(unittest.TestCase):
    def test_disk_levels(self):
        self.assertEqual(classify_disk(1.0).level, FAIL)
        self.assertEqual(classify_disk(5.0).level, WARN)
        self.assertEqual(classify_disk(50.0).level, OK)

    def test_governors(self):
        self.assertEqual(classify_governors(["performance", "performance"]).level, OK)
        self.assertEqual(classify_governors(["powersave"]).level, WARN)
        self.assertEqual(classify_governors([]).level, WARN)

    def test_usb_autosuspend(self):
        self.assertEqual(classify_usb_autosuspend("-1").level, OK)
        self.assertEqual(classify_usb_autosuspend("0").level, OK)
        self.assertEqual(classify_usb_autosuspend("2").level, WARN)
        self.assertEqual(classify_usb_autosuspend("junk").level, WARN)

    def test_sensor_map_status(self):
        ok = sensor_map_status(
            {"arms": {"left": "a", "right": "b"}, "cameras": {"c": "d"}, "leaders": {}}
        )
        self.assertEqual(ok.level, OK)
        bad = sensor_map_status({"arms": {"left": "a"}, "cameras": {}, "leaders": {}})
        self.assertEqual(bad.level, FAIL)
        self.assertIn("right", bad.detail)

    def test_pose_status(self):
        self.assertEqual(pose_status("ready_pos", _POSE).level, OK)
        incomplete = dict(_POSE)
        del incomplete["wrist_roll"]
        r = pose_status("ready_pos", incomplete)
        self.assertEqual(r.level, FAIL)
        self.assertIn("wrist_roll", r.detail)


if __name__ == "__main__":
    unittest.main()
