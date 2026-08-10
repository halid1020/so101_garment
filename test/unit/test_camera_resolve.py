"""Unit tests for wrist-camera assignment + by-path device resolution.

Covers the pure ``overlay_sensor_map_devices`` helper (recording resolves an
assigned stream to its stable /dev/v4l/by-path node) and the assignable-name
list the --assign GUI offers.
"""

import unittest

from tool.meta_quest_teleopration import overlay_sensor_map_devices
from tool.test_sensor_rates import ASSIGNABLE_CAMERA_NAMES


class TestAssignableCameraNames(unittest.TestCase):
    def test_wrist_cameras_are_assignable(self):
        # The two follower wrist cameras can be bound by-path in --assign, and
        # their names match the recorder's dataset feature keys.
        self.assertIn("wrist_camera_left", ASSIGNABLE_CAMERA_NAMES)
        self.assertIn("wrist_camera_right", ASSIGNABLE_CAMERA_NAMES)

    def test_tactile_gripper_names_retained(self):
        for name in (
            "left_arm_left_gripper",
            "left_arm_right_gripper",
            "right_arm_left_gripper",
            "right_arm_right_gripper",
        ):
            self.assertIn(name, ASSIGNABLE_CAMERA_NAMES)

    def test_single_digit_selection_keys(self):
        # The GUI selects with number keys 1..N; keep it single-digit.
        self.assertLessEqual(len(ASSIGNABLE_CAMERA_NAMES), 9)


class TestOverlaySensorMapDevices(unittest.TestCase):
    def _streams(self):
        return {
            "scene": {"device": 0, "width": 640, "height": 480},
            "wrist_camera_left": {"device": 2, "width": 640, "height": 480},
        }

    def test_assigned_stream_takes_by_path_node(self):
        sm = {"cameras": {"wrist_camera_left": "/dev/v4l/by-path/pci-usb-1.2"}}
        out = overlay_sensor_map_devices(self._streams(), sm)
        self.assertEqual(
            out["wrist_camera_left"]["device"], "/dev/v4l/by-path/pci-usb-1.2"
        )
        # Other config fields are preserved.
        self.assertEqual(out["wrist_camera_left"]["width"], 640)

    def test_unassigned_stream_keeps_yaml_index(self):
        sm = {"cameras": {"wrist_camera_left": "/dev/v4l/by-path/x"}}
        out = overlay_sensor_map_devices(self._streams(), sm)
        self.assertEqual(out["scene"]["device"], 0)  # not in the map

    def test_empty_or_missing_map_is_noop(self):
        streams = self._streams()
        for sm in ({}, {"cameras": {}}, {"cameras": None}):
            out = overlay_sensor_map_devices(streams, sm)
            self.assertEqual(out["scene"]["device"], 0)
            self.assertEqual(out["wrist_camera_left"]["device"], 2)

    def test_does_not_mutate_input(self):
        streams = self._streams()
        sm = {"cameras": {"wrist_camera_left": "/dev/v4l/by-path/x"}}
        overlay_sensor_map_devices(streams, sm)
        self.assertEqual(streams["wrist_camera_left"]["device"], 2)  # unchanged


if __name__ == "__main__":
    unittest.main()
