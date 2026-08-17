"""Unit tests for camera assignment + device/serial resolution.

Covers the pure ``overlay_sensor_map_devices`` helper (recording resolves an
assigned wrist stream to its stable /dev/v4l/by-path node), the assignable-name
list the --assign GUI offers, and the central RealSense serial resolution +
capture-gating helpers.
"""

import argparse
import unittest

from common.recording.cameras import hardware_hint
from tool.meta_quest_teleopration import (
    build_realsense_capture,
    overlay_sensor_map_devices,
    resolve_realsense_serial,
)
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

    def test_central_camera_is_assignable(self):
        # The central overhead camera is now a plain UVC stream bound by-path in
        # --assign, like the wrist cameras (no longer a serial-keyed RealSense).
        self.assertIn("central", ASSIGNABLE_CAMERA_NAMES)

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


class TestResolveRealsenseSerial(unittest.TestCase):
    def test_sensor_map_serial_wins(self):
        rs_cfg = {"serial": "YAML123"}
        sm = {"realsense": {"serial": "MAP456"}}
        self.assertEqual(resolve_realsense_serial(rs_cfg, sm), "MAP456")

    def test_falls_back_to_recording_yaml(self):
        self.assertEqual(resolve_realsense_serial({"serial": "YAML123"}, {}), "YAML123")

    def test_empty_when_neither(self):
        self.assertEqual(resolve_realsense_serial({"serial": ""}, {}), "")
        self.assertEqual(resolve_realsense_serial({}, {"realsense": {}}), "")


class TestBuildRealsenseCaptureGating(unittest.TestCase):
    """The gate is exercised WITHOUT importing pyrealsense2 (None paths)."""

    def _args(self, central_depth=False):
        return argparse.Namespace(central_depth=central_depth)

    def test_none_when_no_realsense_section(self):
        self.assertIsNone(build_realsense_capture({}, self._args(), {}, for_view=False))
        self.assertIsNone(build_realsense_capture({}, self._args(), {}, for_view=True))

    def test_none_when_configured_but_not_wanted(self):
        rec_cfg = {"realsense": {"enabled": False}}
        # Recording: off + no --central-depth → None.
        self.assertIsNone(
            build_realsense_capture(rec_cfg, self._args(), {}, for_view=False)
        )
        # View: off + no flag + no assigned serial → None.
        self.assertIsNone(
            build_realsense_capture(rec_cfg, self._args(), {}, for_view=True)
        )

    def test_view_without_serial_stays_none(self):
        rec_cfg = {"realsense": {"enabled": False}}
        sm = {"realsense": {"serial": ""}}  # assigned but empty
        self.assertIsNone(
            build_realsense_capture(rec_cfg, self._args(), sm, for_view=True)
        )


class TestHardwareHint(unittest.TestCase):
    """What a repeatedly dropping camera reports to the operator."""

    def test_names_the_stream_and_the_count(self):
        msg = hardware_hint("wrist_camera_left", 6)
        self.assertIn("wrist_camera_left", msg)
        self.assertIn("6 times", msg)

    def test_a_single_drop_reads_as_a_note_not_an_alarm(self):
        self.assertIn("once", hardware_hint("central", 1))
        self.assertNotIn("1 times", hardware_hint("central", 1))

    def test_says_it_is_physical_and_where_to_look(self):
        # The operator's next action is with their hands, not the code, so the
        # message has to say so rather than just report a retry.
        msg = hardware_hint("wrist_camera_left", 3)
        self.assertIn("physical", msg)
        for where in ("connector", "controller", "hub"):
            self.assertIn(where, msg)


if __name__ == "__main__":
    unittest.main()
