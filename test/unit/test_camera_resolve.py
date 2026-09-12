"""Unit tests for camera assignment + device/serial resolution.

Covers the pure ``overlay_sensor_map_devices`` helper (recording resolves an
assigned wrist stream to its stable /dev/v4l/by-path node), the assignable-name
list the --assign GUI offers, and the central RealSense serial resolution +
capture-gating helpers.
"""

import argparse
import unittest

import cv2  # type: ignore[import]
from actoris_harena.recording.cameras import CameraCapture

from tool.meta_quest_teleopration import (
    build_realsense_capture,
    overlay_sensor_map_devices,
    resolve_camera_streams,
    resolve_realsense_serial,
)
from tool.test_sensor_rates import ASSIGNABLE_CAMERA_NAMES, TACTILE_CAMERA_NAMES


class TestAssignableCameraNames(unittest.TestCase):
    def test_wrist_cameras_are_assignable(self):
        # The two follower wrist cameras can be bound by-path in --assign, and
        # their names match the recorder's dataset feature keys.
        self.assertIn("wrist_camera_left", ASSIGNABLE_CAMERA_NAMES)
        self.assertIn("wrist_camera_right", ASSIGNABLE_CAMERA_NAMES)

    def test_tactile_gripper_names_retained(self):
        for name in TACTILE_CAMERA_NAMES:
            self.assertIn(name, ASSIGNABLE_CAMERA_NAMES)

    def test_the_tactile_names_say_which_arm_and_finger(self):
        # Four identical-looking cameras: the name is the only thing that tells
        # an operator which one a bad stream belongs to.
        self.assertEqual(len(TACTILE_CAMERA_NAMES), 4)
        for name in TACTILE_CAMERA_NAMES:
            self.assertRegex(name, r"^(left|right)_arm_(left|right)_gripper$")


class TestResolveCameraStreams(unittest.TestCase):
    """--tactile, --enable-camera and --disable-camera over the seven streams."""

    def _cfg(self):
        cams = {
            name: {"enabled": False, "device": i}
            for i, name in enumerate(ASSIGNABLE_CAMERA_NAMES)
        }
        cams["central"]["enabled"] = True
        return {"cameras": cams}

    def _args(self, **kw):
        base = {"tactile": False, "enable_camera": [], "disable_camera": []}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_tactile_enables_exactly_the_four_gripper_streams(self):
        out = resolve_camera_streams(self._cfg(), self._args(tactile=True))
        self.assertEqual(
            set(out), {*TACTILE_CAMERA_NAMES, "central"}  # central was already on
        )

    def test_without_tactile_only_the_yaml_defaults_are_on(self):
        out = resolve_camera_streams(self._cfg(), self._args())
        self.assertEqual(set(out), {"central"})

    def test_disable_beats_tactile(self):
        out = resolve_camera_streams(
            self._cfg(),
            self._args(tactile=True, disable_camera=["left_arm_left_gripper"]),
        )
        self.assertNotIn("left_arm_left_gripper", out)

    def test_a_gripper_name_can_be_enabled_on_its_own(self):
        # It used to be rejected as unknown: the name existed in sensor_map.yaml
        # but not in recording.yaml, so nothing could enable it.
        out = resolve_camera_streams(
            self._cfg(), self._args(enable_camera=["right_arm_left_gripper"])
        )
        self.assertIn("right_arm_left_gripper", out)

    def test_an_unknown_camera_is_still_refused_by_name(self):
        with self.assertRaises(SystemExit):
            resolve_camera_streams(self._cfg(), self._args(enable_camera=["tactile_0"]))

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


class TestCameraExposureConfiguration(unittest.TestCase):
    """The capture object must apply a pinned exposure, and only when asked."""

    class _FakeCap:
        def __init__(self):
            self.props = {}

        def set(self, prop, value):
            self.props[prop] = value
            return True

    def _configured(self, exposure):
        cam = CameraCapture(
            "c",
            0,
            640,
            480,
            30,
            False,
            fourcc="MJPG",
            controls={"exposure": exposure} if exposure else {},
        )
        cap = self._FakeCap()
        cam._configure(cap)
        return cap.props

    def test_a_pinned_exposure_switches_the_camera_out_of_automatic(self):
        # Setting the value alone does nothing while the camera is still
        # choosing its own; both have to be sent, in this order.
        props = self._configured(300)
        self.assertEqual(props[cv2.CAP_PROP_AUTO_EXPOSURE], 1)
        self.assertEqual(props[cv2.CAP_PROP_EXPOSURE], 300)

    def test_zero_leaves_automatic_exposure_alone(self):
        props = self._configured(0)
        self.assertNotIn(cv2.CAP_PROP_AUTO_EXPOSURE, props)
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, props)

    def test_the_format_and_size_are_still_applied(self):
        props = self._configured(300)
        self.assertEqual(props[cv2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertIn(cv2.CAP_PROP_FOURCC, props)


if __name__ == "__main__":
    unittest.main()
