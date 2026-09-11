#!/usr/bin/env python3
"""Unit tests for the recording (data-collection) YAML loading layer.

Schema behaviour ONLY — camera device indices are per-machine hardware wiring,
so no frozen device values are asserted here (unlike test_config_yaml.py's
teleop-tuning regression). Covers:

* the checked-in src/conf/recording.yaml loads and validates;
* an unknown key (top-level, section, or camera) raises a clear error;
* a missing key raises a clear error;
* the tactile streams default to disabled.

Run via: python -m unittest test.unit.test_recording_config
(requires PYTHONPATH=.:src, as set by `source setup.sh`).
"""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml
from actoris_harena.recording.camera_controls import CONTROL_NAMES

from common.config_parser import load_recording_config
from tool.test_sensor_rates import ASSIGNABLE_CAMERA_NAMES, TACTILE_CAMERA_NAMES

_VALID: dict = {
    "dataset": {
        "fps": 30,
        "image_writer_threads_per_camera": 2,
        "robot_type": "so101_dual",
    },
    "sidecar": {
        "enabled": True,
        "rate_hz": 100.0,
        "include_hw_frame_goal": True,
    },
    "cameras": {
        "scene": {
            "enabled": True,
            "device": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "rotate180": False,
        },
    },
}

_REALSENSE: dict = {
    "enabled": False,
    "serial": "",
    "width": 640,
    "height": 480,
    "fps": 30,
    "align_to_color": True,
    "rgb_name": "central",
    "depth_name": "central_depth",
    "lock_auto_exposure": True,
}

_AUDIO: dict = {
    "enabled": True,
    "start_sound": "start.wav",
    "stop_sound": "stop.wav",
}


def _write_yaml(data: dict, directory: str) -> Path:
    path = Path(directory) / "recording.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


class TestRecordingConfig(unittest.TestCase):
    def test_checked_in_yaml_loads(self) -> None:
        cfg = load_recording_config()
        self.assertIn("dataset", cfg)
        self.assertIn("sidecar", cfg)
        self.assertIn("cameras", cfg)
        self.assertIsInstance(cfg["dataset"]["fps"], int)
        # Expected stream names present (schema, not device values).
        for name in ("scene", "wrist_camera_left", "wrist_camera_right"):
            self.assertIn(name, cfg["cameras"])

    def test_valid_dict_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(_VALID, d)))
        self.assertEqual(cfg["dataset"]["fps"], 30)
        self.assertTrue(cfg["cameras"]["scene"]["enabled"])

    def test_unknown_top_level_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["surprise"] = {}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "surprise"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_unknown_section_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["dataset"]["bogus"] = 1
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "bogus"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_unknown_camera_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["cameras"]["scene"]["gamma"] = 2.2
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "gamma"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_missing_section_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        del bad["sidecar"]["rate_hz"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "rate_hz"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_missing_camera_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        del bad["cameras"]["scene"]["device"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "device"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_empty_cameras_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["cameras"] = {}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "cameras"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_recording_config("/nonexistent/recording.yaml")

    def test_realsense_optional_absent_ok(self) -> None:
        # A map WITHOUT a realsense section stays valid (backward compatible).
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(_VALID, d)))
        self.assertNotIn("realsense", cfg)

    def test_realsense_present_loads(self) -> None:
        good = copy.deepcopy(_VALID)
        good["realsense"] = copy.deepcopy(_REALSENSE)
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(good, d)))
        self.assertEqual(cfg["realsense"]["rgb_name"], "central")
        self.assertEqual(cfg["realsense"]["depth_name"], "central_depth")

    def test_realsense_unknown_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["realsense"] = copy.deepcopy(_REALSENSE)
        bad["realsense"]["laser_power"] = 150
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "laser_power"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_realsense_missing_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["realsense"] = copy.deepcopy(_REALSENSE)
        del bad["realsense"]["depth_name"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "depth_name"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_checked_in_yaml_has_realsense(self) -> None:
        cfg = load_recording_config()
        self.assertIn("realsense", cfg)
        self.assertFalse(cfg["realsense"]["enabled"])  # default off

    def test_tactile_streams_are_the_gripper_names(self) -> None:
        # The four tactile cameras are named for the finger they sit on, and
        # those are the SAME names sensor_map.yaml binds to by-path nodes. The
        # recorder joins the two files by name, so a stream that appears in only
        # one of them is assigned-but-unrecordable (or configured with no
        # device) -- which is exactly what the old tactile_0..3 slots were.
        cfg = load_recording_config()
        for name in TACTILE_CAMERA_NAMES:
            self.assertIn(name, cfg["cameras"])
            self.assertTrue(cfg["cameras"][name]["enabled"], f"{name} enabled")
        for i in range(4):
            self.assertNotIn(f"tactile_{i}", cfg["cameras"])

    def test_every_assignable_name_is_a_configured_stream(self) -> None:
        # The guard on the split above, for all seven streams rather than the
        # four that caused it: a name the Signals tab can bind must be one the
        # recorder can actually open.
        cfg = load_recording_config()
        for name in ASSIGNABLE_CAMERA_NAMES:
            self.assertIn(name, cfg["cameras"], f"{name} missing from recording.yaml")

    def test_enabled_stream_may_not_have_device_minus_one(self) -> None:
        # -1 is the file's marker for "nothing wired here". Enabled, it used to
        # be accepted and then fail much later as an opaque camera-open error.
        bad = copy.deepcopy(_VALID)
        bad["cameras"]["scene"]["enabled"] = True
        bad["cameras"]["scene"]["device"] = -1
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "device -1"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_disabled_stream_may_have_device_minus_one(self) -> None:
        cfg = copy.deepcopy(_VALID)
        cfg["cameras"]["off"] = {
            **cfg["cameras"]["scene"],
            "enabled": False,
            "device": -1,
        }
        with tempfile.TemporaryDirectory() as d:
            loaded = load_recording_config(str(_write_yaml(cfg, d)))
        self.assertEqual(loaded["cameras"]["off"]["device"], -1)

    def test_checked_in_yaml_scene_off_central_on(self) -> None:
        # Current rig: no scene camera; the central overhead camera is a plain
        # UVC RGB stream recorded by default.
        cfg = load_recording_config()
        self.assertFalse(cfg["cameras"]["scene"]["enabled"])
        self.assertIn("central", cfg["cameras"])
        self.assertTrue(cfg["cameras"]["central"]["enabled"])

    def test_audio_optional_absent_ok(self) -> None:
        # A map WITHOUT an audio section stays valid (backward compatible).
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(_VALID, d)))
        self.assertNotIn("audio", cfg)

    def test_audio_present_loads(self) -> None:
        good = copy.deepcopy(_VALID)
        good["audio"] = copy.deepcopy(_AUDIO)
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(good, d)))
        self.assertEqual(cfg["audio"]["start_sound"], "start.wav")
        self.assertEqual(cfg["audio"]["stop_sound"], "stop.wav")

    def test_audio_unknown_key_raises(self) -> None:
        bad = copy.deepcopy(_VALID)
        bad["audio"] = copy.deepcopy(_AUDIO)
        bad["audio"]["volume"] = 0.5
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "volume"):
                load_recording_config(str(_write_yaml(bad, d)))

    def test_checked_in_yaml_has_audio(self) -> None:
        cfg = load_recording_config()
        self.assertIn("audio", cfg)
        self.assertTrue(cfg["audio"]["enabled"])

    def test_camera_fourcc_defaults_to_compressed(self) -> None:
        # A camera entry WITHOUT a fourcc key stays valid and is defaulted, so a
        # stream can never silently capture in an uncompressed format.
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(_VALID, d)))
        self.assertEqual(cfg["cameras"]["scene"]["fourcc"], "MJPG")

    def test_camera_fourcc_override_honoured(self) -> None:
        good = copy.deepcopy(_VALID)
        good["cameras"]["scene"]["fourcc"] = "YUYV"
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(good, d)))
        self.assertEqual(cfg["cameras"]["scene"]["fourcc"], "YUYV")

    def test_every_checked_in_camera_has_a_fourcc(self) -> None:
        # Every stream must resolve a format, so the record path can index it
        # unconditionally (tool/meta_quest_teleopration.py).
        cfg = load_recording_config()
        for name, cam in cfg["cameras"].items():
            self.assertTrue(cam["fourcc"], f"camera {name} has no fourcc")

    def test_camera_exposure_defaults_to_automatic(self) -> None:
        # Absent means "leave the camera's own automatic exposure alone", so an
        # older recording.yaml keeps behaving exactly as it did.
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(_VALID, d)))
        self.assertIsNone(cfg["cameras"]["scene"]["exposure"])

    def test_camera_exposure_override_honoured(self) -> None:
        good = copy.deepcopy(_VALID)
        good["cameras"]["scene"]["exposure"] = 300
        with tempfile.TemporaryDirectory() as d:
            cfg = load_recording_config(str(_write_yaml(good, d)))
        self.assertEqual(cfg["cameras"]["scene"]["exposure"], 300)

    def test_every_checked_in_camera_resolves_every_control(self) -> None:
        # The record path indexes these unconditionally, so each must be present
        # even when it is None (leave the camera's own setting alone).
        cfg = load_recording_config()
        for name, cam in cfg["cameras"].items():
            for control in CONTROL_NAMES:
                self.assertIn(control, cam, f"camera {name} has no {control}")

    def test_the_wrist_cameras_ship_with_exposure_pinned(self) -> None:
        # Automatic exposure is a frame-rate hazard on these two specifically:
        # a wrist camera looks at a close, shadowed workspace, so it lengthens
        # exposure past the frame period and the stream drops to 18 fps or less.
        # Measured: pinned at 300 they hold 27.4 fps regardless of the lighting.
        cfg = load_recording_config()
        for name in ("wrist_camera_left", "wrist_camera_right"):
            self.assertGreater(cfg["cameras"][name]["exposure"], 0, name)


if __name__ == "__main__":
    unittest.main()
