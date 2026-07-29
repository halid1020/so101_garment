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

from common.config_parser import load_recording_config

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
        for name in ("scene", "wrist_left", "wrist_right"):
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

    def test_tactile_default_disabled(self) -> None:
        cfg = load_recording_config()
        for i in range(4):
            name = f"tactile_{i}"
            self.assertIn(name, cfg["cameras"])
            self.assertFalse(
                cfg["cameras"][name]["enabled"],
                f"{name} must default to disabled (hardware not attached)",
            )


if __name__ == "__main__":
    unittest.main()
