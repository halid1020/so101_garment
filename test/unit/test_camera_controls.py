"""Unit tests for per-camera image controls and the tuner's YAML round-trip.

Two things are worth guarding. Applying controls has an ordering requirement
that is invisible in the code: an exposure value set while the camera is still
choosing its own is simply ignored, so automatic mode has to be switched off
first. And the tuner writes back into a configuration file whose comments carry
the reasoning for every setting, so it must edit lines rather than re-serialise.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_camera_controls
"""

import unittest

import cv2  # type: ignore[import]
import yaml  # type: ignore[import]

from common.camera_controls import (
    CONTROL_NAMES,
    EXPOSURE_MANUAL,
    apply_controls,
    control_yaml_line,
)
from tool.tune_cameras import parse_yaml_controls, write_yaml_controls

_YAML = """cameras:
  # The overhead view. Left automatic on purpose: it is well lit.
  central:
    enabled: true      # capture this stream
    device: 6
    width: 640
  # Left wrist. Pinned because automatic exposure slowed it to 18 fps.
  wrist_camera_left:
    enabled: true
    device: 2
    exposure: 300      # measured: holds 27.4 fps
  wrist_camera_right:
    enabled: true
    device: 4
    exposure: 150
dataset:
  fps: 30
"""


class _FakeCap:
    def __init__(self):
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True


class TestApplyControls(unittest.TestCase):
    def test_exposure_leaves_automatic_mode_before_the_value_is_sent(self):
        # Order is the whole point: a value set while the camera is still
        # choosing its own exposure has no effect at all.
        cap = _FakeCap()
        apply_controls(cap, {"exposure": 300})
        props = [p for p, _ in cap.calls]
        self.assertLess(
            props.index(cv2.CAP_PROP_AUTO_EXPOSURE), props.index(cv2.CAP_PROP_EXPOSURE)
        )
        self.assertEqual(cap.calls[0], (cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL))

    def test_none_leaves_a_control_alone(self):
        cap = _FakeCap()
        applied = apply_controls(cap, {name: None for name in CONTROL_NAMES})
        self.assertEqual(applied, [])
        self.assertEqual(cap.calls, [])

    def test_zero_is_a_real_value_not_an_absent_one(self):
        # Gain and brightness legitimately take 0, which is why absence is None.
        cap = _FakeCap()
        self.assertEqual(apply_controls(cap, {"gain": 0}), ["gain"])

    def test_other_controls_do_not_touch_the_exposure_mode(self):
        cap = _FakeCap()
        apply_controls(cap, {"gain": 5, "brightness": 10})
        self.assertNotIn(cv2.CAP_PROP_AUTO_EXPOSURE, [p for p, _ in cap.calls])

    def test_yaml_line_renders_absence_as_null(self):
        self.assertEqual(control_yaml_line("gain", None).strip(), "gain: null")
        self.assertEqual(control_yaml_line("gain", 30.0).strip(), "gain: 30")


class TestYamlRoundTrip(unittest.TestCase):
    def test_reads_the_values_of_one_camera(self):
        self.assertEqual(
            parse_yaml_controls(_YAML, "wrist_camera_left"), {"exposure": 300.0}
        )

    def test_a_camera_without_controls_reads_as_empty(self):
        self.assertEqual(parse_yaml_controls(_YAML, "central"), {})

    def test_writing_updates_only_the_named_camera(self):
        out = write_yaml_controls(_YAML, "wrist_camera_right", {"exposure": 120})
        self.assertEqual(
            parse_yaml_controls(out, "wrist_camera_right")["exposure"], 120
        )
        self.assertEqual(parse_yaml_controls(out, "wrist_camera_left")["exposure"], 300)

    def test_a_new_control_is_added_to_the_block(self):
        out = write_yaml_controls(_YAML, "wrist_camera_left", {"gain": 30})
        self.assertEqual(parse_yaml_controls(out, "wrist_camera_left")["gain"], 30)

    def test_comments_survive_the_write(self):
        # The file explains why every value is what it is; a YAML round-trip
        # through the parser would delete all of that silently.
        out = write_yaml_controls(_YAML, "wrist_camera_right", {"exposure": 120})
        for comment in (
            "# The overhead view. Left automatic on purpose: it is well lit.",
            "# Left wrist. Pinned because automatic exposure slowed it to 18 fps.",
            "# measured: holds 27.4 fps",
        ):
            self.assertIn(comment, out)

    def test_the_document_stays_valid_and_otherwise_unchanged(self):
        out = write_yaml_controls(_YAML, "wrist_camera_right", {"exposure": 120})
        before, after = yaml.safe_load(_YAML), yaml.safe_load(out)
        self.assertEqual(before["dataset"], after["dataset"])
        self.assertEqual(before["cameras"]["central"], after["cameras"]["central"])
        self.assertEqual(after["cameras"]["wrist_camera_right"]["device"], 4)

    def test_clearing_a_control_writes_null(self):
        out = write_yaml_controls(_YAML, "wrist_camera_left", {"exposure": None})
        self.assertIsNone(parse_yaml_controls(out, "wrist_camera_left")["exposure"])

    def test_an_unknown_camera_leaves_the_file_alone(self):
        self.assertEqual(write_yaml_controls(_YAML, "nope", {"exposure": 1}), _YAML)


if __name__ == "__main__":
    unittest.main()
