"""Unit tests for the camera tuner's YAML round-trip.

The tuner writes back into a configuration file whose comments carry the
reasoning for every setting, so it must edit lines rather than re-serialise.

Applying the controls themselves is tested in actoris_harena
(test/rig/test_camera_controls.py), where that module now lives; only
tool/tune_cameras.py is this repo's.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_camera_controls
"""

import unittest

import yaml  # type: ignore[import]

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
