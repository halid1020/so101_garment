"""Unit tests for the pure helpers of tool/test_sensor_rates.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool.test_sensor_rates import (
    grid_tiles,
    load_sensor_map,
    parse_camera_spec,
    save_sensor_map,
    stable_device_path,
)


class TestParseCameraSpec(unittest.TestCase):
    def test_named_spec(self):
        name, dev = parse_camera_spec("left_arm_left_gripper=/dev/video4")
        self.assertEqual(name, "left_arm_left_gripper")
        self.assertEqual(dev, "/dev/video4")

    def test_bare_index_gets_auto_name(self):
        name, dev = parse_camera_spec("4")
        self.assertEqual(name, "camera[4]")
        self.assertEqual(dev, 4)

    def test_bare_path_gets_auto_name(self):
        name, dev = parse_camera_spec("/dev/video6")
        self.assertEqual(name, "camera[/dev/video6]")
        self.assertEqual(dev, "/dev/video6")

    def test_named_index_is_int(self):
        _, dev = parse_camera_spec("cam=6")
        self.assertEqual(dev, 6)

    def test_bad_specs_raise(self):
        for spec in ("=", "name=", "=dev", ""):
            with self.assertRaises(ValueError):
                parse_camera_spec(spec)


class TestSensorMapRoundTrip(unittest.TestCase):
    def test_save_then_load(self):
        sensor_map = {
            "cameras": {"left_arm_left_gripper": "/dev/video4"},
            "arms": {"right": "/dev/ttyACM0", "left": "/dev/ttyACM1"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            save_sensor_map(path, sensor_map)
            self.assertEqual(load_sensor_map(path), sensor_map)

    def test_load_tolerates_missing_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            path.write_text("cameras:\n  a: /dev/video0\n")
            loaded = load_sensor_map(path)
            self.assertEqual(loaded["cameras"], {"a": "/dev/video0"})
            self.assertEqual(loaded["arms"], {})


class TestGridTiles(unittest.TestCase):
    def _tile(self, h=10, w=20):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_single_row(self):
        out = grid_tiles([self._tile(), self._tile()], max_per_row=3)
        self.assertEqual(out.shape, (10, 40, 3))

    def test_wraps_and_pads_last_row(self):
        out = grid_tiles([self._tile()] * 4, max_per_row=3)
        # 3 tiles on the first row (60 wide), 1 padded on the second.
        self.assertEqual(out.shape, (20, 60, 3))

    def test_mixed_heights_align_within_row(self):
        out = grid_tiles([self._tile(h=10), self._tile(h=20)], max_per_row=3)
        self.assertEqual(out.shape[0], 20)


class TestStableDevicePath(unittest.TestCase):
    def test_prefers_by_path_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "ttyACM0"
            node.write_text("")
            by_path = root / "serial" / "by-path"
            by_path.mkdir(parents=True)
            link = by_path / "pci-usb-0:1.2:1.0"
            link.symlink_to(node)
            self.assertEqual(stable_device_path(str(node), dev_root=root), str(link))

    def test_falls_back_to_raw_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "video4"
            node.write_text("")
            self.assertEqual(
                stable_device_path(str(node), dev_root=Path(tmp)), str(node)
            )

    def test_missing_node_returned_verbatim(self):
        self.assertEqual(
            stable_device_path("/nonexistent/devnode", dev_root="/nonexistent"),
            "/nonexistent/devnode",
        )


if __name__ == "__main__":
    unittest.main()
