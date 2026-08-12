"""Unit tests for the pure helpers of tool/test_sensor_rates.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool.test_sensor_rates import (
    _drop_serial_node,
    _serial_node,
    grid_tiles,
    load_sensor_map,
    parse_camera_spec,
    save_sensor_map,
    select_realsense_serial,
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
        # Leaders now store only their port; the calibration id is fixed by
        # side in robot.yaml (LEADER_ID_LEFT/RIGHT), not in the map.
        sensor_map = {
            "cameras": {"left_arm_left_gripper": "/dev/video4"},
            "arms": {"right": "/dev/ttyACM0", "left": "/dev/ttyACM1"},
            "leaders": {
                "right": {"port": "/dev/ttyACM2"},
                "left": {"port": "/dev/ttyACM3"},
            },
            "realsense": {"serial": "ABC123", "name": "central"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            save_sensor_map(path, sensor_map)
            self.assertEqual(load_sensor_map(path), sensor_map)

    def test_legacy_leader_id_is_tolerated(self):
        # A map written before the id-free assignment change still loads;
        # the extra "id" key is preserved but consumers ignore it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            path.write_text(
                "arms: {right: /dev/ttyACM0, left: /dev/ttyACM1}\n"
                "leaders:\n  right: {port: /dev/ttyACM2, id: leader_0}\n"
            )
            loaded = load_sensor_map(path)
            self.assertEqual(loaded["leaders"]["right"]["port"], "/dev/ttyACM2")

    def test_save_without_leaders_key_defaults_empty(self):
        # Old callers may build a map without a leaders section.
        sensor_map = {"cameras": {}, "arms": {"right": "/dev/ttyACM0"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            save_sensor_map(path, sensor_map)
            self.assertEqual(load_sensor_map(path)["leaders"], {})

    def test_realsense_round_trips(self):
        sensor_map = {
            "cameras": {},
            "arms": {},
            "leaders": {},
            "realsense": {"serial": "ABC123", "name": "central"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            save_sensor_map(path, sensor_map)
            self.assertEqual(
                load_sensor_map(path)["realsense"], sensor_map["realsense"]
            )

    def test_old_map_without_realsense_defaults_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            path.write_text("cameras:\n  a: /dev/video0\n")
            self.assertEqual(load_sensor_map(path)["realsense"], {})

    def test_load_tolerates_missing_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            path.write_text("cameras:\n  a: /dev/video0\n")
            loaded = load_sensor_map(path)
            self.assertEqual(loaded["cameras"], {"a": "/dev/video0"})
            self.assertEqual(loaded["arms"], {})
            self.assertEqual(loaded["leaders"], {})


class TestDropSerialNode(unittest.TestCase):
    def _map(self, tmp):
        # Real files so Path.resolve() is stable across roles.
        for name in ("f0", "f1", "l0", "l1"):
            (Path(tmp) / name).write_text("")
        return {
            "cameras": {},
            "arms": {
                "right": str(Path(tmp) / "f0"),
                "left": str(Path(tmp) / "f1"),
            },
            "leaders": {
                "right": {"port": str(Path(tmp) / "l0"), "id": "leader_0"},
                "left": {"port": str(Path(tmp) / "l1"), "id": "leader_1"},
            },
        }

    def test_serial_node_reads_both_shapes(self):
        self.assertEqual(_serial_node("/dev/ttyACM0"), "/dev/ttyACM0")
        self.assertEqual(
            _serial_node({"port": "/dev/ttyACM2", "id": "x"}), "/dev/ttyACM2"
        )

    def test_drop_removes_follower_node_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._map(tmp)
            _drop_serial_node(m, str(Path(tmp) / "f0"))
            self.assertNotIn("right", m["arms"])
            self.assertIn("left", m["arms"])
            self.assertEqual(set(m["leaders"]), {"right", "left"})

    def test_drop_removes_leader_node_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._map(tmp)
            _drop_serial_node(m, str(Path(tmp) / "l1"))
            self.assertNotIn("left", m["leaders"])
            self.assertIn("right", m["leaders"])
            self.assertEqual(set(m["arms"]), {"right", "left"})

    def test_same_side_across_roles_is_independent(self):
        # A follower-right and a leader-right are different physical arms;
        # dropping the follower node must not touch the leader-right entry.
        with tempfile.TemporaryDirectory() as tmp:
            m = self._map(tmp)
            _drop_serial_node(m, str(Path(tmp) / "f0"))  # follower right
            self.assertIn("right", m["leaders"])


class TestDiscoverLeaderCalibIds(unittest.TestCase):
    def test_lists_sorted_json_stems(self):
        import common.follower_bus as fb

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leader_1.json").write_text("{}")
            (Path(tmp) / "leader_0.json").write_text("{}")
            (Path(tmp) / "notes.txt").write_text("")  # ignored
            orig = fb._leader_calib_dir
            fb._leader_calib_dir = lambda: Path(tmp)
            try:
                self.assertEqual(
                    fb.discover_leader_calib_ids(), ["leader_0", "leader_1"]
                )
            finally:
                fb._leader_calib_dir = orig

    def test_missing_dir_returns_empty(self):
        import common.follower_bus as fb

        orig = fb._leader_calib_dir
        fb._leader_calib_dir = lambda: Path("/nonexistent/so_leader")
        try:
            self.assertEqual(fb.discover_leader_calib_ids(), [])
        finally:
            fb._leader_calib_dir = orig


class TestLeaderCalibIdForSide(unittest.TestCase):
    def test_reads_side_ids_from_robot_yaml(self):
        from common.follower_bus import leader_calib_id_for_side

        # The locked convention (src/conf/robot.yaml LEADER_ID_LEFT/RIGHT).
        self.assertEqual(leader_calib_id_for_side("left"), "leader_left")
        self.assertEqual(leader_calib_id_for_side("right"), "leader_right")

    def test_invalid_side_raises(self):
        from common.follower_bus import leader_calib_id_for_side

        with self.assertRaises(ValueError):
            leader_calib_id_for_side("middle")


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


class TestSelectRealsenseSerial(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(select_realsense_serial([]))

    def test_single_device(self):
        self.assertEqual(select_realsense_serial([("S1", "D435")]), "S1")

    def test_multiple_prefers_previous(self):
        devs = [("S1", "D435"), ("S2", "D455")]
        self.assertEqual(select_realsense_serial(devs, prefer="S2"), "S2")

    def test_multiple_without_valid_prefer_takes_first(self):
        devs = [("S1", "D435"), ("S2", "D455")]
        self.assertEqual(select_realsense_serial(devs, prefer="GONE"), "S1")
        self.assertEqual(select_realsense_serial(devs), "S1")


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
