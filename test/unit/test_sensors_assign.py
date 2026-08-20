"""Unit tests for binding devices to stream names from the console.

No hardware: the map operations are pure dictionary work, and that is where the
mistakes that matter live -- a camera reassigned to a second name while still
listed under the first, or a serial port that ends up being both a follower and
a leader. The wiggle-test arithmetic and the "is this device still here" check
are tested too, because they are what tells an operator that an assignment has
gone stale.
"""

import tempfile
import unittest
from pathlib import Path

from common.web.sensors import (
    ARM_SIDES,
    ASSIGNABLE_CAMERA_NAMES,
    assign_arm,
    assign_camera,
    assign_realsense,
    clear_assignment,
    map_overview,
    read_map,
    tick_rows,
    write_map,
)

EMPTY: dict = {"cameras": {}, "arms": {}, "leaders": {}, "realsense": {}}


class TestCameraAssignment(unittest.TestCase):
    def test_a_name_is_bound_to_a_device(self):
        out = assign_camera(EMPTY, "wrist_camera_left", "/dev/v4l/by-path/a")
        self.assertEqual(out["cameras"]["wrist_camera_left"], "/dev/v4l/by-path/a")

    def test_rebinding_a_device_clears_the_name_it_had(self):
        out = assign_camera(EMPTY, "wrist_camera_left", "/dev/v4l/by-path/a")
        out = assign_camera(out, "wrist_camera_right", "/dev/v4l/by-path/a")
        self.assertNotIn("wrist_camera_left", out["cameras"])
        self.assertEqual(out["cameras"]["wrist_camera_right"], "/dev/v4l/by-path/a")

    def test_two_devices_keep_their_own_names(self):
        out = assign_camera(EMPTY, "wrist_camera_left", "/dev/v4l/by-path/a")
        out = assign_camera(out, "wrist_camera_right", "/dev/v4l/by-path/b")
        self.assertEqual(len(out["cameras"]), 2)

    def test_an_unknown_stream_name_is_refused(self):
        with self.assertRaises(ValueError):
            assign_camera(EMPTY, "not_a_camera", "/dev/video0")

    def test_the_source_map_is_not_mutated(self):
        assign_camera(EMPTY, "central", "/dev/video0")
        self.assertEqual(EMPTY["cameras"], {})


class TestArmAssignment(unittest.TestCase):
    def test_a_follower_side_is_bound(self):
        out = assign_arm(EMPTY, "follower", "right", "/dev/serial/by-path/a")
        self.assertEqual(out["arms"]["right"], "/dev/serial/by-path/a")

    def test_a_leader_stores_only_its_port(self):
        out = assign_arm(EMPTY, "leader", "left", "/dev/serial/by-path/b")
        self.assertEqual(out["leaders"]["left"], {"port": "/dev/serial/by-path/b"})

    def test_one_port_cannot_be_two_arms(self):
        out = assign_arm(EMPTY, "follower", "right", "/dev/serial/by-path/a")
        out = assign_arm(out, "leader", "left", "/dev/serial/by-path/a")
        self.assertEqual(out["arms"], {})
        self.assertEqual(out["leaders"]["left"]["port"], "/dev/serial/by-path/a")

    def test_a_follower_and_a_leader_may_share_a_side(self):
        out = assign_arm(EMPTY, "follower", "right", "/dev/serial/by-path/a")
        out = assign_arm(out, "leader", "right", "/dev/serial/by-path/b")
        self.assertEqual(out["arms"]["right"], "/dev/serial/by-path/a")
        self.assertEqual(out["leaders"]["right"]["port"], "/dev/serial/by-path/b")

    def test_an_unknown_role_or_side_is_refused(self):
        with self.assertRaises(ValueError):
            assign_arm(EMPTY, "gripper", "right", "/dev/x")
        with self.assertRaises(ValueError):
            assign_arm(EMPTY, "follower", "middle", "/dev/x")


class TestClearing(unittest.TestCase):
    def setUp(self):
        m = assign_camera(EMPTY, "central", "/dev/v4l/by-path/a")
        m = assign_arm(m, "follower", "right", "/dev/serial/by-path/b")
        m = assign_arm(m, "leader", "left", "/dev/serial/by-path/c")
        self.map = assign_realsense(m, "0123")

    def test_clearing_each_kind(self):
        self.assertEqual(clear_assignment(self.map, "camera", "central")["cameras"], {})
        self.assertEqual(clear_assignment(self.map, "follower", "right")["arms"], {})
        self.assertEqual(clear_assignment(self.map, "leader", "left")["leaders"], {})
        self.assertEqual(clear_assignment(self.map, "realsense", "")["realsense"], {})

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            clear_assignment(self.map, "hub", "x")


class TestOverview(unittest.TestCase):
    def test_every_assignable_name_is_listed_even_when_unassigned(self):
        view = map_overview(EMPTY, [], [], [])
        self.assertEqual(len(view["cameras"]), len(ASSIGNABLE_CAMERA_NAMES))
        self.assertEqual(len(view["arms"]), 2 * len(ARM_SIDES))
        self.assertTrue(all(c["device"] is None for c in view["cameras"]))

    def test_an_assignment_to_a_connected_device_reads_as_present(self):
        m = assign_camera(EMPTY, "central", "/dev/video0")
        view = map_overview(m, ["/dev/video0"], [], [])
        central = next(c for c in view["cameras"] if c["name"] == "central")
        self.assertTrue(central["present"])

    def test_an_assignment_whose_device_is_gone_reads_as_absent(self):
        m = assign_camera(EMPTY, "central", "/dev/v4l/by-path/moved")
        view = map_overview(m, ["/dev/video0"], [], [])
        central = next(c for c in view["cameras"] if c["name"] == "central")
        self.assertFalse(central["present"])

    def test_the_depth_camera_is_matched_by_serial(self):
        m = assign_realsense(EMPTY, "0123")
        self.assertTrue(
            map_overview(m, [], [], [("0123", "D435")])["realsense"]["present"]
        )
        self.assertFalse(
            map_overview(m, [], [], [("9999", "D435")])["realsense"]["present"]
        )

    def test_unassigned_connected_cameras_are_listed(self):
        m = assign_camera(EMPTY, "central", "/dev/video0")
        view = map_overview(m, ["/dev/video0", "/dev/video2"], [], [])
        self.assertEqual(view["unassigned_cameras"], ["/dev/video2"])


class TestWiggleTest(unittest.TestCase):
    def test_a_joint_that_has_not_moved_is_quiet(self):
        rows = tick_rows({"shoulder_pan": 2048}, {"shoulder_pan": 2048})
        self.assertEqual(rows[0]["delta"], 0)
        self.assertFalse(rows[0]["moving"])

    def test_read_noise_does_not_count_as_movement(self):
        rows = tick_rows({"shoulder_pan": 2051}, {"shoulder_pan": 2048})
        self.assertFalse(rows[0]["moving"])

    def test_a_wiggled_joint_shows_up(self):
        rows = tick_rows({"elbow_flex": 1800}, {"elbow_flex": 2048})
        self.assertTrue(rows[0]["moving"])
        self.assertEqual(rows[0]["delta"], -248)

    def test_without_a_baseline_nothing_has_moved_yet(self):
        rows = tick_rows({"wrist_roll": 900}, None)
        self.assertEqual(rows[0]["delta"], 0)


class TestMapFile(unittest.TestCase):
    def test_a_map_round_trips_through_the_shared_loader_and_saver(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_map.yaml"
            m = assign_camera(EMPTY, "central", "/dev/v4l/by-path/a")
            m = assign_arm(m, "follower", "right", "/dev/serial/by-path/b")
            m = assign_realsense(m, "0123")
            write_map(m, path)
            back = read_map(path)
            self.assertEqual(back["cameras"]["central"], "/dev/v4l/by-path/a")
            self.assertEqual(back["arms"]["right"], "/dev/serial/by-path/b")
            self.assertEqual(back["realsense"]["serial"], "0123")

    def test_a_missing_map_reads_as_nothing_assigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_map(Path(tmp) / "absent.yaml"), EMPTY)


if __name__ == "__main__":
    unittest.main()
