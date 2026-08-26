"""Unit tests for the per-USB-controller camera budget.

The rule these encode is measured, not theoretical. Every camera on the rig
reports bcdUSB 2.00 and no SuperSpeed capability, so each lands on a 480 Mbit/s
bus whose isochronous bandwidth the HOST CONTROLLER allocates across everything
on it -- a hub fans that bus out rather than adding to it. About three 640x480
MJPG streams fit, fewer when the expensive tactile cameras are among them. A
stream refused that bandwidth opens and then delivers nothing for ever, and WHICH
one loses is random -- so the point of this module is to warn about the wiring
before anything is opened.
"""

import unittest

from common.recording.usb_budget import (
    DEFAULT_PER_BUS_LIMIT,
    QUIRK_FIX_BANDWIDTH,
    budget_warnings,
    controller_of,
    group_by_controller,
    quirks_active,
    selection_warnings,
)

_BY_PATH = "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.1.2:1.0-video-index0"


class TestControllerOf(unittest.TestCase):
    def test_by_path_node_names_its_controller(self):
        self.assertEqual(controller_of(_BY_PATH), "pci-0000:05:00.4")

    def test_usbv2_spelling_is_the_same_controller(self):
        # /dev/v4l/by-path carries both spellings for one physical device.
        node = "/dev/v4l/by-path/pci-0000:05:00.4-usbv2-0:1.1.2:1.0-video-index0"
        self.assertEqual(controller_of(node), "pci-0000:05:00.4")

    def test_two_ports_on_one_controller_agree(self):
        a = "/dev/v4l/by-path/pci-0000:06:00.4-usb-0:1.1.3:1.0-video-index0"
        b = "/dev/v4l/by-path/pci-0000:06:00.4-usb-0:1.4:1.0-video-index0"
        self.assertEqual(controller_of(a), controller_of(b))

    def test_kernel_node_has_no_knowable_controller(self):
        # /dev/videoN numbering says nothing about which bus a device is on --
        # guessing from it would invent a budget that is not there.
        self.assertIsNone(controller_of("/dev/video4"))

    def test_index_and_none_are_unknown(self):
        self.assertIsNone(controller_of(4))
        self.assertIsNone(controller_of(None))


class TestGroupByController(unittest.TestCase):
    def _nodes(self):
        return {
            "central": "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.2:1.0-video-index0",
            "wrist_camera_left": (
                "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.1.4:1.0-video-index0"
            ),
            "left_arm_left_gripper": _BY_PATH,
            "wrist_camera_right": (
                "/dev/v4l/by-path/pci-0000:06:00.4-usb-0:1.4:1.0-video-index0"
            ),
        }

    def test_streams_are_grouped_by_their_bus(self):
        groups = group_by_controller(self._nodes())
        self.assertEqual(
            groups["pci-0000:05:00.4"],
            ["central", "left_arm_left_gripper", "wrist_camera_left"],
        )
        self.assertEqual(groups["pci-0000:06:00.4"], ["wrist_camera_right"])

    def test_unknowable_devices_are_left_out_not_pooled(self):
        # Pooling them under one placeholder would invent a crowded bus.
        groups = group_by_controller({"a": "/dev/video0", "b": 3, "c": None})
        self.assertEqual(groups, {})


class TestBudgetWarnings(unittest.TestCase):
    def test_a_bus_within_budget_is_silent(self):
        groups = {"pci-0000:05:00.4": ["a", "b", "c"]}
        self.assertEqual(budget_warnings(groups, per_bus_limit=3), [])

    def test_an_over_subscribed_bus_names_its_streams(self):
        groups = {"pci-0000:05:00.4": ["a", "b", "c", "d"]}
        (warning,) = budget_warnings(groups, per_bus_limit=3)
        self.assertIn("pci-0000:05:00.4", warning)
        for name in ("a", "b", "c", "d"):
            self.assertIn(name, warning)

    def test_each_over_subscribed_bus_warns_once(self):
        groups = {"bus-a": ["a", "b", "c"], "bus-b": ["d", "e", "f"]}
        self.assertEqual(len(budget_warnings(groups, per_bus_limit=2)), 2)

    def test_the_shipped_limit_is_the_measured_one(self):
        self.assertEqual(DEFAULT_PER_BUS_LIMIT, 3)


class TestSelectionWarnings(unittest.TestCase):
    def _nodes(self):
        five = {
            f"cam_{i}": (
                f"/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.{i}:1.0-video-index0"
            )
            for i in range(5)
        }
        return five

    def test_recording_fewer_streams_clears_the_warning(self):
        # The measured escape hatch: the four tactile cameras alone fit, so a
        # selection has to be judged, not just the wiring.
        nodes = self._nodes()
        self.assertTrue(selection_warnings(set(nodes), nodes))
        self.assertEqual(selection_warnings({"cam_0", "cam_1"}, nodes), [])

    def test_unselected_streams_do_not_count_against_the_bus(self):
        nodes = self._nodes()
        self.assertEqual(
            selection_warnings({"cam_0", "cam_1", "cam_2"}, nodes, per_bus_limit=3), []
        )


class TestQuirksActive(unittest.TestCase):
    def _write(self, text):
        import tempfile
        from pathlib import Path

        d = tempfile.mkdtemp()
        p = Path(d) / "quirks"
        p.write_text(text)
        return p

    def test_the_unset_sentinel_is_not_the_quirk_being_on(self):
        # The kernel reports an unset module parameter as all-ones, which does
        # have the quirk bit in it; reading that as "on" would be a lie.
        self.assertFalse(quirks_active(self._write("4294967295")))

    def test_the_bit_set_reads_as_on(self):
        self.assertTrue(quirks_active(self._write(str(QUIRK_FIX_BANDWIDTH))))

    def test_another_quirk_alone_is_not_this_one(self):
        self.assertFalse(quirks_active(self._write("2")))

    def test_unreadable_is_unknown_not_off(self):
        # "no uvcvideo module here" is not "the quirk is off".
        self.assertIsNone(quirks_active("/nonexistent/quirks"))
        self.assertIsNone(quirks_active(self._write("not a number")))


if __name__ == "__main__":
    unittest.main()
