"""Unit tests for USB hub grouping and the preflight topology verdict.

The grouping model is the load-bearing part: devices are grouped by controller
and root port, so a nested hub shares the fate of the hub above it and a USB3
hub's two sysfs faces count as the one piece of hardware they are. The reference
case is a real failure on this rig, where a single hub took a drive, two cameras
and two serial buses with it.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_usb_topology
"""

import unittest

from actoris_harena.recording.usb_topology import parse_sysfs_path

from tool.collect_preflight import FAIL, OK, WARN, classify_usb_topology

_PCI = "/sys/devices/pci0000:00/0000:00:08.1"


def _at(chain, controller="0000:05:00.4"):
    """A sysfs path for a device at ``chain``, e.g. "3-1.1.4"."""
    bus = chain.split("-")[0]
    hops = chain.split("-")[1].split(".")
    parts = [f"{bus}-{'.'.join(hops[: i + 1])}" for i in range(len(hops))]
    return f"{_PCI}/{controller}/usb{bus}/" + "/".join(parts) + f"/{chain}:1.0"


class TestClassifyUsbTopology(unittest.TestCase):
    """The verdict an operator reads before a session."""

    def _rig(self, drive_chain, drive_controller="0000:05:00.4"):
        storage = {
            "dataset drive /mnt/x": parse_sysfs_path(_at(drive_chain, drive_controller))
        }
        sensors = {
            "camera central": parse_sysfs_path(_at("3-1.2")),
            "camera wrist_left": parse_sysfs_path(_at("3-1.1.4")),
            "follower left": parse_sysfs_path(_at("3-1.4")),
            "camera wrist_right": parse_sysfs_path(_at("3-1.4", "0000:06:00.4")),
        }
        return storage, sensors

    def test_drive_sharing_a_hub_with_the_rig_fails(self):
        # The real failure: the drive on 4-1.3 is the same hub as the sensors on
        # 3-1.*, so one fault removes the file being written to.
        result = classify_usb_topology(*self._rig("4-1.3"))
        self.assertEqual(result.level, FAIL)
        self.assertIn("dataset drive", result.detail)

    def test_the_verdict_lists_who_shares_the_hub(self):
        result = classify_usb_topology(*self._rig("4-1.3"))
        for label in ("camera central", "camera wrist_left", "follower left"):
            self.assertIn(label, result.detail)

    def test_drive_on_another_controller_only_warns_about_the_sensors(self):
        result = classify_usb_topology(*self._rig("2-1.1", "0000:07:00.0"))
        self.assertEqual(result.level, WARN)
        self.assertIn("sensors share a hub", result.detail)

    def test_nothing_shared_is_ok(self):
        storage = {"drive": parse_sysfs_path(_at("2-1", "0000:07:00.0"))}
        sensors = {"camera central": parse_sysfs_path(_at("3-1.2"))}
        self.assertEqual(classify_usb_topology(storage, sensors).level, OK)

    def test_nothing_resolvable_warns_rather_than_claiming_safety(self):
        result = classify_usb_topology({"drive": None}, {"camera central": None})
        self.assertEqual(result.level, WARN)
        self.assertIn("cannot judge", result.detail)

    def test_an_unresolvable_drive_does_not_mask_sensor_sharing(self):
        _, sensors = self._rig("4-1.3")
        result = classify_usb_topology({"drive": None}, sensors)
        self.assertEqual(result.level, WARN)


if __name__ == "__main__":
    unittest.main()
