"""Unit tests for the leader-arm joint mapping (common.threads.leader_arm).

Focus: the leader->follower URDF conversion, including the wrist-camera
framing offset that biases only the wrist_roll joint so the two input modes
(leader and Quest) frame the wrist camera the same way.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_leader_arm
"""

import unittest

from common.configs import (
    HANDLE_ROLL_OFFSET_DEG,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
)
from common.threads.leader_arm import (
    _BODY_JOINTS,
    leader_action_to_urdf,
    leader_gripper_to_trigger,
)

_WRIST_ROLL = _BODY_JOINTS.index("wrist_roll")


def _zero_action() -> dict:
    # Leader at its calibrated mid-range: every body joint reads 0 hw degrees.
    return {f"{j}.pos": 0.0 for j in _BODY_JOINTS}


class TestLeaderActionToUrdf(unittest.TestCase):
    def test_wrist_roll_carries_the_framing_offset(self) -> None:
        for side, offs in (
            ("left", LEFT_ARM_HW_TO_URDF_OFFSETS_DEG),
            ("right", RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG),
        ):
            urdf = leader_action_to_urdf(_zero_action(), side)
            # signs are +1 and hw is 0, so every joint equals its URDF offset,
            # EXCEPT wrist_roll which also carries the camera framing offset.
            self.assertAlmostEqual(
                urdf[_WRIST_ROLL], offs[_WRIST_ROLL] + HANDLE_ROLL_OFFSET_DEG
            )

    def test_other_joints_are_unbiased(self) -> None:
        urdf = leader_action_to_urdf(_zero_action(), "left")
        for i, name in enumerate(_BODY_JOINTS):
            if name == "wrist_roll":
                continue
            self.assertAlmostEqual(urdf[i], LEFT_ARM_HW_TO_URDF_OFFSETS_DEG[i])

    def test_default_offset_keeps_neutral_wrist_roll_in_band(self) -> None:
        # URDF wrist_roll limit is about -157..+163 deg; the default offset must
        # keep the neutral (hw 0) target comfortably inside it.
        urdf = leader_action_to_urdf(_zero_action(), "left")
        self.assertGreater(urdf[_WRIST_ROLL], -157.0)
        self.assertLess(urdf[_WRIST_ROLL], 162.0)


class TestLeaderGripperToTrigger(unittest.TestCase):
    def test_open_and_closed_extremes(self) -> None:
        self.assertAlmostEqual(leader_gripper_to_trigger(100.0), 0.0)  # open
        self.assertAlmostEqual(leader_gripper_to_trigger(0.0), 1.0)  # closed

    def test_clamped_to_unit_range(self) -> None:
        self.assertAlmostEqual(leader_gripper_to_trigger(150.0), 0.0)
        self.assertAlmostEqual(leader_gripper_to_trigger(-50.0), 1.0)


if __name__ == "__main__":
    unittest.main()
