"""Unit tests for the leader-arm joint mapping (common.threads.leader_arm).

Focus: the leader->follower URDF conversion, including the wrist-camera
framing offset that biases only the wrist_roll joint so the two input modes
(leader and Quest) frame the wrist camera the same way.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_leader_arm
"""

import math
import unittest

import numpy as np

from common.configs import (
    HANDLE_ROLL_OFFSET_DEG,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
)
from common.threads.leader_arm import (
    _BODY_JOINTS,
    leader_action_to_urdf,
    leader_gripper_to_trigger,
    slew_toward,
    velocity_step_deg,
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


class TestVelocityStepDeg(unittest.TestCase):
    """The slew budget must be a velocity, not a per-tick allowance."""

    MAX_VEL = 2.0  # rad/s, the real-robot limit
    NOMINAL = 0.01  # 100 Hz
    CATCHUP = 5 * NOMINAL

    def test_nominal_tick_matches_the_configured_velocity(self) -> None:
        step = velocity_step_deg(self.NOMINAL, self.MAX_VEL, self.CATCHUP)
        self.assertAlmostEqual(step, math.degrees(self.MAX_VEL) * self.NOMINAL)

    def test_a_slow_tick_gets_proportionally_more_budget(self) -> None:
        # The regression this guards: a starved loop used to keep the 10 ms
        # budget per tick, so the followers tracked at a fraction of the limit.
        fast = velocity_step_deg(self.NOMINAL, self.MAX_VEL, self.CATCHUP)
        slow = velocity_step_deg(2 * self.NOMINAL, self.MAX_VEL, self.CATCHUP)
        self.assertAlmostEqual(slow, 2 * fast)

    def test_velocity_is_constant_across_tick_lengths(self) -> None:
        # deg/s implied by the budget must not depend on the loop rate.
        for elapsed in (0.005, 0.01, 0.02, 0.04):
            implied = velocity_step_deg(elapsed, self.MAX_VEL, self.CATCHUP) / elapsed
            self.assertAlmostEqual(implied, math.degrees(self.MAX_VEL), places=6)

    def test_a_long_stall_cannot_be_cashed_in_as_one_jump(self) -> None:
        # Safety: the clamp keeps a 1 s stall from authorising a 114 deg step.
        step = velocity_step_deg(1.0, self.MAX_VEL, self.CATCHUP)
        self.assertAlmostEqual(step, math.degrees(self.MAX_VEL) * self.CATCHUP)
        self.assertLess(step, math.degrees(self.MAX_VEL))

    def test_negative_or_zero_elapsed_yields_no_motion(self) -> None:
        self.assertEqual(velocity_step_deg(0.0, self.MAX_VEL, self.CATCHUP), 0.0)
        self.assertEqual(velocity_step_deg(-0.5, self.MAX_VEL, self.CATCHUP), 0.0)

    def test_slew_reaches_the_target_faster_when_ticks_are_slow(self) -> None:
        # End to end over the two helpers: at half the loop rate, the same wall
        # time must cover the same angle.
        target = np.array([30.0])
        fast_cmd = np.array([0.0])
        for _ in range(20):  # 20 ticks of 10 ms = 200 ms
            fast_cmd = slew_toward(
                fast_cmd,
                target,
                velocity_step_deg(self.NOMINAL, self.MAX_VEL, self.CATCHUP),
            )
        slow_cmd = np.array([0.0])
        for _ in range(10):  # 10 ticks of 20 ms = the same 200 ms
            slow_cmd = slew_toward(
                slow_cmd,
                target,
                velocity_step_deg(2 * self.NOMINAL, self.MAX_VEL, self.CATCHUP),
            )
        self.assertAlmostEqual(float(fast_cmd[0]), float(slow_cmd[0]), places=6)


if __name__ == "__main__":
    unittest.main()
