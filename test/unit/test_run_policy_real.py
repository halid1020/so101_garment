"""Unit tests for the pure action-translation helper of tool/run_policy_real.py.

No hardware and no policy: checks that a 12-D policy action reaches per-side
hardware goals through the same conversion a recorded action uses.
"""

import unittest

import numpy as np

from tool.run_policy_real import policy_action_to_goals

_BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class TestPolicyActionToGoals(unittest.TestCase):
    def test_splits_into_both_sides_with_gripper(self):
        # left body 0..4, left gripper (idx5), right body 6..10, right gripper (idx11)
        action = np.arange(12, dtype=float)
        goals = policy_action_to_goals(action)
        self.assertEqual(set(goals), {"left", "right"})
        for side in ("left", "right"):
            self.assertEqual(set(goals[side]), {*_BODY, "gripper"})

    def test_gripper_fraction_scaled_to_0_100(self):
        action = np.zeros(12)
        action[5] = 0.5  # left gripper fraction
        action[11] = 1.0  # right gripper fraction
        goals = policy_action_to_goals(action)
        self.assertAlmostEqual(goals["left"]["gripper"], 50.0)
        self.assertAlmostEqual(goals["right"]["gripper"], 100.0)

    def test_matches_manual_offset_sign_conversion(self):
        # With the shipped signs (+1) and offsets, hw = sign*(urdf - offset).
        from common.configs import LEFT_ARM_HW_TO_URDF_OFFSETS_DEG as OFF
        from common.configs import LEFT_ARM_HW_TO_URDF_SIGNS as SGN

        urdf = np.array([10.0, 20.0, -5.0, 0.0, 90.0])
        action = np.zeros(12)
        action[0:5] = urdf
        goals = policy_action_to_goals(action)
        expected = np.array(SGN) * (urdf - np.array(OFF))
        got = np.array([goals["left"][j] for j in _BODY])
        np.testing.assert_allclose(got, expected)


if __name__ == "__main__":
    unittest.main()
