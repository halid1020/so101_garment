"""The one piece of arithmetic between a policy action and the 3D twin.

The scene itself is viser's problem and the browser's; what this module owns is
the claim that a 12-D action and the URDF's twelve actuated joints mean the same
thing in the same order. Get that wrong and the ghost is confidently wrong --
the worst kind of preview -- so it is checked here, without a server.
"""

import unittest

import numpy as np

from common.configs import GRIPPER_OPEN_MAX_FRAC
from common.web.policy_ghost import GRIP_HI, GRIP_LO, action_to_urdf_cfg


class TestActionToUrdfCfg(unittest.TestCase):
    def action(self, left=0.0, right=0.0, lgrip=0.0, rgrip=0.0):
        return np.array([left] * 5 + [lgrip] + [right] * 5 + [rgrip], dtype=float)

    def test_the_arm_channels_are_the_same_angles_in_radians(self):
        cfg = action_to_urdf_cfg(self.action(left=90.0, right=-45.0))

        np.testing.assert_allclose(cfg[0:5], np.radians(90.0))
        np.testing.assert_allclose(cfg[6:11], np.radians(-45.0))

    def test_the_layout_puts_each_gripper_after_its_own_arm(self):
        # This is the whole reason the helper exists: the URDF's actuated order
        # is left5, left_gripper, right5, right_gripper -- and so is the action.
        cfg = action_to_urdf_cfg(self.action(lgrip=GRIPPER_OPEN_MAX_FRAC))

        self.assertEqual(len(cfg), 12)
        self.assertAlmostEqual(cfg[5], GRIP_HI)
        self.assertAlmostEqual(cfg[11], GRIP_LO)

    def test_a_shut_gripper_sits_at_the_joint_s_lower_limit(self):
        cfg = action_to_urdf_cfg(self.action(lgrip=0.0, rgrip=0.0))

        self.assertAlmostEqual(cfg[5], GRIP_LO)
        self.assertAlmostEqual(cfg[11], GRIP_LO)

    def test_a_gripper_half_open_sits_halfway_along_its_travel(self):
        cfg = action_to_urdf_cfg(self.action(lgrip=GRIPPER_OPEN_MAX_FRAC / 2))

        self.assertAlmostEqual(cfg[5], GRIP_LO + 0.5 * (GRIP_HI - GRIP_LO))

    def test_a_gripper_beyond_its_cap_is_clipped_not_extrapolated(self):
        # An out-of-range action must not push the ghost through the mesh: a
        # preview that shows an impossible pose is worse than none.
        cfg = action_to_urdf_cfg(self.action(lgrip=99.0, rgrip=-99.0))

        self.assertAlmostEqual(cfg[5], GRIP_HI)
        self.assertAlmostEqual(cfg[11], GRIP_LO)

    def test_the_limits_are_taken_from_the_caller_not_from_this_module(self):
        # GhostPair reads them off the loaded model, so a URDF change follows
        # through without anyone remembering to edit a constant here.
        cfg = action_to_urdf_cfg(
            self.action(lgrip=GRIPPER_OPEN_MAX_FRAC), grip_lo=-1.0, grip_hi=2.0
        )

        self.assertAlmostEqual(cfg[5], 2.0)
        self.assertAlmostEqual(cfg[11], -1.0)


if __name__ == "__main__":
    unittest.main()
