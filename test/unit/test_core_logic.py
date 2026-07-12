#!/usr/bin/env python3
"""
Unit tests for hardware-independent mathematical and filtering utilities.
Run via: python3 -m unittest test/test_core_logic.py
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.common.one_euro_filter import OneEuroFilter
from src.common.utils import scale_and_add_delta_transform


class TestCoreLogic(unittest.TestCase):
    def test_one_euro_filter_smoothing(self):
        """Verify the 1€ filter properly smooths noisy scalar signals."""
        filter_1e = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)

        # Inject a sudden spike (noise)
        noisy_val = 10.0
        smoothed_val = filter_1e(timestamp=0.1, x=noisy_val)

        # The smoothed value should resist the sudden jump
        self.assertLess(smoothed_val, noisy_val)
        self.assertGreater(smoothed_val, 0.0)

    def test_scale_and_add_delta_transform(self):
        """Verify Cartesian deltas are scaled and applied correctly to a base transform."""
        initial_transform = np.eye(4)

        # Move 1 unit in X, rotate 90 degrees around Z
        delta_position = np.array([1.0, 0.0, 0.0])
        delta_orientation = Rotation.from_euler("z", 90, degrees=True).as_matrix()

        # Apply a 2.0x translation scale, and 0.5x rotation scale
        target_transform = scale_and_add_delta_transform(
            delta_position=delta_position,
            delta_orientation=delta_orientation,
            translation_scale=2.0,
            rotation_scale=0.5,
            initial_transform=initial_transform,
        )

        # Verify position was doubled (1.0 * 2.0 = 2.0)
        self.assertAlmostEqual(target_transform[0, 3], 2.0)
        self.assertAlmostEqual(target_transform[1, 3], 0.0)
        self.assertAlmostEqual(target_transform[2, 3], 0.0)

        # Verify rotation was halved (90 * 0.5 = 45 degrees)
        resulting_rot = Rotation.from_matrix(target_transform[:3, :3])
        euler_angles = resulting_rot.as_euler("xyz", degrees=True)
        self.assertAlmostEqual(euler_angles[2], 45.0)


class TestOperatorFrame(unittest.TestCase):
    """Headset-anywhere invariance of the operator control frame."""

    @staticmethod
    def _hand_pair():
        left = np.eye(4)
        left[:3, 3] = [0.35, 0.15, -0.25]
        left[:3, :3] = Rotation.from_euler(
            "xyz", [5, -10, 20], degrees=True
        ).as_matrix()
        right = np.eye(4)
        right[:3, 3] = [0.32, -0.17, -0.22]
        right[:3, :3] = Rotation.from_euler(
            "xyz", [-8, 6, -15], degrees=True
        ).as_matrix()
        return left, right

    def test_invariant_to_headset_yaw_and_position(self):
        """Yawing/translating the reader frame (i.e. placing the headset
        anywhere) must not change the operator-frame hand poses."""
        from src.common.utils import compute_operator_frame, to_operator_frame

        left, right = self._hand_pair()
        rot0, org0 = compute_operator_frame(left, right)
        ref_left = to_operator_frame(left, rot0, org0)
        ref_right = to_operator_frame(right, rot0, org0)

        for yaw_deg, offset in ((45, [1.0, -2.0, 0.3]), (-120, [0.2, 5.0, -1.0])):
            world_shift = np.eye(4)
            world_shift[:3, :3] = Rotation.from_euler(
                "z", yaw_deg, degrees=True
            ).as_matrix()
            world_shift[:3, 3] = offset
            left_s = world_shift @ left
            right_s = world_shift @ right
            rot_s, org_s = compute_operator_frame(left_s, right_s)
            np.testing.assert_allclose(
                to_operator_frame(left_s, rot_s, org_s), ref_left, atol=1e-9
            )
            np.testing.assert_allclose(
                to_operator_frame(right_s, rot_s, org_s), ref_right, atol=1e-9
            )

    def test_frame_axes_and_origin(self):
        """y axis follows the left-right handle line, z is up, and the
        origin sits 20 cm behind and above the handle midpoint."""
        from src.common.utils import (
            OPERATOR_FRAME_BACK_M,
            OPERATOR_FRAME_UP_M,
            compute_operator_frame,
        )

        left, right = self._hand_pair()
        rot, org = compute_operator_frame(left, right)
        # Right-handed, unit, z-up.
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rot[:, 2], [0, 0, 1], atol=1e-12)
        self.assertGreater(rot[:2, 1] @ (left[:2, 3] - right[:2, 3]), 0.0)
        midpoint = 0.5 * (left[:3, 3] + right[:3, 3])
        expected = (
            midpoint
            - OPERATOR_FRAME_BACK_M * rot[:, 0]
            + OPERATOR_FRAME_UP_M * rot[:, 2]
        )
        np.testing.assert_allclose(org, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
