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


if __name__ == "__main__":
    unittest.main()
