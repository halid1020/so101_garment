#!/usr/bin/env python3
"""Unit tests for the sidecar's per-arm base-frame transforms.

Uses pinocchio on the checked-in dual URDF (no MuJoCo). Covers:

* the constant world->base transforms match the URDF's fixed base placements
  (left base translated by (0, 0.15, 0));
* mapping a world pose into the base frame and back is the identity;
* the quaternion convention is wxyz (identity rotation -> (1, 0, 0, 0)).

Run via: python -m unittest test.unit.test_sidecar_frames
(requires PYTHONPATH=.:src, as set by `source setup.sh`).
"""

import unittest

import numpy as np

from common.recording.sidecar import _quat_wxyz, compute_world_base_transforms


class TestSidecarBaseTransforms(unittest.TestCase):
    transforms: dict[str, np.ndarray]

    @classmethod
    def setUpClass(cls) -> None:
        cls.transforms = compute_world_base_transforms()

    def test_both_sides_present(self) -> None:
        self.assertEqual(set(self.transforms), {"left", "right"})
        for tf in self.transforms.values():
            self.assertEqual(tf.shape, (4, 4))
            # Bottom row of a homogeneous transform.
            np.testing.assert_allclose(tf[3], [0.0, 0.0, 0.0, 1.0])

    def test_left_base_translation(self) -> None:
        # The dual URDF fixes the left arm base at xyz "0 0.15 0" and the
        # right at "0 -0.15 0" (300 mm arm spacing), both relative to world.
        np.testing.assert_allclose(
            self.transforms["left"][:3, 3], [0.0, 0.15, 0.0], atol=1e-9
        )
        np.testing.assert_allclose(
            self.transforms["right"][:3, 3], [0.0, -0.15, 0.0], atol=1e-9
        )

    def test_inverse_round_trip(self) -> None:
        from common.data_manager_dual import DualDataManager
        from common.recording.sidecar import SidecarSampler

        sampler = SidecarSampler(
            data_manager=DualDataManager(),
            quest_reader=None,
            root="/tmp",  # never written to in this test
        )
        rng = np.random.default_rng(0)
        for side in ("left", "right"):
            world_pose = np.eye(4)
            # A random valid rotation + translation.
            angle = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            world_pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            world_pose[:3, 3] = rng.uniform(-0.5, 0.5, size=3)
            base_pose = sampler.to_base_frame(side, world_pose)
            back = sampler.base_transform(side) @ base_pose
            np.testing.assert_allclose(back, world_pose, atol=1e-12)

    def test_quaternion_wxyz_convention(self) -> None:
        w, x, y, z = _quat_wxyz(np.eye(3))
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 0.0)
        # 90 deg about z: w = cos(45 deg), z = sin(45 deg), x = y = 0.
        rot_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        w, x, y, z = _quat_wxyz(rot_z90)
        self.assertAlmostEqual(abs(w), np.cos(np.pi / 4), places=12)
        self.assertAlmostEqual(abs(z), np.sin(np.pi / 4), places=12)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 0.0, places=12)
        # Same sign for w and z (either both + or both -).
        self.assertGreater(w * z, 0.0)


if __name__ == "__main__":
    unittest.main()
