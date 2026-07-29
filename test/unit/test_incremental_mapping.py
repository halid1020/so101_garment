"""Unit tests for the incremental (clutched) wrist pitch/roll mapping.

Pure maths behind ``mymethod`` (tool/meta_quest_teleopration.py --method
mymethod): the operator's wrist pitch/roll are extracted absolutely each tick
and the gripper attitude tracks their CHANGE from a per-grip anchor. These
tests fix the decode/encode round-trip and the pitch/roll decoupling.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_incremental_mapping
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.utils import (
    gripper_orientation_from_pitch_roll,
    gripper_pitch_roll_from_rotation,
    signed_angle_about,
    wrist_roll_pitch_delta,
)


class TestGripperAngleRoundTrip(unittest.TestCase):
    def test_reconstruct_is_orthonormal_and_invertible(self) -> None:
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(1000):
            az = rng.uniform(-np.pi, np.pi)
            pitch = rng.uniform(-1.4, 1.4)
            roll = rng.uniform(-3.0, 3.0)
            r = gripper_orientation_from_pitch_roll(az, pitch, roll)
            self.assertTrue(np.allclose(r.T @ r, np.eye(3), atol=1e-8))
            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=6)
            az2, pitch2, roll2 = gripper_pitch_roll_from_rotation(r)
            r2 = gripper_orientation_from_pitch_roll(az2, pitch2, roll2)
            worst = max(worst, float(np.abs(r - r2).max()))
        self.assertLess(worst, 1e-9)

    def test_tip_azimuth_and_pitch_are_the_first_column(self) -> None:
        r = gripper_orientation_from_pitch_roll(0.5, 0.3, 1.0)
        tip = r[:3, 0]
        self.assertAlmostEqual(float(np.arctan2(tip[1], tip[0])), 0.5, places=6)
        self.assertAlmostEqual(
            float(np.arctan2(tip[2], np.linalg.norm(tip[:2]))), 0.3, places=6
        )


class TestWristRollPitchDelta(unittest.TestCase):
    """rel_rot is the hand rotation in the operator frame (x fwd, y left, z up)."""

    def test_identity_is_zero(self) -> None:
        roll, pitch = wrist_roll_pitch_delta(np.eye(3))
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)

    def test_pure_twist_about_forward_is_roll_only(self) -> None:
        rel = Rotation.from_rotvec([0.5, 0.0, 0.0]).as_matrix()  # about +x (fwd)
        roll, pitch = wrist_roll_pitch_delta(rel)
        self.assertAlmostEqual(roll, 0.5, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)

    def test_pure_nod_about_lateral_is_pitch_only(self) -> None:
        rel = Rotation.from_rotvec([0.0, 0.3, 0.0]).as_matrix()  # about +y (left)
        roll, pitch = wrist_roll_pitch_delta(rel)
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.3, places=6)

    def test_yaw_about_up_is_dropped(self) -> None:
        rel = Rotation.from_rotvec([0.0, 0.0, 0.4]).as_matrix()  # about +z (up)
        roll, pitch = wrist_roll_pitch_delta(rel)
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)


class TestSignedAngleAbout(unittest.TestCase):
    def test_orthogonal_reference_gives_ninety_degrees(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        ref = np.array([1.0, 0.0, 0.0])
        vec = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(signed_angle_about(axis, vec, ref), np.pi / 2, places=6)

    def test_parallel_to_axis_is_zero(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        self.assertEqual(signed_angle_about(axis, axis, np.array([1.0, 0.0, 0.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
