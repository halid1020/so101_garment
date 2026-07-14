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
    operator_wrist_pitch_roll,
    signed_angle_about,
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


class TestOperatorWristDecoupling(unittest.TestCase):
    def setUp(self) -> None:
        self.handle_axis = np.array([0.8242, 0.2110, -0.5255])
        self.handle_axis = self.handle_axis / np.linalg.norm(self.handle_axis)
        self.knuckle_axis = np.array([0.0, 1.0, 0.0])
        self.hand = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()

    def test_pure_twist_changes_roll_only(self) -> None:
        p0, r0 = operator_wrist_pitch_roll(
            self.hand, self.handle_axis, self.knuckle_axis
        )
        handle_world = self.hand @ self.handle_axis
        twist = Rotation.from_rotvec(
            0.5 * handle_world / np.linalg.norm(handle_world)
        ).as_matrix()
        p1, r1 = operator_wrist_pitch_roll(
            twist @ self.hand, self.handle_axis, self.knuckle_axis
        )
        self.assertAlmostEqual(p1 - p0, 0.0, places=6)
        self.assertAlmostEqual(r1 - r0, 0.5, places=6)

    def test_tilting_handle_up_changes_pitch(self) -> None:
        p0, _ = operator_wrist_pitch_roll(
            self.hand, self.handle_axis, self.knuckle_axis
        )
        # Rotate the hand about a world-horizontal axis perpendicular to the
        # handle's ground projection -> the handle elevates -> pitch changes.
        hw = self.hand @ self.handle_axis
        horiz = np.array([hw[0], hw[1], 0.0])
        horiz /= np.linalg.norm(horiz)
        lateral = np.cross(np.array([0.0, 0.0, 1.0]), horiz)
        lift = Rotation.from_rotvec(0.2 * lateral).as_matrix()
        p1, _ = operator_wrist_pitch_roll(
            lift @ self.hand, self.handle_axis, self.knuckle_axis
        )
        self.assertGreater(abs(p1 - p0), 0.1)


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
