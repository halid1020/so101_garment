#!/usr/bin/env python3
"""Unit tests for the Telegrip-style split-IK teleop method.

Run via: python -m unittest test.integration.test_telegrip
(requires PYTHONPATH=.:src, as set by `source setup.sh`)
"""

import unittest

import numpy as np

from sim_benchmark.methods import METHODS
from sim_benchmark.methods.telegrip_split import (
    TelegripSplit,
    _tip_elevation,
    _tip_roll,
    _wrap_pi,
)
from sim_benchmark.scene import DualArmSim
from src.common.configs import NEUTRAL_JOINT_ANGLES_DUAL


class TestTelegripCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = DualArmSim()
        cls.method = TelegripSplit(cls.sim.model)

    def test_registered(self):
        self.assertIn("telegrip", METHODS)

    def test_slopes_are_unit_signs(self):
        for side in ("left", "right"):
            cal = self.method._wrist_cal[side]
            for key in ("s2", "s3", "s4", "s5"):
                self.assertIn(cal[key], (-1.0, 1.0), msg=f"{side}/{key}")

    def test_elevation_affine_model_on_grid(self):
        """elev = e0 + s2 q2 + s3 q3 + s4 q4 must hold across joint space."""
        for side in ("left", "right"):
            cal = self.method._wrist_cal[side]
            for q2 in np.linspace(-1.2, 1.2, 4):
                for q3 in np.linspace(-1.2, 1.2, 4):
                    for q4 in np.linspace(-1.2, 1.2, 4):
                        q_arm = np.array([0.2, q2, q3, q4, 0.1])
                        elev_fk = _tip_elevation(self.method._fk_rot(side, q_arm))
                        elev_model = _wrap_pi(
                            cal["e0"] + cal["s2"] * q2 + cal["s3"] * q3 + cal["s4"] * q4
                        )
                        # atan2-based elevation folds outside [-pi/2, pi/2];
                        # only compare where the model is in range.
                        if abs(elev_model) < np.radians(85):
                            self.assertAlmostEqual(
                                elev_fk,
                                elev_model,
                                delta=np.radians(1.0),
                                msg=f"{side} q=({q2:.2f},{q3:.2f},{q4:.2f})",
                            )

    def test_roll_affine_model(self):
        """roll = r0 + s5 q5 at several arm poses (tip not vertical)."""
        for side in ("left", "right"):
            cal = self.method._wrist_cal[side]
            for q5 in np.linspace(-1.5, 1.5, 7):
                q_arm = np.array([0.1, 0.3, 0.5, 0.3, q5])
                roll_fk = _tip_roll(self.method._fk_rot(side, q_arm))
                self.assertIsNotNone(roll_fk)
                roll_model = _wrap_pi(cal["r0"] + cal["s5"] * q5)
                self.assertAlmostEqual(
                    _wrap_pi(roll_fk - roll_model),
                    0.0,
                    delta=np.radians(1.0),
                    msg=f"{side} q5={q5:.2f}",
                )


class TestTelegripSolve(unittest.TestCase):
    def setUp(self):
        self.sim = DualArmSim()
        self.method = TelegripSplit(self.sim.model)
        self.q0 = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
        self.method.reset(self.q0)

    def _targets_from(self, q):
        poses = self.sim.fk_eef_pose(q)
        return {
            side: (poses[side][:3, 3].copy(), poses[side][:3, :3].copy())
            for side in ("left", "right")
        }

    def test_static_convergence_position_and_wrist(self):
        """A reachable target pose (position + tilted wrist) is reached to
        <5 mm and <2 deg within 100 ticks."""
        q_goal = self.q0.copy()
        q_goal[[0, 1, 2, 3, 4]] += [0.15, -0.2, 0.25, -0.35, 0.5]
        q_goal[[5, 6, 7, 8, 9]] += [-0.15, -0.2, 0.25, -0.35, -0.5]
        targets = self._targets_from(q_goal)
        q = self.q0.copy()
        for _ in range(100):
            q = self.method.solve(targets, 0.01)
        fk = self.sim.fk_eef_pose(q)
        for side in ("left", "right"):
            pos_err = np.linalg.norm(fk[side][:3, 3] - targets[side][0])
            self.assertLess(pos_err, 0.005, msg=f"{side} pos_err={pos_err:.4f}")
            elev_err = abs(
                _tip_elevation(fk[side][:3, :3]) - _tip_elevation(targets[side][1])
            )
            self.assertLess(np.degrees(elev_err), 2.0, msg=f"{side} elevation")
            r_t = _tip_roll(targets[side][1])
            r_m = _tip_roll(fk[side][:3, :3])
            self.assertIsNotNone(r_t)
            self.assertIsNotNone(r_m)
            roll_err = abs(_wrap_pi(r_t - r_m))
            self.assertLess(np.degrees(roll_err), 2.0, msg=f"{side} roll")

    def test_vertical_tip_holds_roll(self):
        """With the tip pointing straight down (roll reference degenerate),
        the method must hold the previous wrist_roll rather than jump."""
        q5_before = self.method.q[4]
        # Build a target with the tip exactly vertical (down).
        down = np.eye(3)
        down[:, 0] = [0.0, 0.0, -1.0]  # tip
        down[:, 1] = [0.0, 1.0, 0.0]
        down[:, 2] = [1.0, 0.0, 0.0]  # right-handed: x cross y = z? checked below
        # Ensure right-handedness: z = x cross y.
        down[:, 2] = np.cross(down[:, 0], down[:, 1])
        self.assertIsNone(_tip_roll(down))
        poses = self.sim.fk_eef_pose(self.q0)
        targets = {
            side: (poses[side][:3, 3].copy(), down.copy()) for side in ("left", "right")
        }
        for _ in range(20):
            self.method.solve(targets, 0.01)
        self.assertAlmostEqual(self.method.q[4], q5_before, delta=1e-9)

    def test_joint_limits_respected(self):
        """An extreme unreachable pose request must clamp inside limits."""
        poses = self.sim.fk_eef_pose(self.q0)
        rot = poses["left"][:3, :3]
        targets = {
            side: (np.array([0.9, 0.0, 0.8]), rot.copy()) for side in ("left", "right")
        }
        q = self.q0.copy()
        for _ in range(300):
            q = self.method.solve(targets, 0.01)
        self.assertTrue(np.all(q >= self.method._q_low - 1e-9))
        self.assertTrue(np.all(q <= self.method._q_high + 1e-9))

    def test_velocity_bounded(self):
        """Per-tick joint step never exceeds MAX_JOINT_VEL."""
        q_goal = self.q0.copy()
        q_goal[4] += 2.0  # big roll jump request
        q_goal[9] -= 2.0
        targets = self._targets_from(q_goal)
        dt = 0.01
        q_prev = self.method.q.copy()
        for _ in range(50):
            q = self.method.solve(targets, dt)
            self.assertLessEqual(
                np.abs(q - q_prev).max() / dt, self.method.MAX_JOINT_VEL + 1e-6
            )
            q_prev = q


if __name__ == "__main__":
    unittest.main()
