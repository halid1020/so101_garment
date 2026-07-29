"""Unit tests for the sim-VLA oracle scripts and scenario generation.

Pure pinocchio / numpy (no MuJoCo): scenario determinism and geometry, the
cube grasp depth, the grip schedule shape, and the target dict structure.
"""

import unittest

import numpy as np

from sim_datagen.oracle import (
    CUBE_HALF,
    GRASP_DZ,
    RELAY_MIDDLE_X,
    HandoverContactScript,
    SinglePickPlaceScript,
    generate_relay_scenarios,
    generate_single_scenarios,
)

# Fake neutral IK poses (MuJoCo-free): identity attitude, left +y / right -y.
_INIT_POSES = {
    "left": (np.array([0.12, 0.15, 0.10]), np.eye(3)),
    "right": (np.array([0.12, -0.15, 0.10]), np.eye(3)),
}
_TABLE_Z = -0.0364


class TestScenarioGeneration(unittest.TestCase):
    def test_single_deterministic(self):
        a = generate_single_scenarios(4, seed=7)
        b = generate_single_scenarios(4, seed=7)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x.payload_xy, y.payload_xy)
            np.testing.assert_allclose(x.target_xy, y.target_xy)
            self.assertEqual(x.side, y.side)

    def test_single_side_matches_y_sign(self):
        for sc in generate_single_scenarios(6, seed=0):
            sign = 1.0 if sc.side == "left" else -1.0
            self.assertGreater(sign * sc.payload_xy[1], 0.0)
            self.assertGreater(sign * sc.target_xy[1], 0.0)

    def test_relay_deterministic(self):
        a = generate_relay_scenarios(4, seed=3)
        b = generate_relay_scenarios(4, seed=3)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x.payload_xy, y.payload_xy)
            np.testing.assert_allclose(x.middle_xy, y.middle_xy)
            np.testing.assert_allclose(x.target_xy, y.target_xy)

    def test_relay_geometry(self):
        # Cube spawns on the left (+y), target on the right (-y), hand-off at
        # the midline; left always picks, right always places.
        for sc in generate_relay_scenarios(6, seed=1):
            self.assertEqual(sc.pick_side, "left")
            self.assertEqual(sc.place_side, "right")
            self.assertGreater(sc.payload_xy[1], 0.0)
            self.assertLess(sc.target_xy[1], 0.0)
            self.assertLess(abs(sc.middle_xy[1]), 0.03)
            self.assertAlmostEqual(sc.middle_xy[0], RELAY_MIDDLE_X, delta=0.03)


class TestGraspDepth(unittest.TestCase):
    def test_grasp_commanded_below_cube_centre(self):
        # The IK undershoots the low descent, so the pinch is aimed below the
        # cube centre (which sits at CUBE_HALF) — this is what fixed the grasp.
        self.assertLess(GRASP_DZ, CUBE_HALF)
        self.assertGreaterEqual(GRASP_DZ, 0.0)


class TestScripts(unittest.TestCase):
    def _single(self):
        sc = generate_single_scenarios(1, seed=0)[0]
        return sc, SinglePickPlaceScript(sc, _INIT_POSES, _TABLE_Z)

    def test_single_grip_schedule(self):
        _, script = self._single()
        # Open at the start, fully squeezed after the close ramp, open at the end.
        self.assertGreater(script.grip_fractions(0.0)[script.side], 0.8)
        mid = script.t_close_start + 0.35
        self.assertLess(script.grip_fractions(mid)[script.side], 0.2)
        self.assertGreater(script.grip_fractions(script.duration)[script.side], 0.8)

    def test_single_targets_structure(self):
        sc, script = self._single()
        tg = script.targets(1.0)
        self.assertEqual(set(tg), {"left", "right"})
        for side in ("left", "right"):
            pos, rot = tg[side]
            self.assertEqual(pos.shape, (3,))
            self.assertEqual(rot.shape, (3, 3))
        # The idle arm holds its initial position.
        np.testing.assert_allclose(tg[script.idle][0], _INIT_POSES[script.idle][0])

    def test_relay_two_phase_grip(self):
        sc = generate_relay_scenarios(1, seed=0)[0]
        script = HandoverContactScript(sc, _INIT_POSES, _TABLE_Z)
        # Left squeezes first (picks), then releases; right squeezes later.
        self.assertLess(script.grip_fractions(script.t_pick_close + 0.35)["left"], 0.2)
        self.assertGreater(
            script.grip_fractions(script.t_place_close - 0.05)["right"], 0.8
        )
        self.assertLess(
            script.grip_fractions(script.t_place_close + 0.35)["right"], 0.2
        )
        # Both open again by the end.
        for side in ("left", "right"):
            self.assertGreater(script.grip_fractions(script.duration)[side], 0.8)


if __name__ == "__main__":
    unittest.main()
