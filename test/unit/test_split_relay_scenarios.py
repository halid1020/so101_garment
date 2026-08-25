"""The split relay is only a relay if neither arm can finish it alone.

The plain relay generator checks that the ACTING arm can reach each keypose
and stops there, so a sampled target may sit inside both envelopes -- and a
policy that finds one has learned a single-arm task wearing a relay's name.
These tests assert the property the split variant exists to guarantee: the
LEFT arm cannot place on the plate, so the hand-off is forced by geometry.
"""

from __future__ import annotations

import unittest

import numpy as np

from sim_benchmark.handover import FEASIBILITY_TOL, _ReachChecker
from sim_datagen.oracle import (
    CONSERVATIVE_RADIAL_MARGIN,
    EXCLUSION_MARGIN,
    GRASP_DZ,
    GRASP_OFFSET_WORLD,
    NOMINAL_TABLE_Z_IK,
    SPLIT_CUBE_X_RANGE,
    SPLIT_CUBE_Y_RANGE,
    SPLIT_PLATE_Y_RANGE,
    _radial_margin,
    generate_split_relay_scenarios,
)


def _ee(xy: np.ndarray, z: float) -> np.ndarray:
    return np.array([xy[0], xy[1], z]) - GRASP_OFFSET_WORLD


class TestSplitRelayScenarios(unittest.TestCase):
    """One sampling pass shared by every check -- each costs ~0.2 s."""

    scenarios: list
    checker: _ReachChecker
    envelopes: dict
    gz: float

    @classmethod
    def setUpClass(cls) -> None:
        from common.workspace_envelope import build_envelopes

        cls.scenarios = generate_split_relay_scenarios(8, seed=0)
        cls.checker = _ReachChecker()
        cls.envelopes = build_envelopes(
            cls.checker.model, z_floor=NOMINAL_TABLE_Z_IK + 0.005
        )
        cls.gz = NOMINAL_TABLE_Z_IK + GRASP_DZ

    def test_it_produces_the_requested_number(self) -> None:
        self.assertEqual(len(self.scenarios), 8)
        self.assertEqual([s.index for s in self.scenarios], list(range(8)))

    def test_the_cube_is_on_the_left_and_the_plate_on_the_right(self) -> None:
        for s in self.scenarios:
            self.assertEqual(s.pick_side, "left")
            self.assertEqual(s.place_side, "right")
            self.assertGreater(s.payload_xy[1], 0.0)
            self.assertLess(s.target_xy[1], 0.0)

    def test_the_left_arm_cannot_reach_the_plate(self) -> None:
        miss = FEASIBILITY_TOL + EXCLUSION_MARGIN
        for s in self.scenarios:
            err = self.checker.reach_error("left", _ee(s.target_xy, self.gz))
            self.assertGreaterEqual(err, miss, f"plate at {s.target_xy}")

    def test_each_acting_arm_reaches_its_own_item(self) -> None:
        for s in self.scenarios:
            self.assertLess(
                self.checker.reach_error("left", _ee(s.payload_xy, self.gz)),
                FEASIBILITY_TOL,
            )
            self.assertLess(
                self.checker.reach_error("right", _ee(s.target_xy, self.gz)),
                FEASIBILITY_TOL,
            )

    def test_both_arms_reach_the_staging_point(self) -> None:
        for s in self.scenarios:
            for side in ("left", "right"):
                self.assertLess(
                    self.checker.reach_error(side, _ee(s.middle_xy, self.gz)),
                    FEASIBILITY_TOL,
                )

    def test_the_acting_arm_has_conservative_radial_room(self) -> None:
        for s in self.scenarios:
            self.assertGreaterEqual(
                _radial_margin(self.envelopes["left"], _ee(s.payload_xy, self.gz)),
                CONSERVATIVE_RADIAL_MARGIN,
            )
            self.assertGreaterEqual(
                _radial_margin(self.envelopes["right"], _ee(s.target_xy, self.gz)),
                CONSERVATIVE_RADIAL_MARGIN,
            )

    def test_samples_stay_inside_the_measured_sampling_boxes(self) -> None:
        for s in self.scenarios:
            self.assertGreaterEqual(s.payload_xy[0], SPLIT_CUBE_X_RANGE[0])
            self.assertLessEqual(s.payload_xy[0], SPLIT_CUBE_X_RANGE[1])
            self.assertGreaterEqual(s.payload_xy[1], SPLIT_CUBE_Y_RANGE[0])
            self.assertLessEqual(s.payload_xy[1], SPLIT_CUBE_Y_RANGE[1])
            self.assertGreaterEqual(-s.target_xy[1], SPLIT_PLATE_Y_RANGE[0])
            self.assertLessEqual(-s.target_xy[1], SPLIT_PLATE_Y_RANGE[1])

    def test_the_cube_stays_in_the_band_the_oracle_can_pick_from(self) -> None:
        """Not widened on purpose. MEASURED: pushing the cube out to where the
        right arm cannot reach it either breaks the pick -- the direct oracle
        failed 3 of 4 probes there against 100 % inside this box."""
        self.assertEqual(SPLIT_CUBE_X_RANGE, (0.24, 0.31))
        self.assertEqual(SPLIT_CUBE_Y_RANGE, (0.09, 0.17))

    def test_a_seed_always_denotes_the_same_scenario(self) -> None:
        again = generate_split_relay_scenarios(3, seed=11)
        once = generate_split_relay_scenarios(3, seed=11)
        for a, b in zip(again, once):
            np.testing.assert_allclose(a.payload_xy, b.payload_xy)
            np.testing.assert_allclose(a.middle_xy, b.middle_xy)
            np.testing.assert_allclose(a.target_xy, b.target_xy)

    def test_different_seeds_give_different_cube_spawns(self) -> None:
        """The whole point: the initial state moves. The plain simple mode
        repeated one scenario 100 times with a per-channel spread of zero."""
        spawns = {
            tuple(np.round(generate_split_relay_scenarios(1, seed=s)[0].payload_xy, 6))
            for s in range(6)
        }
        self.assertEqual(len(spawns), 6)


if __name__ == "__main__":
    unittest.main()
