"""The control rate is a parameter, and it has to divide the physics rate.

A control tick that is not a whole number of integrator steps makes recorded
time and simulated time drift apart a fraction of a step at a time, which is
the kind of error that never announces itself -- it just quietly puts every
timestamp in a dataset slightly out of step with the motion it labels.
"""

from __future__ import annotations

import unittest

from sim_datagen.env import DEFAULT_FPS, PHYSICS_HZ, substeps_for


class TestSubstepsFor(unittest.TestCase):
    def test_the_two_rates_the_project_uses(self) -> None:
        self.assertEqual(substeps_for(25), 24)
        self.assertEqual(substeps_for(30), 20)

    def test_the_default_is_25_and_divides_the_physics_rate(self) -> None:
        self.assertEqual(DEFAULT_FPS, 25.0)
        self.assertEqual(PHYSICS_HZ / DEFAULT_FPS, substeps_for(DEFAULT_FPS))

    def test_a_rate_that_does_not_divide_is_refused(self) -> None:
        for fps in (7, 16, 45, 33):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError) as ctx:
                    substeps_for(fps)
                self.assertIn(str(PHYSICS_HZ), str(ctx.exception))

    def test_non_positive_rates_are_refused(self) -> None:
        for fps in (0, -25):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError):
                    substeps_for(fps)

    def test_a_whole_number_of_substeps_is_always_returned(self) -> None:
        for fps in (10, 12, 20, 24, 25, 30, 40, 50, 60):
            with self.subTest(fps=fps):
                n = substeps_for(fps)
                self.assertIsInstance(n, int)
                self.assertEqual(n * fps, PHYSICS_HZ)


if __name__ == "__main__":
    unittest.main()
