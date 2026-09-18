"""The two coordinate systems an attribution time plot mixes, and the bug they caused.

A contribution-over-time figure draws curves against a frame number that counts
across the WHOLE dataset, and shades phase bands that count positions WITHIN an
episode. Both were being drawn as though they were the same number. Neither
mistake raises: the figure renders, and it says that a recording made two thirds
of the way through a session began at frame zero and spent all of itself in one
phase.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from actoris_harena.analysis.phases import spans_of  # noqa: E402
from actoris_harena.analysis.report import shade_phases  # noqa: E402

from tool.analyse_policy_inputs import phase_labels  # noqa: E402


class PhaseLabelTest(unittest.TestCase):
    def test_the_first_episode_is_unchanged(self):
        # Episode zero starts at row zero, which is why the bug hid: here the
        # two numbering schemes agree.
        phases = ["reaching", "reaching", "closing", "holding"]
        self.assertEqual(
            phase_labels([0, 1, 2, 3], phases),
            ["reaching", "reaching", "closing", "holding"],
        )

    def test_a_later_episode_is_shifted_not_clamped(self):
        # The real case: episode 58 begins at row 27465. Clamping gave every
        # frame the last phase; shifting gives each frame its own.
        phases = ["reaching", "reaching", "closing", "holding"]
        got = phase_labels([27465, 27466, 27467, 27468], phases)
        self.assertEqual(got, ["reaching", "reaching", "closing", "holding"])
        # What the clamp produced: the last phase, for every frame.
        self.assertNotEqual(got, ["holding"] * 4)

    def test_sampling_every_nth_frame_still_lands_in_the_right_phase(self):
        phases = ["reaching"] * 10 + ["closing"] * 10
        got = phase_labels([100, 105, 110, 115], phases)
        self.assertEqual(got, ["reaching", "reaching", "closing", "closing"])

    def test_a_frame_past_the_end_takes_the_last_phase_rather_than_raising(self):
        self.assertEqual(phase_labels([10, 11], ["closing"]), ["closing", "closing"])

    def test_no_phases_gives_nothing_to_shade(self):
        self.assertIsNone(phase_labels([1, 2], []))


class ShadingTest(unittest.TestCase):
    """Where the bands land, in the coordinates the curves are drawn in."""

    def spans(self, axis):
        return [
            (patch.get_x(), patch.get_x() + patch.get_width()) for patch in axis.patches
        ]

    def test_bands_follow_the_x_the_curves_were_plotted_against(self):
        figure, axis = plt.subplots()
        frames = [27465, 27470, 27475, 27480]
        # "closing" and "holding" are shaded; "reaching" is deliberately not.
        labels = ["closing", "closing", "holding", "holding"]
        shade_phases(axis, labels, spans_of, x=frames)
        drawn = self.spans(axis)
        plt.close(figure)
        self.assertTrue(drawn, "no phase was shaded at all")
        # Every band sits inside the range the curves occupy. Before the fix
        # they sat at 0..2, off the left edge of a plot running to 27480.
        for left, right in drawn:
            self.assertGreaterEqual(left, frames[0])
            self.assertLessEqual(right, frames[-1])

    def test_without_an_x_the_bands_stay_in_position_coordinates(self):
        # The older behaviour is kept for a caller that plots against position,
        # so the fix adds a conversion rather than changing what x means.
        figure, axis = plt.subplots()
        shade_phases(axis, ["closing", "closing", "holding"], spans_of)
        drawn = self.spans(axis)
        plt.close(figure)
        self.assertTrue(drawn)
        for left, right in drawn:
            self.assertLessEqual(right, 3)


if __name__ == "__main__":
    unittest.main()
