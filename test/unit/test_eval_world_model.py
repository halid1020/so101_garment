"""The world-model evaluation tool's pure parts.

Loading a checkpoint and decoding video belong to the integration tier; the
range parsing, the summarising and the report are arithmetic and string work,
and they are where a wrong answer would be quiet rather than loud.
"""

from __future__ import annotations

import unittest

from tool.eval_world_model import parse_range, report, summarise


def frame(psnr, baseline, ssim=0.5, ssim_baseline=0.5, camera="central"):
    return {
        camera: {
            "psnr": psnr,
            "psnr_baseline": baseline,
            "ssim": [ssim] * len(psnr),
            "ssim_baseline": [ssim_baseline] * len(psnr),
            "mse": [0.1] * len(psnr),
            "mse_baseline": [0.1] * len(psnr),
        }
    }


class ParseRangeTest(unittest.TestCase):
    def test_a_span(self):
        self.assertEqual(parse_range("0-4"), [0, 1, 2, 3, 4])

    def test_a_list(self):
        self.assertEqual(parse_range("0,3,7"), [0, 3, 7])

    def test_spans_and_singles_together(self):
        self.assertEqual(parse_range("0-2,9"), [0, 1, 2, 9])

    def test_whitespace_and_empty_pieces_are_tolerated(self):
        self.assertEqual(parse_range(" 1 , 2 ,"), [1, 2])


class SummariseTest(unittest.TestCase):
    def test_it_averages_over_frames_and_keeps_the_horizon(self):
        """The fall with horizon IS the result; a single mean would hide it."""
        frames = [frame([10.0, 8.0], [5.0, 5.0]), frame([20.0, 12.0], [7.0, 7.0])]
        out = summarise(frames)
        self.assertEqual(out["central"]["psnr"], [15.0, 10.0])
        self.assertEqual(out["central"]["psnr_baseline"], [6.0, 6.0])

    def test_no_frames_gives_nothing_rather_than_raising(self):
        self.assertEqual(summarise([]), {})

    def test_every_camera_survives(self):
        combined = {
            **frame([1.0], [1.0], camera="a"),
            **frame([2.0], [2.0], camera="b"),
        }
        self.assertEqual(set(summarise([combined])), {"a", "b"})


class ReportTest(unittest.TestCase):
    def test_it_counts_the_horizon_steps_that_beat_the_baseline(self):
        summary = summarise([frame([30.0, 4.0, 30.0], [10.0, 10.0, 10.0])])
        self.assertIn("2/3 steps", report(summary))

    def test_a_model_that_never_beats_holding_says_so(self):
        summary = summarise([frame([4.0, 4.0], [10.0, 10.0])])
        self.assertIn("0/2 steps", report(summary))

    def test_the_baseline_is_named_in_the_output(self):
        """A reader must not have to know what 'held' means."""
        text = report(summarise([frame([10.0], [10.0])]))
        self.assertIn("last observed frame repeated", text)


if __name__ == "__main__":
    unittest.main()
