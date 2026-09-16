"""The framewise attribution video: what it refuses, and what it lines up.

The failure this guards against is not a crash. It is a video that plays
perfectly and compares the wrong things -- different moments in different
panels, or a policy shown as attending to nothing when the method simply does
not apply to it.
"""

from __future__ import annotations

import unittest

from tool.attribution_video import (
    NoSuchMethod,
    common_frames,
    frames_of,
    require_gradcam,
    shares_at,
    tile_shape,
)


def run_with(frames, episode="58"):
    return {"episodes": {episode: {"frames": frames}}}


def frame(index, shares=None, cam=None):
    out = {"index": index}
    if shares is not None:
        out["occlusion"] = {
            "streams": {name: {"share": value} for name, value in shares.items()}
        }
    if cam is not None:
        out["gradcam"] = cam
    return out


class ReadingTest(unittest.TestCase):
    def test_an_episode_nobody_scored_reads_as_empty_not_as_an_error(self):
        # The caller decides what to do about it; this is not the place to fail.
        self.assertEqual(frames_of(run_with([frame(0)]), 99), [])

    def test_shares_come_out_per_stream(self):
        got = shares_at(frame(0, {"state": 0.4, "central": 0.6}))
        self.assertEqual(got, {"state": 0.4, "central": 0.6})

    def test_a_frame_with_no_occlusion_gives_no_shares_rather_than_raising(self):
        self.assertEqual(shares_at(frame(0)), {})


class AlignmentTest(unittest.TestCase):
    """Every panel of a video frame must show the SAME instant."""

    def test_the_video_covers_only_frames_every_arm_scored(self):
        runs = {
            "a": run_with([frame(0), frame(5), frame(10)]),
            "b": run_with([frame(0), frame(10)]),
        }
        self.assertEqual(common_frames(runs, 58), [0, 10])

    def test_a_shorter_arm_does_not_shift_the_others(self):
        # The wrong fix is to zip the lists: arm b's second panel would then be
        # frame 10 while arm a's is frame 5, and the video would compare two
        # different moments while looking perfectly well formed.
        runs = {
            "a": run_with([frame(0), frame(5), frame(10)]),
            "b": run_with([frame(0), frame(10)]),
        }
        self.assertNotIn(5, common_frames(runs, 58))

    def test_arms_that_share_no_frames_give_nothing(self):
        runs = {"a": run_with([frame(0)]), "b": run_with([frame(1)])}
        self.assertEqual(common_frames(runs, 58), [])


class GradcamRefusalTest(unittest.TestCase):
    def test_a_policy_with_maps_is_accepted(self):
        run = run_with([frame(0, cam={"central": [[1.0, 2.0]]})])
        require_gradcam("act", run, 58)  # does not raise

    def test_a_policy_without_maps_is_refused_by_name(self):
        run = run_with([frame(0, {"state": 1.0})])
        with self.assertRaises(NoSuchMethod) as caught:
            require_gradcam("pi05", run, 58)
        self.assertIn("pi05", str(caught.exception))

    def test_the_refusal_is_its_own_type(self):
        # Distinct from a plain error on purpose: "the method does not apply to
        # this architecture" and "this run did not include it" both arrive as a
        # missing key, and the caller drops the panel rather than failing.
        self.assertTrue(issubclass(NoSuchMethod, RuntimeError))


class LayoutTest(unittest.TestCase):
    def test_panels_go_across_before_they_go_down(self):
        self.assertEqual(tile_shape(2), (1, 2))
        self.assertEqual(tile_shape(3), (1, 3))

    def test_a_fourth_panel_starts_a_second_row(self):
        self.assertEqual(tile_shape(4), (2, 3))

    def test_six_panels_fill_two_rows(self):
        self.assertEqual(tile_shape(6), (2, 3))

    def test_no_panels_is_refused(self):
        with self.assertRaises(ValueError):
            tile_shape(0)


if __name__ == "__main__":
    unittest.main()
