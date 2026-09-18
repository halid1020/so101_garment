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
    apply_crop,
    common_frames,
    crop_for,
    frames_of,
    maps_at,
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


class GradcamDrawingTest(unittest.TestCase):
    """The renderer, not merely the pieces it is built from.

    Every test above passed while the Grad-CAM video drew empty share bars,
    because each one exercised a helper and none exercised the choice between
    them. These cover that choice.
    """

    def test_maps_come_back_for_the_frame_asked_for(self):
        run = run_with([frame(10, cam={"central": [[1.0]]}), frame(20)])
        self.assertEqual(maps_at(run, 58, 10), {"central": [[1.0]]})

    def test_a_frame_with_no_maps_gives_nothing_rather_than_raising(self):
        run = run_with([frame(10, cam={"central": [[1.0]]}), frame(20)])
        self.assertIsNone(maps_at(run, 58, 20))
        self.assertIsNone(maps_at(run, 58, 999))

    def test_a_gradcam_video_is_refused_without_the_recording(self):
        # The maps are stored without the frames they were computed from, so a
        # Grad-CAM video with no dataset would have nothing to draw them over.
        # It must say so rather than render panels of heatmap on nothing.
        import subprocess
        import sys

        done = subprocess.run(
            [
                sys.executable,
                "tool/attribution_video.py",
                "--arm",
                "a=/nonexistent.json",
                "--episode",
                "58",
                "--mode",
                "gradcam",
                "--out",
                "/tmp/never_written.mp4",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(done.returncode, 0)

    def test_an_arm_with_no_crop_in_its_config_reports_none(self):
        self.assertIsNone(crop_for({}))
        self.assertIsNone(crop_for({"checkpoint": "/nowhere/at/all"}))

    def test_a_crop_is_read_from_the_checkpoint_it_scored(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps(
                    {
                        "tactile_crop": [0.8, 1.0],
                        "tactile_cameras": ["left_arm_left_gripper"],
                    }
                )
            )
            got = crop_for({"checkpoint": directory})
        self.assertEqual(got, ((0.8, 1.0), ("left_arm_left_gripper",)))

    def test_a_cropped_panel_is_not_the_frame_the_policy_never_saw(self):
        # The whole reason the crop is applied here: a cropped arm drawn from
        # the full frame would illustrate attention to rows it never received.
        # Same shape, different content, and the top rows are what changed.
        import numpy as np

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[:8] = 255  # a bright band only the uncropped arm can see
        cropped = apply_crop(image, (0.8, 1.0))
        self.assertEqual(cropped.shape, image.shape)
        self.assertLess(cropped[:8].mean(), image[:8].mean())

    def test_a_crop_of_one_keeps_the_image_exactly(self):
        import numpy as np

        image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
        self.assertTrue((apply_crop(image, (1.0, 1.0)) == image).all())


if __name__ == "__main__":
    unittest.main()
