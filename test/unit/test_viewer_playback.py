"""Structural guards on the dataset viewer's end-of-episode stop.

This behaviour has been wrong twice, in two different ways, and both were
invisible to the Python tests because the logic lives in the page's script. The
first was a boundary error, fixed by handing the browser an inclusive window
(see test_dataset_read). The second was subtler and survived that fix: the stop
was driven by the video element's own ``timeupdate`` event, which browsers
throttle to roughly four times a second. A quarter of a second is seven frames
at the dataset rate, so playback ran well past the episode's end and into the
next recording before the check ever fired -- simulated at 233 ms of overshoot.

The check therefore has to be driven by the frame loop, and the overshoot undone
rather than merely stopped. These assertions are deliberately structural: they
cannot verify the arithmetic, but they do catch the specific regression of
moving that decision back onto a throttled event.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_viewer_playback
"""

import re
import unittest

from tool.dataset_web import _INDEX_HTML


def _script() -> str:
    match = re.search(r"<script>(.*)</script>", _INDEX_HTML, re.S)
    assert match, "the page has no script block"
    return match.group(1)


class TestEndOfEpisodeStop(unittest.TestCase):
    def test_the_stop_is_not_driven_by_a_throttled_event(self):
        script = _script()
        offenders = [
            line.strip()
            for line in script.splitlines()
            if "ontimeupdate" in line or "'timeupdate'" in line
        ]
        self.assertEqual(
            offenders,
            [],
            "the end-of-episode stop must not hang off timeupdate: browsers "
            "throttle it to about 4 Hz, which is seven frames of overshoot at "
            "the dataset rate, so playback reaches the next recording before "
            f"the check fires. Found: {offenders}",
        )

    def test_the_stop_is_checked_every_frame(self):
        script = _script()
        paint = script[script.index("const paint = ") : script.index("const play = ")]
        self.assertIn(
            "pause()",
            paint,
            "the frame loop must decide when the episode ends; it runs at the "
            "display rate, so it can stop within one frame of the boundary",
        )

    def test_the_overshoot_is_undone_not_merely_stopped(self):
        # Pausing leaves whatever frame the decoder had reached on screen, which
        # is the wrong one if it went past the end at all.
        self.assertIn("const settle = ", _script())

    def test_settling_ignores_the_drift_tolerance(self):
        # The drift correction only nudges when a stream is more than 40 ms out,
        # which is wider than a frame period, so reusing it here would decline
        # to undo exactly the overshoot this is for.
        script = _script()
        settle = script[script.index("const settle = ") :]
        settle = settle[: settle.index("});") + 3]
        self.assertIn("currentTime =", settle)
        self.assertNotIn(
            "Math.abs",
            settle,
            "settling must assign the time outright; a tolerance check here "
            "would refuse to correct an overshoot smaller than the tolerance, "
            "which is every overshoot worth correcting",
        )

    def test_a_follower_at_its_end_is_paused_not_dragged_back(self):
        # The drift correction clamps its target to the stream's end while the
        # video keeps playing past it, so correcting a follower there would drag
        # it back every frame while it ran forward into the next recording in
        # between. Pausing it removes that oscillation.
        script = _script()
        self.assertIn("videos[i].pause()", script)

    def test_the_view_stops_short_of_the_next_recording(self):
        # Episodes are packed end to end, so the frames after one belong to the
        # next. A margin keeps every displayed frame unambiguously inside the
        # episode even if the decoders stray; the rendered mp4 keeps them all.
        from tool.dataset_web import _VIEW_END_MARGIN_S

        self.assertGreater(_VIEW_END_MARGIN_S, 0.0)
        self.assertLess(
            _VIEW_END_MARGIN_S,
            0.5,
            "a large margin would hide the end of the task, which is the part a "
            "reviewer is usually judging",
        )


if __name__ == "__main__":
    unittest.main()
