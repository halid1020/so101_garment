"""Unit tests for the out-of-envelope operator feedback layer.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_envelope_feedback

Pure Python — OOEStatus objects are constructed directly; no MuJoCo, no adb.
"""

import unittest

from common.configs import FEEDBACK_REPEAT_PERIOD_S, WORKSPACE_SOFT_MARGIN
from common.envelope_feedback import NullFeedback, RateLimitedFeedback
from common.workspace_envelope import OOEStatus


def _status(margin_m: float) -> OOEStatus:
    """Build an OOEStatus from a signed margin (negative = outside)."""
    return OOEStatus(inside=margin_m >= 0.0, margin_m=margin_m, clamped=margin_m < 0.0)


class RecordingFeedback(RateLimitedFeedback):
    """Test backend that records every _emit call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.emits: list[tuple[str, float, float]] = []

    def _emit(self, side: str, intensity: float, t: float) -> None:
        self.emits.append((side, intensity, t))


class TestRateLimitedFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.fb = RecordingFeedback()

    def test_rising_edge_emits_immediately(self) -> None:
        self.fb.notify("left", _status(0.01), t=0.0)  # inside: nothing
        self.assertEqual(self.fb.emits, [])
        self.fb.notify("left", _status(-0.02), t=1.0)  # edge: immediate
        self.assertEqual(len(self.fb.emits), 1)
        side, intensity, t = self.fb.emits[0]
        self.assertEqual(side, "left")
        self.assertGreater(intensity, 0.0)
        self.assertEqual(t, 1.0)

    def test_sustained_outside_respects_repeat_period(self) -> None:
        period = FEEDBACK_REPEAT_PERIOD_S
        self.fb.notify("left", _status(-0.02), t=0.0)  # edge
        self.fb.notify("left", _status(-0.02), t=period * 0.5)  # too soon
        self.assertEqual(len(self.fb.emits), 1)
        self.fb.notify("left", _status(-0.02), t=period * 1.1)  # due
        self.assertEqual(len(self.fb.emits), 2)
        self.fb.notify("left", _status(-0.02), t=period * 1.5)  # too soon again
        self.assertEqual(len(self.fb.emits), 2)

    def test_reentry_emits_single_zero_intensity_stop(self) -> None:
        self.fb.notify("left", _status(-0.02), t=0.0)
        self.fb.notify("left", _status(0.01), t=0.1)  # falling edge
        self.fb.notify("left", _status(0.01), t=0.2)  # still inside: nothing
        self.assertEqual(len(self.fb.emits), 2)
        self.assertEqual(self.fb.emits[-1][1], 0.0)

    def test_intensity_monotone_and_saturating(self) -> None:
        soft = WORKSPACE_SOFT_MARGIN
        i_boundary = self.fb._intensity(0.0)
        i_shallow = self.fb._intensity(-0.25 * soft)
        i_mid = self.fb._intensity(-0.5 * soft)
        i_deep = self.fb._intensity(-soft)
        i_deeper = self.fb._intensity(-2.0 * soft)
        self.assertEqual(i_boundary, 0.0)
        self.assertLess(i_shallow, i_mid)
        self.assertLess(i_mid, i_deep)
        self.assertEqual(i_deep, 1.0)
        self.assertEqual(i_deeper, 1.0)

    def test_reset_rearms_the_edge(self) -> None:
        self.fb.notify("left", _status(-0.02), t=0.0)
        self.fb.reset()
        # Immediately after reset the same outside state is a fresh edge.
        self.fb.notify("left", _status(-0.02), t=0.01)
        self.assertEqual(len(self.fb.emits), 2)

    def test_sides_are_independent(self) -> None:
        self.fb.notify("left", _status(-0.02), t=0.0)
        self.fb.notify("right", _status(0.01), t=0.0)  # right inside: nothing
        self.fb.notify("right", _status(-0.02), t=0.05)  # right edge fires
        self.assertEqual([e[0] for e in self.fb.emits], ["left", "right"])
        # Left's repeat clock is unaffected by right's edge.
        self.fb.notify("left", _status(-0.02), t=0.1)
        self.assertEqual(len(self.fb.emits), 2)

    def test_null_feedback_is_inert(self) -> None:
        nf = NullFeedback()
        nf.notify("left", _status(-0.02), t=0.0)  # must not raise
        nf.reset()


if __name__ == "__main__":
    unittest.main()
