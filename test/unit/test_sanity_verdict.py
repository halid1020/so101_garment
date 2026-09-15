"""Adebayo's model-randomisation test, and which way round it fails.

The check asks whether randomising a policy's weights leaves its saliency
UNCHANGED. If it does, the attribution was measuring the input, not the model.
"Unchanged" is a strong POSITIVE rank correlation between before and after.

The first version wrapped the correlation in `abs()`, so a strongly NEGATIVE
one -- the ranking reversed, which is randomisation destroying it -- was
reported as a failure. pi0.5's deck said "the ranking SURVIVED randomisation"
on a measured agreement of -0.77, the opposite of what happened, on a figure
headed for a supervisor.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_sanity_verdict
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "tool" / "analyse_policy_inputs.py"


def verdict_for(agreement: float, total: float = 1.0) -> str:
    """The branch the driver takes, kept in step with it by the test below."""
    if total <= 1e-9:
        return "ignores every input entirely"
    if agreement != agreement or agreement < 0.5:
        out = "did not survive"
        if agreement < -0.5:
            out += " (reversed)"
        return out
    return "SURVIVED"


class WhichWayItFailsTest(unittest.TestCase):
    def test_a_reversed_ranking_is_a_pass(self):
        # pi0.5's measured value. Randomisation reordered the streams
        # completely; that is the check working, not the check failing.
        self.assertIn("did not survive", verdict_for(-0.7745966692414834))
        self.assertIn("reversed", verdict_for(-0.7745966692414834))

    def test_an_unchanged_ranking_is_the_failure(self):
        self.assertIn("SURVIVED", verdict_for(0.9))
        self.assertIn("SURVIVED", verdict_for(0.51))

    def test_an_uncorrelated_ranking_is_a_pass(self):
        self.assertIn("did not survive", verdict_for(0.1))
        self.assertNotIn("reversed", verdict_for(0.1))

    def test_nan_is_a_pass_not_a_crash(self):
        # No ranking to correlate -- fewer than two streams moved at all.
        self.assertIn("did not survive", verdict_for(float("nan")))

    def test_every_share_zero_is_the_strongest_pass(self):
        self.assertIn("ignores every input", verdict_for(0.9, total=0.0))


class TheDriverUsesThisRuleTest(unittest.TestCase):
    def test_the_condition_is_not_wrapped_in_abs(self):
        """The bug, pinned. `abs()` here calls a reversal a failure."""
        text = DRIVER.read_text(encoding="utf-8")
        branch = re.search(r"elif agreement != agreement or ([^:]+):", text)
        self.assertIsNotNone(branch, "the sanity branch moved; re-read it")
        self.assertNotIn("abs(", branch.group(1))
        self.assertIn("agreement < 0.5", branch.group(1))


if __name__ == "__main__":
    unittest.main()
