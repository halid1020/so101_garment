"""The world-model evaluation tool's pure parts.

Loading a checkpoint and decoding video belong to the integration tier; the
range parsing, the summarising and the report are arithmetic and string work,
and they are where a wrong answer would be quiet rather than loud.
"""

from __future__ import annotations

import unittest

import torch

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


class WorldModelContractTest(unittest.TestCase):
    """What `eval_world_model.py` calls, asserted against the real classes.

    Stage 7a's claim is that the two world models are now comparable on one
    metric. That is only true if the tool can actually drive both, and the tool
    refuses a policy by DUCK TYPING -- it asks for `predict_future_frames` and
    gives up if it is missing. So the claim is checkable here, cheaply, without
    a GPU or 20 GB of Wan weights: the contract is a set of attribute names on
    a class, and a class carries them whether or not anything has loaded into
    it.

    This is worth pinning rather than trusting because the failure is silent in
    the wrong direction: a renamed method does not break a test that imports
    the policy, it makes the tool report the policy as "not a world model" --
    which reads as a fact about the architecture rather than about a rename.
    """

    # Every attribute tool/eval_world_model.py reaches for, and where.
    POLICY_METHODS = ("predict_future_frames", "tile_cameras", "untile_cameras")
    CONFIG_FIELDS = (
        "n_context_chunks",
        "latent_frames_per_chunk",
        "image_features",
        "observation_delta_indices",
        "action_delta_indices",
    )

    def policies(self):
        from actoris_harena.policies.dreamzero.modeling_dreamzero import (
            HarenaDreamzeroPolicy,
        )
        from actoris_harena.policies.fastwam_predict.modeling_fastwam_predict import (
            HarenaFastwamPredictPolicy,
        )

        return {
            "harena_dreamzero": HarenaDreamzeroPolicy,
            "harena_fastwam_predict": HarenaFastwamPredictPolicy,
        }

    def test_both_world_models_answer_every_call_the_tool_makes(self):
        for name, cls in self.policies().items():
            for method in self.POLICY_METHODS:
                with self.subTest(policy=name, method=method):
                    self.assertTrue(
                        callable(getattr(cls, method, None)),
                        f"{name} has no {method}(); eval_world_model.py would "
                        "report it as not a world model",
                    )
            for field in self.CONFIG_FIELDS:
                with self.subTest(policy=name, field=field):
                    self.assertTrue(
                        hasattr(cls.config_class, field),
                        f"{name}'s config has no {field}",
                    )

    def test_a_policy_that_only_predicts_actions_does_not_answer(self):
        # The refusal has to keep working, or the tool would try to score ACT
        # on a future it never computes and fail somewhere less legible.
        from actoris_harena.policies.act.modeling_act import HarenaActPolicy

        self.assertFalse(hasattr(HarenaActPolicy, "predict_future_frames"))


class SamplerPinTest(unittest.TestCase):
    """The sampler is pinned, because a world model's prediction is sampled.

    Both world models integrate from noise -- DreamZero from `torch.randn`,
    FastWAM through `infer_joint` -- so an unpinned run scores a different
    number every time. MEASURED 2026-09-16: two copies of the scoring script
    over one step-2000 checkpoint disagreed by up to 0.05 dB per camera while
    every held-last-frame baseline matched to the digit, which is exactly where
    the difference should show if it is the sampler and nowhere else.
    """

    class Recorder:
        """A stand-in whose 'prediction' is just the next random number."""

        class config:
            n_context_chunks = 1
            latent_frames_per_chunk = 1

        def predict_future_frames(self, batch):
            return torch.rand(1)

        def tile_cameras(self, batch):
            return torch.zeros(1, 2)

        def untile_cameras(self, tiled):
            return {}

    def draw(self, seed, reset=True):
        from tool.eval_world_model import evaluate_frame

        policy = self.Recorder()
        if reset:
            # Start each pinned draw from the SAME unrelated state, so that
            # anything the two draws share came from the seed and not from
            # where the global generator happened to be.
            torch.manual_seed(12345)
        evaluate_frame(policy, {}, seed=seed)
        return float(torch.rand(1))

    def test_one_seed_gives_one_answer(self):
        self.assertEqual(self.draw(0), self.draw(0))

    def test_a_different_seed_gives_a_different_answer(self):
        # Otherwise the pin would be vacuous -- it would look pinned because
        # nothing was random, not because the seed took effect.
        self.assertNotEqual(self.draw(0), self.draw(1))

    def test_seed_none_leaves_the_sampler_free(self):
        # --seed -1 exists to MEASURE the spread, so it must genuinely not pin.
        # No reset here, deliberately: resetting the generator before each draw
        # would pin the run from OUTSIDE and the test would pass while proving
        # nothing -- which is how this test read on its first attempt.
        torch.manual_seed(999)
        self.assertNotEqual(self.draw(None, reset=False), self.draw(None, reset=False))
