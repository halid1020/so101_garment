"""The world-model evaluation tool's pure parts.

Loading a checkpoint and decoding video belong to the integration tier; the
range parsing, the summarising and the report are arithmetic and string work,
and they are where a wrong answer would be quiet rather than loud.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tool.eval_world_model import (
    as_uint8,
    check_coverage,
    draw_filmstrip,
    make_batch,
    parse_range,
    preprocess,
    report,
    summarise,
)


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


class EmptyResultTest(unittest.TestCase):
    """A run that scores nothing must not look like a run that scored.

    MEASURED 2026-09-18: a shape mismatch in the conditioning state made the
    scorer refuse all twenty-four sampled frames. It printed a table with no
    rows, wrote a result file, and exited zero with a tick -- so the only
    evidence was in a log nobody had to read. The frame-level exception exists
    for the occasional frame too near an episode edge to pad, and this
    distinguishes that from a fault applying to every frame.
    """

    def test_scoring_nothing_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            check_coverage(0, ["bad shape"] * 24)
        self.assertIn("scored no frames", str(caught.exception))

    def test_the_refusal_carries_the_reason_the_frames_gave(self):
        with self.assertRaises(SystemExit) as caught:
            check_coverage(0, ["`proprio` must be [D] or [1,D], got (1, 9, 12)"])
        self.assertIn("proprio", str(caught.exception))

    def test_the_refusal_names_it_a_fault_and_not_an_edge_case(self):
        with self.assertRaises(SystemExit) as caught:
            check_coverage(0, [])
        self.assertIn("not an edge case", str(caught.exception))

    def test_a_healthy_run_is_silent(self):
        self.assertEqual(check_coverage(24, []), "")

    def test_a_few_edge_frames_do_not_raise_a_warning(self):
        # Two skips against twenty-four scored is the case this must tolerate.
        self.assertEqual(check_coverage(24, ["edge", "edge"]), "")

    def test_a_majority_of_skips_warns_without_refusing(self):
        warning = check_coverage(2, ["edge"] * 22)
        self.assertIn("minority of the recording", warning)

    def test_the_skip_count_reaches_the_result_file(self):
        # So a reader of the result can see coverage without the log.
        body = Path("tool/eval_world_model.py").read_text()
        self.assertIn('"skipped": len(skipped),', body)


class BatchCarriesTheTaskTest(unittest.TestCase):
    """A tensors-only batch drops the task, and a prompted model then refuses.

    MEASURED 2026-09-19: every sampled frame was refused with "Either `prompt`
    or both `context/context_mask` must be provided", because the batch keeps
    only tensors and the task description is a string. The world model this
    scorer was written against conditions on a learned task embedding and never
    needed it.
    """

    def item(self):
        return {
            "observation.state": torch.zeros(9, 12),
            "observation.images.central": torch.zeros(9, 3, 8, 8),
            "task": "fold the garment",
            "episode_index": torch.tensor(58),
        }

    def test_the_task_survives(self):
        batch = make_batch(self.item(), "cpu")
        self.assertEqual(batch["task"], "fold the garment")

    def test_tensors_gain_a_batch_dimension(self):
        batch = make_batch(self.item(), "cpu")
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 9, 12))

    def test_the_task_is_not_given_a_batch_dimension(self):
        # It is a string; unsqueezing it is not defined and wrapping it in a
        # list would change what the model receives.
        batch = make_batch(self.item(), "cpu")
        self.assertIsInstance(batch["task"], str)

    def test_a_row_with_no_task_is_fine(self):
        batch = make_batch({"observation.state": torch.zeros(2)}, "cpu")
        self.assertNotIn("task", batch)

    def test_a_tensor_task_is_left_to_the_tensor_path(self):
        # Some datasets carry a task INDEX rather than a description; that is a
        # tensor and must keep its batch dimension like any other.
        batch = make_batch({"task": torch.tensor(3)}, "cpu")
        self.assertEqual(tuple(batch["task"].shape), (1,))


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


class ThePreprocessorIsAppliedTest(unittest.TestCase):
    """The model sees normalised inputs; the truth stays the recorded frames.

    Leaving the preprocessor out raised nothing and scored the 80 000-step
    DreamZero at noise level: raw joint angles in degrees went in where
    training had used normalised ones. MEASURED 2026-09-25 -- about 6 dB raw
    against 14 to 18 dB preprocessed, on the same frames and seed.
    """

    class Seer:
        """Records which batch each call was handed."""

        class config:
            n_context_chunks = 1
            latent_frames_per_chunk = 1

        def __init__(self):
            self.predicted_from = None
            self.truth_from = None

        def predict_future_frames(self, batch):
            self.predicted_from = batch
            return torch.zeros(1, 2)

        def tile_cameras(self, batch):
            self.truth_from = batch
            return torch.zeros(1, 2)

        def untile_cameras(self, tiled):
            return {}

    @staticmethod
    def halve(batch):
        return {
            k: v / 2 if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def test_preprocess_runs_the_checkpoints_own_processor(self):
        raw = {"observation.state": torch.tensor([100.0]), "task": "fold"}
        out = preprocess(self.halve, raw, "cpu")
        self.assertEqual(float(out["observation.state"]), 50.0)
        self.assertEqual(out["task"], "fold")
        # The raw batch is left alone: it is still where the truth comes from.
        self.assertEqual(float(raw["observation.state"]), 100.0)

    def test_the_model_is_handed_the_preprocessed_batch(self):
        from tool.eval_world_model import evaluate_frame

        policy = self.Seer()
        raw = {"observation.state": torch.tensor([100.0])}
        model_batch = preprocess(self.halve, raw, "cpu")
        evaluate_frame(policy, raw, seed=0, model_batch=model_batch)
        self.assertIs(policy.predicted_from, model_batch)

    def test_the_truth_comes_from_the_raw_batch(self):
        # A processor that rescaled images must not move the ground truth with
        # the prediction, or a wrong scale would score as agreement.
        from tool.eval_world_model import evaluate_frame

        policy = self.Seer()
        raw = {"observation.state": torch.tensor([100.0])}
        evaluate_frame(
            policy, raw, seed=0, model_batch=preprocess(self.halve, raw, "cpu")
        )
        self.assertIs(policy.truth_from, raw)


class FilmstripTest(unittest.TestCase):
    """The pictures are of the frames the numbers were computed on."""

    class Tiny:
        """Two cameras side by side; the prediction is the truth brightened."""

        class config:
            n_context_chunks = 1
            latent_frames_per_chunk = 1

        def tile_cameras(self, batch):
            return batch["video"]

        def predict_future_frames(self, batch):
            return (batch["video"][:, 1:] + 0.25).clamp(0, 1)

        def untile_cameras(self, tiled):
            half = tiled.shape[-1] // 2
            return {"a.left": tiled[..., :half], "a.right": tiled[..., half:]}

    def kept(self):
        from tool.eval_world_model import evaluate_frame

        video = torch.linspace(0, 0.5, 3).view(1, 3, 1, 1, 1).expand(1, 3, 3, 16, 32)
        keep: list = []
        evaluate_frame(self.Tiny(), {"video": video.clone()}, seed=0, keep=keep)
        return keep[0]

    def test_every_camera_keeps_held_actual_and_predicted(self):
        kept = self.kept()
        self.assertEqual(sorted(kept), ["left", "right"])
        self.assertEqual(len(kept["left"]["actual"]), 2)
        self.assertEqual(len(kept["left"]["predicted"]), 2)
        self.assertEqual(kept["left"]["held"].shape, (16, 16, 3))

    def test_held_is_the_last_observed_frame_and_predicted_is_the_prediction(self):
        kept = self.kept()
        self.assertEqual(int(kept["left"]["held"].max()), 0)
        # actual step 1 is 0.25 -> 64; predicted is that brightened by 0.25.
        self.assertEqual(int(kept["left"]["actual"][0].max()), 64)
        self.assertEqual(int(kept["left"]["predicted"][0].max()), 128)

    def test_as_uint8_is_height_width_channels(self):
        self.assertEqual(as_uint8(torch.ones(3, 2, 5)).shape, (2, 5, 3))

    def test_a_filmstrip_is_written(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = draw_filmstrip(self.kept(), Path(tmp) / "strip.png", "frame 0")
            self.assertGreater(out.stat().st_size, 1000)


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
