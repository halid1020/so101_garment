"""The world-model evaluation tool's pure parts.

Loading a checkpoint and decoding video belong to the integration tier; the
range parsing, the summarising and the report are arithmetic and string work,
and they are where a wrong answer would be quiet rather than loud.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tool.eval_world_model import (
    as_uint8,
    check_coverage,
    draw_filmstrip,
    load_filmstrip,
    make_batch,
    parse_range,
    preprocess,
    report,
    save_filmstrip,
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

    def test_saved_arrays_come_back_unchanged(self):
        import tempfile

        kept = self.kept()
        with tempfile.TemporaryDirectory() as tmp:
            back = load_filmstrip(save_filmstrip(kept, Path(tmp) / "s.npz"))
        self.assertEqual(sorted(back), sorted(kept))
        self.assertTrue((back["left"]["held"] == kept["left"]["held"]).all())
        self.assertTrue(
            (back["right"]["predicted"][1] == kept["right"]["predicted"][1]).all()
        )

    def test_a_subset_of_cameras_can_be_drawn(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = draw_filmstrip(self.kept(), Path(tmp) / "s.png", cameras=["right"])
            self.assertGreater(out.stat().st_size, 1000)

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

    def test_a_model_with_its_own_seed_field_is_seeded_too(self):
        from tool.eval_world_model import evaluate_frame

        policy = self.Recorder()
        policy.config.predict_seed = 99
        evaluate_frame(policy, {}, seed=5)
        self.assertEqual(policy.config.predict_seed, 5)
        del policy.config.predict_seed

    def test_seed_none_leaves_the_sampler_free(self):
        # --seed -1 exists to MEASURE the spread, so it must genuinely not pin.
        # No reset here, deliberately: resetting the generator before each draw
        # would pin the run from OUTSIDE and the test would pass while proving
        # nothing -- which is how this test read on its first attempt.
        torch.manual_seed(999)
        self.assertNotEqual(self.draw(None, reset=False), self.draw(None, reset=False))


class CroppedTruthTest(unittest.TestCase):
    """A model trained on cropped fingertips is scored against cropped truth."""

    KEY = "observation.images.left_arm_left_gripper"

    def rimmed(self):
        img = torch.zeros(1, 2, 3, 20, 20)
        img[..., :2, :] = 1.0
        img[..., -2:, :] = 1.0
        return img

    def test_the_checkpoint_s_crop_reaches_the_truth(self):
        from actoris_harena.policies.common.tactile import (
            HarenaTactileCropProcessorStep,
        )

        from tool.eval_world_model import truth_batch

        class Pipeline:
            steps = [HarenaTactileCropProcessorStep(fraction=(0.6, 1.0)), lambda b: b]

        out = truth_batch(Pipeline(), {self.KEY: self.rimmed(), "task": "fold"})
        self.assertEqual(float(out[self.KEY].max()), 0.0)
        self.assertEqual(out["task"], "fold")

    def test_nothing_else_in_the_pipeline_touches_it(self):
        # Normalisation must never reach the truth; only the crop does.
        from tool.eval_world_model import truth_batch

        class Pipeline:
            steps = [lambda b: {k: v * 0 for k, v in b.items()}]

        raw = {self.KEY: self.rimmed()}
        self.assertTrue(
            torch.equal(truth_batch(Pipeline(), raw)[self.KEY], raw[self.KEY])
        )


class SplitCompositesTest(unittest.TestCase):
    def test_each_tile_becomes_its_own_sensor(self):
        from tool.eval_world_model import split_composites

        quad = torch.zeros(1, 1, 3, 4, 6)
        for index in range(4):
            row, col = divmod(index, 2)
            quad[..., row * 2 : (row + 1) * 2, col * 3 : (col + 1) * 3] = index
        views = {
            "observation.images.tactile_quad": quad,
            "observation.images.central": quad,
        }
        out = split_composites(views, {"tactile_quad": ("a", "b", "c", "d")})
        self.assertEqual(
            sorted(out), ["a", "b", "c", "d", "observation.images.central"]
        )
        for index, name in enumerate("abcd"):
            self.assertEqual(tuple(out[name].shape), (1, 1, 3, 2, 3))
            self.assertTrue(torch.all(out[name] == index), name)


class ReconstructionTest(unittest.TestCase):
    """The observed frames through the model's own autoencoder: the ceiling."""

    class Blurry:
        class config:
            n_context_chunks = 1
            latent_frames_per_chunk = 1

        def tile_cameras(self, batch):
            return batch["frames"]

        def predict_future_frames(self, batch):
            return batch["frames"][:, 1:]

        def untile_cameras(self, tiled):
            return {"central": tiled}

        def reconstruct_frames(self, frames):
            return frames * 0.5

    def test_a_perfect_autoencoder_is_not_needed_to_score_one(self):
        from tool.eval_world_model import evaluate_frame

        frames = torch.rand(1, 3, 3, 16, 16) * 0.5 + 0.25
        row = evaluate_frame(self.Blurry(), {"frames": frames}, seed=0)["central"]
        self.assertEqual(len(row["recon_psnr"]), 1)
        self.assertLess(row["recon_psnr"][0], 40.0)
        self.assertIn("recon_ssim", row)

    def test_both_world_models_can_reconstruct(self):
        from actoris_harena.policies.dreamzero.modeling_dreamzero import (
            HarenaDreamzeroPolicy,
        )
        from actoris_harena.policies.fastwam_predict.modeling_fastwam_predict import (
            HarenaFastwamPredictPolicy,
        )

        for cls in (HarenaDreamzeroPolicy, HarenaFastwamPredictPolicy):
            self.assertTrue(callable(getattr(cls, "reconstruct_frames", None)), cls)

    def test_the_held_frame_s_reconstruction_is_kept_for_the_filmstrip(self):
        import tempfile

        from tool.eval_world_model import evaluate_frame

        frames = torch.full((1, 3, 3, 16, 16), 0.5)
        keep: list = []
        evaluate_frame(self.Blurry(), {"frames": frames}, seed=0, keep=keep)
        recon = keep[0]["central"]["held_recon"]
        self.assertEqual(int(recon.max()), 64)  # 0.5 halved -> 0.25 -> 64
        with tempfile.TemporaryDirectory() as tmp:
            back = load_filmstrip(save_filmstrip(keep[0], Path(tmp) / "s.npz"))
            self.assertTrue((back["central"]["held_recon"] == recon).all())
            drawn = draw_filmstrip(keep[0], Path(tmp) / "s.png")
            self.assertGreater(drawn.stat().st_size, 1000)

    def test_a_difference_image_is_grey_where_the_frames_agree(self):
        from tool.eval_world_model import difference_image

        same = np.full((2, 2, 3), 200, dtype=np.uint8)
        self.assertEqual(int(difference_image(same, same).max()), 128)
        black = np.zeros_like(same)
        white = np.full_like(same, 255)
        self.assertEqual(int(difference_image(white, black).min()), 255)
        self.assertEqual(int(difference_image(black, white).max()), 0)

    def test_the_report_shows_the_ceiling(self):
        summary = {
            "central": {**frame([20.0], [18.0])["central"], "recon_psnr": [27.5]}
        }
        self.assertIn("27.50", report(summary))


class GripperExperimentTest(unittest.TestCase):
    """The controls of the grasp experiment change only what they name."""

    def test_overrides_parse_numbers_and_the_dataset_range(self):
        from tool.eval_world_model import parse_overrides

        stats = {
            "observation.state": {
                "min": [0.0] * 12,
                "max": [float(i) for i in range(12)],
            }
        }
        self.assertEqual(parse_overrides("5=0.02,11=max", stats), {5: 0.02, 11: 11.0})
        self.assertEqual(parse_overrides(""), {})
        with self.assertRaises(ValueError):
            parse_overrides("5=min")

    def test_state_override_touches_only_its_columns(self):
        from tool.eval_world_model import override_state

        batch = {"observation.state": torch.zeros(1, 3, 12)}
        out = override_state(batch, {5: 1.0})
        self.assertTrue(torch.all(out["observation.state"][..., 5] == 1.0))
        self.assertEqual(float(out["observation.state"].sum()), 3.0)
        self.assertEqual(float(batch["observation.state"].sum()), 0.0)

    def test_hold_gripper_freezes_only_the_grippers(self):
        from tool.eval_world_model import conditioning_actions

        actions = torch.arange(40 * 12, dtype=torch.float32).view(1, 40, 12)
        batch = {"action": actions}
        self.assertIsNone(conditioning_actions(batch, 32, "none", (5, 11)))
        recorded = conditioning_actions(batch, 32, "recorded", (5, 11))
        self.assertTrue(torch.equal(recorded, actions[:, :32]))
        held = conditioning_actions(batch, 32, "hold-gripper", (5, 11))
        self.assertTrue(torch.all(held[0, :, 5] == actions[0, 0, 5]))
        self.assertTrue(torch.equal(held[..., :5], actions[:, :32, :5]))
