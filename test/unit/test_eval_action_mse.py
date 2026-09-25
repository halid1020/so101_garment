"""Scoring a policy against the actions the operator actually performed.

Training loss cannot answer "is the cropped arm better": each policy minimises a
different objective, and a run trained on cropped inputs is fitting different
data from its baseline. The distance between a planned chunk and the recorded
one is comparable across all of them.

The part worth testing hardest is the SPLIT. LeRobot holds out the last
ceil(n * eval_split) episodes OF EACH TASK, and a validation set that is not the
one the trainer held out is worse than none -- it looks like evidence.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_eval_action_mse
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool.eval_action_mse import (
    chunk_errors,
    future_truth,
    held_out_episodes,
    needs_window,
    split_from_checkpoint,
    summarise,
    task_prompt,
)


class SplitMatchesLeRobotTest(unittest.TestCase):
    """Reproduced from lerobot/datasets/factory.py:make_train_eval_datasets."""

    def test_the_last_tenth_is_held_out(self):
        tasks = [["fold the short"]] * 65
        train, val = held_out_episodes(tasks, 0.1)
        # ceil(65 * 0.1) == 7, taken from the END.
        self.assertEqual(len(val), 7)
        self.assertEqual(val, list(range(58, 65)))
        self.assertEqual(len(train), 58)
        self.assertEqual(set(train) & set(val), set())

    def test_it_rounds_up_not_down(self):
        # ceil, so a split can never silently hold out nothing.
        tasks = [["a"]] * 11
        _, val = held_out_episodes(tasks, 0.05)
        self.assertEqual(len(val), 1)

    def test_it_splits_per_task_and_not_overall(self):
        """The difference that matters on a mixed dataset.

        Ten episodes of one task and ten of another at 0.1 gives ONE held-out
        episode from each, not two from whichever sorts last. A split done
        overall would hold out two of the same task and none of the other, and
        would disagree with the run it claims to describe.
        """
        tasks = [["fold"]] * 10 + [["place"]] * 10
        train, val = held_out_episodes(tasks, 0.1)
        self.assertEqual(val, [9, 19])
        self.assertEqual(len(train), 18)

    def test_no_split_holds_nothing_out(self):
        tasks = [["a"]] * 20
        train, val = held_out_episodes(tasks, 0.0)
        self.assertEqual(val, [])
        self.assertEqual(len(train), 20)

    def test_an_explicit_episode_subset_is_respected(self):
        # --episodes narrows what is considered before the split is applied.
        tasks = [["a"]] * 20
        train, val = held_out_episodes(tasks, 0.5, episodes=[0, 1, 2, 3])
        self.assertEqual(val, [2, 3])
        self.assertEqual(train, [0, 1])

    def test_an_episode_with_no_task_still_groups(self):
        # Empty task lists must not raise; they form their own group.
        tasks = [[], [], ["a"], ["a"]]
        train, val = held_out_episodes(tasks, 0.5)
        self.assertEqual(sorted(val), [1, 3])


class ChunkErrorTest(unittest.TestCase):
    def test_a_perfect_plan_has_no_error(self):
        truth = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.assertEqual(chunk_errors(truth.copy(), truth).sum(), 0.0)

    def test_the_error_is_squared_per_element(self):
        planned = np.zeros((2, 2))
        truth = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            chunk_errors(planned, truth), [[1.0, 4.0], [9.0, 16.0]]
        )

    def test_it_trims_to_the_shorter(self):
        """A chunk running past the end of its episode has nothing to be wrong about.

        Padding the truth with the last action would credit the policy for
        holding still exactly where the episode stopped.
        """
        planned = np.zeros((10, 2))
        truth = np.ones((3, 2))
        self.assertEqual(len(chunk_errors(planned, truth)), 3)

    def test_no_overlap_is_empty_rather_than_an_error(self):
        self.assertEqual(len(chunk_errors(np.zeros((5, 2)), np.zeros((0, 2)))), 0)


class SummaryTest(unittest.TestCase):
    def test_the_numbers_are_what_they_say(self):
        # Two frames, horizon 2, 2 dims, every squared error 4.0 -> MSE 4, RMSE 2.
        errors = [np.full((2, 2), 4.0), np.full((2, 2), 4.0)]
        out = summarise(errors, 2)
        self.assertEqual(out["frames"], 2)
        self.assertAlmostEqual(out["mse"], 4.0)
        self.assertAlmostEqual(out["rmse"], 2.0)
        self.assertEqual(len(out["mse_per_step"]), 2)
        self.assertEqual(len(out["mse_per_dim"]), 2)

    def test_a_short_chunk_does_not_drag_later_steps_down(self):
        """Averaging must divide by how many frames REACHED each step.

        One frame with a two-step chunk and one with a ten-step chunk: step 10
        was measured once, and dividing its total by two would halve it and
        make the horizon look flatter than it is.
        """
        errors = [np.full((10, 1), 1.0), np.full((2, 1), 1.0)]
        out = summarise(errors, 1)
        np.testing.assert_allclose(out["mse_per_step"], [1.0] * 10)

    def test_the_horizon_is_kept_rather_than_averaged_away(self):
        rising = [np.arange(5, dtype=float).reshape(5, 1)]
        out = summarise(rising, 1)
        self.assertLess(out["first_step_mse"], out["last_step_mse"])

    def test_grippers_are_reported_apart_from_joints(self):
        """They span tenths where the joints span radians.

        A mean over all twelve dimensions is dominated by the joints, and a
        grasp is won or lost in the part it buries.
        """
        errors = [np.zeros((1, 12))]
        errors[0][0, 5] = 9.0  # a gripper column
        errors[0][0, 11] = 9.0
        out = summarise(errors, 12)
        self.assertAlmostEqual(out["mse_joints"], 0.0)
        self.assertAlmostEqual(out["mse_grippers"], 9.0)

    def test_no_frames_says_so_rather_than_dividing_by_zero(self):
        self.assertEqual(summarise([], 12), {"frames": 0})


class TaskPromptTest(unittest.TestCase):
    """An empty prompt is a different instruction, not no instruction."""

    def test_a_given_prompt_wins(self):
        self.assertEqual(task_prompt("fold it", [["fold the short"]]), "fold it")

    def test_a_single_task_dataset_supplies_its_own(self):
        tasks = [["fold the short"], ["fold the short"]]
        self.assertEqual(task_prompt("", tasks), "fold the short")

    def test_several_tasks_are_refused_rather_than_one_picked(self):
        with self.assertRaises(SystemExit):
            task_prompt("", [["fold the short"], ["stack the cups"]])

    def test_a_dataset_with_no_tasks_stays_unprompted(self):
        self.assertEqual(task_prompt("", [[], []]), "")


class WindowedTruthTest(unittest.TestCase):
    """A world model's window holds past actions; only the future is scored."""

    def test_the_past_is_context_and_not_scored(self):
        item = {"action": np.arange(6.0).reshape(6, 1)}
        truth = future_truth(item, [-2, -1, 0, 1, 2, 3])
        self.assertEqual(truth.ravel().tolist(), [2.0, 3.0, 4.0, 5.0])

    def test_padding_past_the_episode_end_is_cut(self):
        item = {
            "action": np.arange(5.0).reshape(5, 1),
            "action_is_pad": np.array([False, False, False, True, True]),
        }
        truth = future_truth(item, [-1, 0, 1, 2, 3])
        self.assertEqual(truth.ravel().tolist(), [1.0, 2.0])

    def test_only_a_world_model_with_a_window_takes_the_windowed_path(self):
        class Config:
            observation_delta_indices = [-48, -24, 0, 24]

        class WorldModel:
            config = Config()

            def predict_future_frames(self, batch):
                return None

        class Policy:
            config = Config()

        self.assertTrue(needs_window(WorldModel()))
        # A policy with a multi-step window but no world model is left to the
        # single-frame path, which repeats the frame as it always has.
        self.assertFalse(needs_window(Policy()))


class SplitFromCheckpointTest(unittest.TestCase):
    def test_it_reads_the_run_s_own_split(self):
        d = Path(tempfile.mkdtemp())
        (d / "train_config.json").write_text(
            json.dumps({"dataset": {"eval_split": 0.1}})
        )
        self.assertEqual(split_from_checkpoint(str(d)), 0.1)

    def test_a_run_that_held_nothing_out_reads_as_zero(self):
        # Every checkpoint trained on this rig so far, which is why the tool
        # has to say so rather than quietly reporting training data as val.
        d = Path(tempfile.mkdtemp())
        (d / "train_config.json").write_text(
            json.dumps({"dataset": {"eval_split": 0.0}})
        )
        self.assertEqual(split_from_checkpoint(str(d)), 0.0)

    def test_a_missing_or_unreadable_config_is_zero_not_a_crash(self):
        self.assertEqual(split_from_checkpoint(tempfile.mkdtemp()), 0.0)


if __name__ == "__main__":
    unittest.main()
