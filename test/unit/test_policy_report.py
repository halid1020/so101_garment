"""Reading many rollouts back: the table an ablation is reported from.

Three conventions carry the whole meaning of this table, and each is checked
here: a discarded attempt leaves the denominator rather than counting against
the policy, a run nobody judged reads as unscored rather than as nought per
cent, and several runs of one checkpoint pool into one row.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_report
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from common.policy_log import RunLog
from tool.policy_report import by_checkpoint, collect, describe_checkpoint

CKPT = (
    "/home/ah390/.cache/huggingface/lerobot/so101_outputs/vla_real_long/"
    "fold-short-from-flattend-tactile__all/train/act/checkpoints/last/pretrained_model"
)


class TestNamingACheckpoint(unittest.TestCase):
    """A dozen identical path segments tell two runs apart not at all."""

    def test_the_run_directory_and_the_policy_are_what_identify_it(self):
        self.assertEqual(
            describe_checkpoint({"checkpoint": CKPT}),
            "fold-short-from-flattend-tactile__all/train/act",
        )

    def test_two_camera_sets_of_one_dataset_stay_distinct(self):
        central = CKPT.replace("__all", "__central")
        self.assertNotEqual(
            describe_checkpoint({"checkpoint": CKPT}),
            describe_checkpoint({"checkpoint": central}),
        )

    def test_a_run_from_before_this_was_recorded_says_so(self):
        self.assertEqual(describe_checkpoint({}), "(unrecorded)")
        self.assertEqual(describe_checkpoint({"policy_type": "act"}), "act")

    def test_a_path_with_no_train_segment_falls_back_to_its_name(self):
        self.assertEqual(describe_checkpoint({"checkpoint": "/x/y/weights"}), "weights")


class TestTheTable(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_log(self, name, checkpoint, outcomes):
        root = self.tmp / name
        log = RunLog.create(
            task="fold the short",
            source="remote 'act'",
            hz=30.0,
            root=self.tmp,
            checkpoint=checkpoint,
            policy_type="act",
        )
        # RunLog.create stamps its own directory; move it to a known name so the
        # test does not race two runs into one second.
        log.close()
        log.root.rename(root)
        for i, outcome in enumerate(outcomes):
            handle = open(root / "trials.jsonl", "a", encoding="utf-8")
            handle.write(
                json.dumps({"trial": i, "outcome": outcome, "notes": "", "t": 1.0})
                + "\n"
            )
            handle.close()
        return root

    def test_a_discarded_attempt_leaves_the_denominator(self):
        self.run_log("a", CKPT, ["success", "discard", "failure"])
        (record,) = collect(self.tmp)
        self.assertEqual((record["trials"], record["successes"]), (2, 1))
        self.assertEqual(record["discarded"], 1)
        self.assertEqual(record["rate"], 0.5)

    def test_a_run_nobody_judged_is_unscored_not_nought(self):
        self.run_log("a", CKPT, [])
        (record,) = collect(self.tmp)
        self.assertEqual(record["trials"], 0)
        self.assertIsNone(record["rate"])

    def test_runs_of_one_checkpoint_pool_into_one_row(self):
        self.run_log("a", CKPT, ["success", "failure"])
        self.run_log("b", CKPT, ["success", "success"])
        pooled = by_checkpoint(collect(self.tmp))
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0]["runs"], 2)
        self.assertEqual((pooled[0]["trials"], pooled[0]["successes"]), (4, 3))

    def test_two_checkpoints_stay_apart(self):
        self.run_log("a", CKPT, ["success"])
        self.run_log("b", CKPT.replace("__all", "__central"), ["failure"])
        pooled = by_checkpoint(collect(self.tmp))
        self.assertEqual(len(pooled), 2)
        self.assertEqual([p["successes"] for p in pooled], [1, 0])

    def test_a_filter_selects_by_checkpoint_or_by_stamp(self):
        self.run_log("aaa", CKPT, ["success"])
        self.run_log("bbb", CKPT.replace("__all", "__central"), ["failure"])
        self.assertEqual(len(collect(self.tmp, match="__central")), 1)
        self.assertEqual(len(collect(self.tmp, match="aaa")), 1)

    def test_a_run_with_no_meta_at_all_is_listed_rather_than_crashing(self):
        # Half-written directories exist; a report must survive them.
        (self.tmp / "broken").mkdir()
        (self.tmp / "broken" / "meta.json").write_text("{not json")
        (record,) = collect(self.tmp)
        self.assertEqual(record["checkpoint"], "(unrecorded)")

    def test_a_run_killed_before_close_has_no_durations_but_still_lists(self):
        # ticks.parquet is written by close(); a killed run never gets one.
        root = self.run_log("a", CKPT, ["success"])
        (root / "ticks.parquet").unlink()
        (record,) = collect(self.tmp)
        self.assertIsNone(record["mean_trial_s"])
        self.assertEqual(record["successes"], 1)


if __name__ == "__main__":
    unittest.main()
