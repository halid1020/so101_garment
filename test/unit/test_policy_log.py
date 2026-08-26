"""The run log: what a rollout leaves behind for afterwards.

A failed grasp cannot be paused mid-air, so the log is the only chance to see it
again. Two properties matter and are checked here: the plans reach disk as they
land (a run that dies still leaves them), and one plan is written once however
many ticks execute it.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_log
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.policy_log import RunLog


class StubSource:
    round_trip_s = 0.55
    server_infer_s = 0.52

    def __init__(self):
        self.last_chunk = np.zeros((4, 12))
        self.last_chunk_seq = 1
        self.last_chunk_at = 1000.0

    def last_sent(self):
        return (np.arange(12.0), {})


class TestRunLog(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.log = RunLog.create(
            task="pick up the cube", source="remote 'act'", hz=30.0, root=self.tmp
        )
        self.addCleanup(self.log.close)

    def chunks(self):
        text = (self.log.root / "chunks.jsonl").read_text().splitlines()
        return [json.loads(line) for line in text if line.strip()]

    def test_the_run_is_named_and_described(self):
        meta = json.loads((self.log.root / "meta.json").read_text())

        self.assertEqual(meta["task"], "pick up the cube")
        self.assertEqual(meta["hz"], 30.0)

    def test_a_tick_keeps_both_what_was_measured_and_what_was_sent(self):
        import pandas as pd

        self.log.tick(0.0, np.arange(12.0), np.arange(12.0) + 1, "run", 30)
        self.log.tick(0.033, np.arange(12.0), None, "hold", 0)
        root = self.log.close()

        frame = pd.read_parquet(root / "ticks.parquet")
        self.assertEqual(list(frame["mode"]), ["run", "hold"])
        self.assertEqual(list(frame["served"]), [True, False])
        self.assertEqual(len(frame["state"][0]), 12)
        self.assertIsNone(frame["commanded"][1])

    def test_every_tick_says_which_attempt_it_belongs_to(self):
        import pandas as pd

        self.log.tick(0.0, np.arange(12.0), np.arange(12.0), "run", 30)
        self.assertEqual(self.log.new_trial(), 1)
        self.log.tick(1.0, np.arange(12.0), np.arange(12.0), "run", 30)
        root = self.log.close()

        # Without this column two attempts at one scene read as a single long
        # series with an unexplained pause in the middle of it.
        frame = pd.read_parquet(root / "ticks.parquet")
        self.assertEqual(list(frame["trial"]), [0, 1])

    def test_a_plan_is_written_once_however_often_it_is_offered(self):
        source = StubSource()

        self.assertTrue(self.log.note_chunk(source))
        self.assertFalse(self.log.note_chunk(source))
        source.last_chunk_seq = 2
        self.assertTrue(self.log.note_chunk(source))

        self.assertEqual([c["seq"] for c in self.chunks()], [1, 2])

    def test_a_plan_reaches_disk_before_the_run_ends(self):
        # A rollout that dies mid-episode is exactly when the log matters.
        self.log.note_chunk(StubSource())

        written = self.chunks()
        self.assertEqual(len(written[0]["actions"]), 4)
        self.assertEqual(len(written[0]["state"]), 12)
        self.assertEqual(written[0]["round_trip_s"], 0.55)

    def test_a_source_with_no_plan_yet_writes_nothing(self):
        class Empty:
            last_chunk = None
            last_chunk_seq = 0
            round_trip_s = 0.0

        self.assertFalse(self.log.note_chunk(Empty()))
        self.assertEqual(self.chunks(), [])

    def test_closing_twice_is_harmless(self):
        self.log.tick(0.0, np.zeros(12), None, "run", 0)
        self.assertEqual(self.log.close(), self.log.close())


if __name__ == "__main__":
    unittest.main()
