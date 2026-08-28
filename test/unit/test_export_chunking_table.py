"""Unit tests for folding several sweep journals into one table."""

import tempfile
import unittest
from pathlib import Path

from common.sweep_journal import append_row
from tool.export_chunking_table import (
    PROSE,
    latency_of,
    latency_table,
    latex_table,
    merge,
)


def episode(cell, trial, success=True, seam=2.0):
    return {
        "cell": cell,
        "trial": trial,
        "strategy": cell.split("@")[0],
        "success": success,
        "seam_ratio": seam,
        "held_fraction": 0.0,
        "path_length": 100.0,
        "place_err_mm": 5.0,
        "round_trip_ms_median": 700.0,
        "ticks_to_success": 900 if success else None,
        "fps": 30.0,
        "wall_s": 60.0,
    }


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.a = self.dir / "a.jsonl"
        self.b = self.dir / "b.jsonl"

    def test_one_journal_comes_back_whole(self):
        for i in range(3):
            append_row(self.a, episode("sync", i))
        self.assertEqual(len(merge([self.a])), 3)

    def test_two_journals_of_different_cells_are_added_together(self):
        append_row(self.a, episode("sync", 0))
        append_row(self.b, episode("append", 0))
        cells = {r["cell"] for r in merge([self.a, self.b])}
        self.assertEqual(cells, {"sync", "append"})

    def test_a_later_pass_replaces_an_earlier_one_rather_than_joining_it(self):
        # Ten episodes and thirty must not fold into a mean over forty: the
        # later, larger pass is the measurement, the earlier one was the probe.
        for i in range(2):
            append_row(self.a, episode("sync", i))
        for i in range(5):
            append_row(self.b, episode("sync", i))
        rows = merge([self.a, self.b])
        self.assertEqual(len(rows), 5)

    def test_a_cell_absent_from_the_later_pass_survives_it(self):
        append_row(self.a, episode("append", 0))
        append_row(self.b, episode("sync", 0))
        cells = {r["cell"] for r in merge([self.a, self.b])}
        self.assertIn("append", cells)

    def test_a_missing_journal_is_not_an_error(self):
        append_row(self.a, episode("sync", 0))
        self.assertEqual(len(merge([self.a, self.dir / "nope.jsonl"])), 1)

    def test_a_row_with_no_trial_index_is_skipped(self):
        append_row(self.a, {"cell": "sync", "success": True})
        self.assertEqual(merge([self.a]), [])


class TestLatexTable(unittest.TestCase):
    def folded(self):
        return [
            {
                "cell": "receding@0.5",
                "episodes": 30,
                "success": 0.9,
                "seconds_to_success_median": 44.2,
                "seam_ratio": 3.1,
                "held_fraction": 0.02,
                "path_length": 182.4,
            }
        ]

    def test_percentages_are_escaped_for_latex(self):
        out = latex_table(self.folded(), "cap", "tab:x")
        self.assertIn(r"90\%", out)
        self.assertNotIn("90%", out.replace(r"90\%", ""))

    def test_the_tuning_is_split_out_of_the_cell_label(self):
        out = latex_table(self.folded(), "cap", "tab:x")
        self.assertIn("0.5", out)
        self.assertNotIn("receding@0.5", out)

    def test_the_strategy_is_named_in_prose_not_in_our_vocabulary(self):
        out = latex_table(self.folded(), "cap", "tab:x")
        self.assertIn(PROSE["receding"], out)

    def test_a_cell_that_never_succeeded_shows_no_time(self):
        rows = self.folded()
        rows[0]["seconds_to_success_median"] = None
        rows[0]["success"] = 0.0
        self.assertIn("--", latex_table(rows, "cap", "tab:x"))

    def test_the_caption_and_label_are_carried_through(self):
        out = latex_table(self.folded(), "a caption", "tab:mine")
        self.assertIn(r"\caption{a caption}", out)
        self.assertIn(r"\label{tab:mine}", out)


if __name__ == "__main__":
    unittest.main()


class TestLatencyOf(unittest.TestCase):
    """A journal must be able to say what delay it was measured at."""

    def test_the_rows_are_believed_first(self):
        rows = [{"latency_ticks": 16, "pace": "virtual"}]
        self.assertEqual(latency_of(Path("whatever.jsonl"), rows), 16)

    def test_an_older_journal_falls_back_to_its_file_name(self):
        # Written before the pacing was recorded; the campaign named the files.
        self.assertEqual(latency_of(Path("stage_c_lat24.md.episodes.jsonl"), [{}]), 24)

    def test_a_realtime_journal_has_no_pinned_delay(self):
        rows = [{"pace": "realtime", "latency_ticks": None}]
        self.assertIsNone(latency_of(Path("stage_c_lat8.jsonl"), rows))

    def test_and_neither_has_an_unnamed_one(self):
        self.assertIsNone(latency_of(Path("stage_a.md.episodes.jsonl"), [{}]))


class TestLatencyTable(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def journal(self, lat, cell, successes, n=5):
        p = self.dir / f"stage_c_lat{lat}.md.episodes.jsonl"
        for i in range(n):
            row = episode(cell, i, success=i < successes)
            row["latency_ticks"] = lat
            row["pace"] = "virtual"
            append_row(p, row)
        return p

    def test_the_delays_become_the_columns_in_order(self):
        a = self.journal(8, "sync", 5)
        b = self.journal(32, "sync", 0)
        out = latency_table([b, a], "cap", "tab:x")
        self.assertIn("8 ticks & 32 ticks", out)

    def test_a_cell_missing_at_one_delay_is_dashed_not_dropped(self):
        a = self.journal(8, "sync", 5)
        b = self.journal(32, "append", 0)
        out = latency_table([a, b], "cap", "tab:x")
        self.assertIn("--", out)
        self.assertIn(PROSE["sync"], out)
        self.assertIn(PROSE["append"], out)

    def test_the_held_metric_can_be_asked_for_instead(self):
        a = self.journal(8, "sync", 5)
        out = latency_table([a], "cap", "tab:x", metric="held")
        self.assertIn(r"0\%", out)

    def test_a_realtime_journal_is_left_out_of_a_latency_table(self):
        a = self.journal(8, "sync", 5)
        rt = self.dir / "stage_a.md.episodes.jsonl"
        row = episode("sync", 0)
        row["pace"] = "realtime"
        row["latency_ticks"] = None
        append_row(rt, row)
        out = latency_table([a, rt], "cap", "tab:x")
        self.assertIn("8 ticks", out)
        self.assertNotIn("None", out)
