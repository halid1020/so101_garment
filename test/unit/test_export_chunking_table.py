"""Unit tests for folding several sweep journals into one table."""

import tempfile
import unittest
from pathlib import Path

from common.sweep_journal import append_row
from tool.export_chunking_table import PROSE, latex_table, merge


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
