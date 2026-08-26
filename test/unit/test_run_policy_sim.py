"""Unit tests for how the sim sweep folds and ranks its episodes.

The folding is where a sweep can lie. Three ways it did or could:

* ``success`` as ``all(...)`` -- twenty-nine wins out of thirty and zero out of
  thirty both read as False, so the whole comparison collapses to "did anything
  fail", which on a hard task is always yes.
* time-to-success averaged over every episode -- a failure has no finishing
  time, and counting its timeout makes the column a second, worse-scaled
  success rate.
* a ranking by seam ratio -- the prettiest joins belong to the cell that ignores
  the newest plan hardest, which is not the cell to deploy.

None of this needs a twin, a policy or a network: it is arithmetic over rows.
"""

import unittest

from tool.run_policy_sim import _fold, _rank, _table, _video_name, is_session_conflict


def episode(
    cell: str,
    success: bool,
    ticks_to_success=None,
    *,
    seed: int = 0,
    seam: float = 2.0,
    fps: float = 30.0,
    place_err_mm: float = 5.0,
    held: float = 0.1,
    path: float = 10.0,
    rtt: float = 120.0,
    wall: float = 4.0,
):
    """One per-episode row shaped like the one ``run_episode`` returns."""
    return {
        "cell": cell,
        "strategy": cell.split("@")[0],
        "seed": seed,
        "success": success,
        "ticks_to_success": ticks_to_success,
        "fps": fps,
        "seam_ratio": seam,
        "held_fraction": held,
        "path_length": path,
        "round_trip_ms_median": rtt,
        "place_err_mm": place_err_mm,
        "wall_s": wall,
    }


class TestSuccessRate(unittest.TestCase):
    def test_success_is_a_rate_not_a_verdict(self):
        rows = [episode("blend@w5/linear", i < 3, 60) for i in range(10)]
        (folded,) = _fold(rows)
        self.assertIsInstance(folded["success"], float)
        self.assertAlmostEqual(folded["success"], 0.3)
        self.assertEqual(folded["successes"], 3)
        self.assertEqual(folded["episodes"], 10)

    def test_all_failed_is_zero_and_all_won_is_one(self):
        (none,) = _fold([episode("append", False) for _ in range(4)])
        self.assertEqual(none["success"], 0.0)
        (every,) = _fold([episode("append", True, 30) for _ in range(4)])
        self.assertEqual(every["success"], 1.0)

    def test_cells_are_folded_apart(self):
        rows = [
            episode("receding@0.25", True, 30),
            episode("receding@0.5", False),
            episode("receding@0.5", True, 90),
        ]
        folded = {row["cell"]: row for row in _fold(rows)}
        self.assertEqual(set(folded), {"receding@0.25", "receding@0.5"})
        self.assertAlmostEqual(folded["receding@0.5"]["success"], 0.5)
        self.assertEqual(folded["receding@0.25"]["base_strategy"], "receding")


class TestTimeToSuccess(unittest.TestCase):
    def test_the_median_ignores_the_episodes_that_never_finished(self):
        rows = [
            episode("blend@w5/exp", True, 30),
            episode("blend@w5/exp", True, 60),
            episode("blend@w5/exp", True, 90),
            episode("blend@w5/exp", False, None),
            episode("blend@w5/exp", False, None),
        ]
        (folded,) = _fold(rows)
        self.assertEqual(folded["ticks_to_success_median"], 60.0)
        self.assertAlmostEqual(folded["seconds_to_success_median"], 2.0)

    def test_an_even_number_of_wins_averages_the_middle_pair(self):
        rows = [episode("sync", True, t) for t in (30, 60, 90, 120)]
        (folded,) = _fold(rows)
        self.assertEqual(folded["ticks_to_success_median"], 75.0)

    def test_a_cell_that_never_succeeded_reports_no_time(self):
        # Not a zero, and above all not a division by the empty set.
        (folded,) = _fold([episode("append", False) for _ in range(3)])
        self.assertIsNone(folded["ticks_to_success_median"])
        self.assertIsNone(folded["seconds_to_success_median"])

    def test_seconds_follow_the_rate_the_run_was_paced_at(self):
        rows = [episode("sync", True, 50, fps=25.0) for _ in range(3)]
        (folded,) = _fold(rows)
        self.assertAlmostEqual(folded["seconds_to_success_median"], 2.0)


class TestOtherFolds(unittest.TestCase):
    def test_place_error_is_a_mean_over_every_episode(self):
        rows = [
            episode("append", True, 30, place_err_mm=10.0),
            episode("append", False, place_err_mm=20.0),
        ]
        (folded,) = _fold(rows)
        self.assertAlmostEqual(folded["place_err_mm"], 15.0)

    def test_wall_clock_is_the_total_the_cell_cost(self):
        rows = [episode("append", True, 30, wall=2.5) for _ in range(4)]
        (folded,) = _fold(rows)
        self.assertAlmostEqual(folded["wall_s"], 10.0)

    def test_a_run_with_no_measurable_seam_folds_to_none(self):
        rows = [episode("sync", True, 30, seam=None) for _ in range(3)]
        (folded,) = _fold(rows)
        self.assertIsNone(folded["seam_ratio"])
        self.assertIsNone(
            _fold([episode("sync", True, 30, rtt=None)])[0]["round_trip_ms_median"]
        )


class TestRanking(unittest.TestCase):
    def fold(self, rows):
        return [row["cell"] for row in _fold(rows)]

    def test_success_rate_beats_everything_else(self):
        rows = [
            # Beautiful joins, finishes fast, but only ever wins one in four.
            episode("ensemble@0.3", True, 30, seam=1.0),
            *[episode("ensemble@0.3", False, seam=1.0) for _ in range(3)],
            # Ugly joins and slow, but always gets there.
            *[episode("replace", True, 300, seam=8.0) for _ in range(4)],
        ]
        self.assertEqual(self.fold(rows), ["replace", "ensemble@0.3"])

    def test_time_to_success_breaks_a_tie_on_success_rate(self):
        rows = [
            episode("blend@w10/exp", True, 200, seam=1.1),
            episode("blend@w3/linear", True, 100, seam=4.0),
        ]
        self.assertEqual(self.fold(rows), ["blend@w3/linear", "blend@w10/exp"])

    def test_seam_ratio_only_breaks_a_remaining_tie(self):
        rows = [
            episode("blend@w3/linear", True, 100, seam=4.0),
            episode("blend@w10/exp", True, 100, seam=1.1),
        ]
        self.assertEqual(self.fold(rows), ["blend@w10/exp", "blend@w3/linear"])

    def test_a_cell_that_never_finished_sorts_last_among_its_equals(self):
        rows = [
            episode("append", False),
            episode("sync", False),
            episode("replace", False),
        ]
        # All zero success and no finishing time: the order is then stable by
        # name rather than arbitrary, so two runs print the same table.
        self.assertEqual(self.fold(rows), ["append", "replace", "sync"])

    def test_rank_is_the_key_the_table_sorts_by(self):
        rows = _fold(
            [
                episode("append", False),
                episode("replace", True, 100),
            ]
        )
        self.assertEqual(rows, sorted(rows, key=_rank))


class TestTable(unittest.TestCase):
    def test_the_table_shows_the_finishing_time_and_the_rate(self):
        table = _table(
            _fold(
                [
                    episode("replace", True, 60),
                    episode("replace", False),
                ]
            )
        )
        self.assertIn("median t→done", table)
        self.assertIn("50% (1/2)", table)
        self.assertIn("2.00 s", table)

    def test_a_cell_with_no_finish_prints_a_dash_rather_than_a_zero(self):
        table = _table(_fold([episode("append", False)]))
        self.assertIn("0% (0/1)", table)
        self.assertIn("| — |", table)

    def test_no_runs_says_so(self):
        self.assertEqual(_table([]), "_no runs_\n")

    def test_the_table_keeps_the_seam_held_and_round_trip_columns(self):
        table = _table(_fold([episode("blend@w5/exp", True, 60)]))
        for column in ("seam ratio", "held", "path", "rtt median"):
            self.assertIn(column, table)


class TestVideoName(unittest.TestCase):
    def test_a_cell_label_becomes_a_legal_filename(self):
        name = _video_name("blend@w5/exp", 10001, 3)
        self.assertNotIn("/", name)
        self.assertTrue(name.endswith("_seed10001_3.mp4"))


if __name__ == "__main__":
    unittest.main()


class TestSessionConflict(unittest.TestCase):
    """Which refusals a long sweep may retry, and which end it."""

    def test_a_stolen_session_slot_is_recognised(self):
        self.assertTrue(
            is_session_conflict(
                "HTTP 409: session is not the one that reset this server"
            )
        )

    def test_the_status_code_alone_is_enough(self):
        self.assertTrue(is_session_conflict("HTTP 409: conflict"))

    def test_a_bad_observation_is_not_retried(self):
        # Repeating a request the host called malformed only wastes the hours.
        self.assertFalse(is_session_conflict("HTTP 400: observation carries cameras"))

    def test_nor_is_a_missing_route(self):
        self.assertFalse(is_session_conflict("HTTP 404: not found"))
