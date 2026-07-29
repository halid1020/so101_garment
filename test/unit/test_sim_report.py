"""Unit tests for the long-run report aggregator (pure python, tmpdir)."""

import json
import tempfile
import unittest
from pathlib import Path

from sim_datagen.report import collect, render_markdown


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class TestReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name)
        _write(
            self.run / "oracle_gate" / "single_teleop.json",
            {
                "task": "single",
                "oracle_mode": "teleop",
                "oracle_success_rate": 1.0,
                "episode_attempts": 30,
            },
        )
        _write(
            self.run / "collect_stats" / "full_single.json",
            {
                "episodes_collected": 1000,
                "per_seed_success_rate": 0.997,
                "episode_attempts": 1042,
            },
        )
        _write(
            self.run / "full" / "single" / "act" / "selected.json",
            {"step": "0080000", "val_success": 0.8, "checkpoint": "x"},
        )
        _write(
            self.run / "full" / "single" / "act" / "eval" / "results.json",
            {"success_rate": 0.73, "place_err_mm_mean": 14.2},
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_collect_gathers_cells(self):
        data = collect(self.run)
        self.assertIn("single_teleop", data["oracle_gate"])
        self.assertIn("full_single", data["collection"])
        cells = {(c["mode"], c["task"], c["policy"]) for c in data["cells"]}
        self.assertIn(("full", "single", "act"), cells)

    def test_markdown_renders_tables(self):
        md = render_markdown(collect(self.run))
        self.assertIn("73% (14 mm)", md)  # eval cell
        self.assertIn("| full | single | act | 0080000 | 80% |", md)  # selection
        self.assertIn("100%", md)  # oracle gate
        # Absent policies render as em-dashes, not errors.
        self.assertIn("—", md)

    def test_missing_cells_tolerated(self):
        with tempfile.TemporaryDirectory() as empty:
            data = collect(Path(empty))
            self.assertEqual(data["cells"], [])
            md = render_markdown(data)
            self.assertIn("Sim-VLA long-run results", md)


if __name__ == "__main__":
    unittest.main()
