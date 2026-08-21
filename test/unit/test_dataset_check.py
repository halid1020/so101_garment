"""Integrity of a collected dataset, and the repair of an episode nobody wrote.

A recording counted in ``meta/info.json`` and never written makes LeRobot judge
the entire local copy incomplete, go to the Hub for a version tag, and -- offline
-- report a local gap as an unreachable huggingface.co. Every rewrite path runs
through that constructor, so this is checked here on synthetic datasets built
the way the recorder builds them: one metadata row and one data file per
episode, plus the side files under ``extra/``.

What the repair must get right is the bookkeeping nobody sees: the surviving
episodes end up contiguous, their offsets are the cumulative lengths again (the
invariant LeRobot slices episodes with), a reviewer's marks follow their
episodes through the renumbering, and the per-episode statistics keep their
exact parquet types rather than being round-tripped through pandas.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_dataset_check
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.recording.dataset_check import (
    DatasetDamaged,
    dataset_integrity,
    ensure_loadable,
    offsets_consistent,
    repair_phantom_episodes,
)
from common.recording.dataset_edit import read_soft_deleted, write_soft_deleted

LENGTH = 10


def _episode_row(index: int, length: int, start: int) -> "dict":
    return {
        "episode_index": index,
        "length": length,
        "dataset_from_index": start,
        "dataset_to_index": start + length,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": index,
        "data/chunk_index": 0,
        "data/file_index": index,
        "stats/episode_index/min": [float(index)],
        "stats/episode_index/max": [float(index)],
        "stats/episode_index/mean": [float(index)],
        "stats/episode_index/std": [0.0],
        "stats/episode_index/count": [float(length)],
    }


def build_dataset(root: Path, episodes: "list[int]", counted: "int | None" = None):
    """A dataset holding exactly ``episodes``, counted as ``counted``.

    Written the way the recorder writes it: one metadata file and one data file
    per episode, named after the episode, with the offsets following the
    episodes actually present -- so a dataset built with a gap has the same
    bookkeeping hole a phantom leaves behind.
    """
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "extra").mkdir(parents=True, exist_ok=True)

    start = 0
    for index in episodes:
        row = _episode_row(index, LENGTH, start)
        # The offsets are written as if the counted-but-absent episode occupied
        # its share of the frame space, which is exactly the hole in the real one.
        start = row["dataset_to_index"] + (LENGTH if (index + 1) not in episodes else 0)
        pq.write_table(
            pa.Table.from_pylist([row]),
            root / "meta" / "episodes" / "chunk-000" / f"file-{index:03d}.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([index] * LENGTH, pa.int64()),
                    "frame_index": pa.array(range(LENGTH), pa.int64()),
                }
            ),
            root / "data" / "chunk-000" / f"file-{index:03d}.parquet",
        )
        (root / "extra" / f"drift_{index:06d}.parquet").write_bytes(b"drift")
        (root / "extra" / f"uid_{index:06d}.json").write_text(
            json.dumps({"episode_uid": f"uid-{index}"})
        )

    total = len(episodes) if counted is None else counted
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": total,
                "total_frames": total * LENGTH,
                "splits": {"train": f"0:{total}"},
            }
        )
    )
    return root


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "cube-pnp-new"

    def healthy(self, n: int = 5) -> Path:
        return build_dataset(self.root, list(range(n)))

    def with_phantom(self, missing: int = 4, n: int = 8) -> Path:
        """Counted 0..n-1, but ``missing`` was never written -- the real fault."""
        return build_dataset(
            self.root, [i for i in range(n) if i != missing], counted=n
        )

    def rows(self) -> "list[dict]":
        out = []
        for path in sorted(
            (self.root / "meta" / "episodes" / "chunk-000").glob("file-*.parquet")
        ):
            out += pq.read_table(path).to_pylist()
        return sorted(out, key=lambda r: r["episode_index"])

    def info(self) -> dict:
        return json.loads((self.root / "meta" / "info.json").read_text())


class TestIntegrity(DatasetCase):
    def test_a_whole_dataset_is_reported_whole(self):
        self.healthy()

        report = dataset_integrity(self.root)

        self.assertTrue(report["ok"])
        self.assertEqual(report["phantom"], [])
        self.assertIn("all accounted for", report["summary"])

    def test_a_counted_but_unwritten_episode_is_found_and_named(self):
        self.with_phantom()

        report = dataset_integrity(self.root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["repairable"])
        self.assertEqual(report["phantom"], [4])
        self.assertEqual(report["frames_present"], 7 * LENGTH)
        self.assertIn("never written", report["summary"])

    def test_the_summary_does_not_mention_the_hub(self):
        # The whole point: a local gap must never be reported as a network fault.
        self.with_phantom()

        summary = dataset_integrity(self.root)["summary"].lower()

        self.assertNotIn("huggingface", summary)
        self.assertNotIn("offline", summary)

    def test_an_episode_with_data_but_no_metadata_is_not_repairable(self):
        self.healthy(4)
        # Delete one metadata row: its frames are real and cannot be invented.
        (self.root / "meta" / "episodes" / "chunk-000" / "file-002.parquet").unlink()

        report = dataset_integrity(self.root)

        self.assertFalse(report["ok"])
        self.assertFalse(report["repairable"])
        self.assertEqual(report["orphan_data"], [2])

    def test_offsets_are_checked_against_the_lengths(self):
        self.assertTrue(offsets_consistent({0: {"length": 3, "from": 0, "to": 3}}))
        self.assertFalse(offsets_consistent({0: {"length": 3, "from": 0, "to": 4}}))


class TestGuard(DatasetCase):
    def test_a_whole_dataset_passes(self):
        self.healthy()

        self.assertTrue(ensure_loadable(self.root)["ok"])

    def test_a_phantom_stops_a_rewrite_with_the_repair_named(self):
        self.with_phantom()

        with self.assertRaises(DatasetDamaged) as caught:
            ensure_loadable(self.root)

        self.assertIn("never written", str(caught.exception))
        self.assertIn("Repair", str(caught.exception))

    def test_damage_that_needs_a_decision_says_so_instead(self):
        self.healthy(4)
        (self.root / "meta" / "episodes" / "chunk-000" / "file-002.parquet").unlink()

        with self.assertRaises(DatasetDamaged) as caught:
            ensure_loadable(self.root)

        self.assertIn("not repaired", str(caught.exception).lower())


class TestRepair(DatasetCase):
    def test_the_survivors_end_up_contiguous(self):
        self.with_phantom()

        result = repair_phantom_episodes(self.root)

        self.assertTrue(result["changed"])
        self.assertEqual(result["dropped"], [4])
        self.assertEqual([r["episode_index"] for r in self.rows()], list(range(7)))
        self.assertTrue(dataset_integrity(self.root)["ok"])

    def test_the_offsets_become_the_cumulative_lengths_again(self):
        self.with_phantom()

        repair_phantom_episodes(self.root)

        rows = self.rows()
        self.assertEqual([r["dataset_from_index"] for r in rows][:3], [0, 10, 20])
        self.assertEqual(rows[-1]["dataset_to_index"], 7 * LENGTH)

    def test_the_counts_come_from_what_is_there(self):
        self.with_phantom()

        repair_phantom_episodes(self.root)

        info = self.info()
        self.assertEqual(info["total_episodes"], 7)
        self.assertEqual(info["total_frames"], 7 * LENGTH)
        self.assertEqual(info["splits"]["train"], "0:7")

    def test_the_frames_are_renumbered_with_their_episode(self):
        self.with_phantom()

        repair_phantom_episodes(self.root)

        # The file keeps its name; only what it says about itself changes.
        table = pq.read_table(self.root / "data" / "chunk-000" / "file-005.parquet")
        self.assertEqual(set(table.column("episode_index").to_pylist()), {4})

    def test_the_statistics_of_the_index_column_follow_it(self):
        self.with_phantom()

        repair_phantom_episodes(self.root)

        row = [r for r in self.rows() if r["episode_index"] == 4][0]
        self.assertEqual(row["stats/episode_index/min"], [4.0])
        self.assertEqual(row["stats/episode_index/max"], [4.0])
        self.assertEqual(row["stats/episode_index/count"], [float(LENGTH)])

    def test_the_side_files_are_renumbered_too(self):
        self.with_phantom()

        repair_phantom_episodes(self.root)

        extra = self.root / "extra"
        self.assertTrue((extra / "drift_000004.parquet").exists())
        self.assertFalse((extra / "drift_000007.parquet").exists())
        self.assertEqual(
            json.loads((extra / "uid_000004.json").read_text())["episode_uid"],
            "uid-5",
        )

    def test_a_reviewers_marks_follow_their_episodes(self):
        self.with_phantom()
        write_soft_deleted(self.root, [0, 5])

        repair_phantom_episodes(self.root)

        # Episode 5 became 4; the mark moved with the recording, not the number.
        self.assertEqual(read_soft_deleted(self.root), [0, 4])

    def test_the_parquet_types_survive_the_rewrite(self):
        self.with_phantom()
        before = pq.read_table(
            self.root / "meta" / "episodes" / "chunk-000" / "file-005.parquet"
        ).schema

        repair_phantom_episodes(self.root)

        after = pq.read_table(
            self.root / "meta" / "episodes" / "chunk-000" / "file-005.parquet"
        ).schema
        self.assertEqual(before, after)

    def test_a_whole_dataset_is_left_exactly_alone(self):
        self.healthy()
        before = {p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()}

        result = repair_phantom_episodes(self.root)

        self.assertFalse(result["changed"])
        self.assertEqual(
            {p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()},
            before,
        )

    def test_damage_that_needs_a_decision_is_refused(self):
        self.healthy(4)
        (self.root / "meta" / "episodes" / "chunk-000" / "file-002.parquet").unlink()

        with self.assertRaises(DatasetDamaged):
            repair_phantom_episodes(self.root)


if __name__ == "__main__":
    unittest.main()
