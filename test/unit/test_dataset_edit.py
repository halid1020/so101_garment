"""Unit tests for the pure curation helpers of common.recording.dataset_edit.

Mostly no LeRobot and no filesystem: exercises the deletion re-indexing
arithmetic that keeps our ``extra/`` side files aligned with LeRobot's renumbered
dataset. The soft-delete marker tests do touch a temporary directory, because the
atomic-write behaviour is the point of them.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.recording.dataset_edit import (
    SOFT_DELETE_REL,
    ReadOnlyDatasetError,
    clear_soft_deleted,
    commit_episode_metadata,
    delete_episodes_in_place,
    deletion_mapping,
    episode_lengths,
    extra_reindex_ops,
    new_episode_uid,
    read_episode_lengths,
    read_episode_uid,
    read_soft_deleted,
    surviving_indices,
    write_episode_uid,
    write_soft_deleted,
)


class TestDeletionMapping(unittest.TestCase):
    def test_middle_delete_renumbers_tail(self):
        # delete ep2 of 0..4 -> keep 0,1,3,4 renumbered 0,1,2,3
        self.assertEqual(deletion_mapping(5, [2]), {0: 0, 1: 1, 3: 2, 4: 3})

    def test_first_delete_shifts_all_down(self):
        self.assertEqual(deletion_mapping(3, [0]), {1: 0, 2: 1})

    def test_last_delete_leaves_prefix_unchanged(self):
        self.assertEqual(deletion_mapping(3, [2]), {0: 0, 1: 1})

    def test_batch_delete(self):
        self.assertEqual(deletion_mapping(5, [1, 3]), {0: 0, 2: 1, 4: 2})

    def test_deleted_index_absent_from_map(self):
        self.assertNotIn(2, deletion_mapping(5, [2]))


class TestExtraReindexOps(unittest.TestCase):
    def test_maps_drift_sidecar_and_depth_per_kept_episode(self):
        mapping = {0: 0, 3: 1}  # kept ep0->0, ep3->1
        ops = extra_reindex_ops(mapping, ["central_depth"])
        self.assertIn(("extra/drift_000003.parquet", "extra/drift_000001.parquet"), ops)
        self.assertIn(
            ("extra/episode_000003.parquet", "extra/episode_000001.parquet"), ops
        )
        self.assertIn(
            (
                "extra/depth/central_depth/episode_000003",
                "extra/depth/central_depth/episode_000001",
            ),
            ops,
        )
        # ep0 maps to itself (still copied into the fresh temp dataset).
        self.assertIn(("extra/drift_000000.parquet", "extra/drift_000000.parquet"), ops)

    def test_no_depth_streams_omits_depth_ops(self):
        ops = extra_reindex_ops({0: 0}, [])
        self.assertTrue(all("depth" not in dst for _, dst in ops))
        self.assertEqual(len(ops), 3)  # drift + sidecar + identity only

    def test_batch_deleted_episodes_not_in_ops(self):
        ops = extra_reindex_ops(deletion_mapping(5, [1, 3]), ["d"])
        srcs = [s for s, _ in ops]
        self.assertFalse(any("000001" in s or "000003" in s for s in srcs))


class TestEpisodeUid(unittest.TestCase):
    def test_format_is_sortable_and_millisecond_unique(self):
        a = new_episode_uid(1_700_000_000.100)
        b = new_episode_uid(1_700_000_000.900)
        self.assertRegex(a, r"^\d{8}-\d{6}-\d{3}$")
        self.assertNotEqual(a, b)
        self.assertLess(a, b)  # lexical order == chronological order

    def test_two_calls_in_the_same_second_differ(self):
        self.assertNotEqual(
            new_episode_uid(1_700_000_000.001), new_episode_uid(1_700_000_000.002)
        )

    def test_round_trip_through_the_identity_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_episode_uid(d, 7, "20260817-120000-001", task="pick the cube")
            self.assertEqual(read_episode_uid(d, 7), "20260817-120000-001")

    def test_missing_identity_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_episode_uid(d, 3), "")

    def test_identity_survives_renumbering(self):
        # The whole point: after deleting ep0, the recording that was ep1 keeps
        # its id even though its index became 0.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_episode_uid(root, 0, "uid-zero")
            write_episode_uid(root, 1, "uid-one")
            ops = dict(extra_reindex_ops(deletion_mapping(2, [0]), []))
            src, dst = "extra/uid_000001.json", "extra/uid_000000.json"
            self.assertEqual(ops[src], dst)
            # Apply the rename the way delete_episodes_in_place does.
            (root / src).replace(root / dst)
            self.assertEqual(read_episode_uid(root, 0), "uid-one")


class _FakeMeta:
    def __init__(self, lengths):
        self.total_episodes = len(lengths)
        self.episodes = [{"length": L} for L in lengths]


class _FakeWriterMeta:
    """Stands in for LeRobotDatasetMetadata's write path (no LeRobot import).

    Mirrors only what commit_episode_metadata touches: a ``_close_writer`` that
    ends the current parquet, and ``latest_episode`` carrying the chunk/file
    indices that decide where the NEXT episode's row is written.
    """

    def __init__(self, chunk=0, file=0, chunks_size=1000, root=None):
        self.chunks_size = chunks_size
        self.root = root
        self.episodes = None
        self.latest_episode = {
            "episode_index": [0],
            "meta/episodes/chunk_index": [chunk],
            "meta/episodes/file_index": [file],
        }
        self.closes = 0

    def _close_writer(self):
        self.closes += 1

    def target(self):
        """Where the next flush would write, the way LeRobot derives it."""
        return (
            self.latest_episode["meta/episodes/chunk_index"][0],
            self.latest_episode["meta/episodes/file_index"][0],
        )


class _FakeDataWriter:
    """Stands in for DatasetWriter: the frame-data parquet side."""

    def __init__(self):
        self.closes = 0
        self._latest_episode = {"data/file_index": 0}

    def close_writer(self):
        self.closes += 1


class _FakeDataset:
    def __init__(self, meta, writer=None):
        self.meta = meta
        self.writer = writer


class TestCommitEpisodeMetadata(unittest.TestCase):
    def test_closes_the_writer_so_the_footer_is_written(self):
        meta = _FakeWriterMeta()
        self.assertTrue(commit_episode_metadata(_FakeDataset(meta)))
        self.assertEqual(meta.closes, 1)

    def test_next_episode_gets_a_fresh_file(self):
        # The regression this guards: without the bump, LeRobot reopens a writer
        # at the path just closed and truncates it, losing the earlier episodes.
        meta = _FakeWriterMeta(chunk=0, file=0)
        before = meta.target()
        commit_episode_metadata(_FakeDataset(meta))
        self.assertNotEqual(meta.target(), before)
        self.assertEqual(meta.target(), (0, 1))

    def test_every_episode_lands_in_its_own_file(self):
        meta = _FakeWriterMeta()
        seen = {meta.target()}
        for _ in range(5):
            commit_episode_metadata(_FakeDataset(meta))
            seen.add(meta.target())
        self.assertEqual(len(seen), 6)  # no path is ever reused

    def test_file_index_rolls_over_into_the_next_chunk(self):
        meta = _FakeWriterMeta(chunk=0, file=9, chunks_size=10)
        commit_episode_metadata(_FakeDataset(meta))
        self.assertEqual(meta.target(), (1, 0))

    def test_unknown_metadata_object_is_reported_not_raised(self):
        # A different LeRobot version must degrade to exit-time finalization
        # instead of failing the episode that was just recorded.
        class _Bare:
            pass

        self.assertFalse(commit_episode_metadata(_Bare()))
        self.assertFalse(commit_episode_metadata(_FakeDataset(_Bare())))

    def test_first_episode_before_any_write_is_skipped(self):
        meta = _FakeWriterMeta()
        meta.latest_episode = None
        self.assertFalse(commit_episode_metadata(_FakeDataset(meta)))
        self.assertEqual(meta.closes, 0)


class TestCommitEpisodeFrameData(unittest.TestCase):
    """The frame-data half: the file that actually holds the recorded frames."""

    def _dataset(self, d, chunk=0, file=0):
        # The committed row has to be readable back from its own parquet: it is
        # what LeRobot consults to place the next data file.
        _write_episode_meta(
            d,
            f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet",
            [0],
            [6],
        )
        meta = _FakeWriterMeta(chunk=chunk, file=file, root=Path(d))
        return _FakeDataset(meta, _FakeDataWriter()), meta

    def test_data_writer_is_closed_and_handed_back_to_lerobot(self):
        with tempfile.TemporaryDirectory() as d:
            ds, meta = self._dataset(d)
            self.assertTrue(commit_episode_metadata(ds))
            self.assertEqual(ds.writer.closes, 1)
            # None makes LeRobot take its own "start a new file" branch, which
            # derives the next file's indices and frame offsets from the row we
            # just committed rather than from anything we compute here.
            self.assertIsNone(ds.writer._latest_episode)
            self.assertEqual(meta.episodes[-1]["episode_index"], 0)

    def test_unreadable_committed_row_leaves_the_data_writer_alone(self):
        # Without the row on disk, LeRobot's new-file branch would restart at
        # file 000 and overwrite recorded frames. Better to leave the writer as
        # it is and let exit-time finalization close it.
        with tempfile.TemporaryDirectory() as d:
            meta = _FakeWriterMeta(root=Path(d))  # no parquet written
            ds = _FakeDataset(meta, _FakeDataWriter())
            self.assertTrue(commit_episode_metadata(ds))
            self.assertEqual(ds.writer.closes, 0)
            self.assertIsNotNone(ds.writer._latest_episode)
            self.assertIsNone(meta.episodes)

    def test_read_only_dataset_without_a_writer_still_commits_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            ds, meta = self._dataset(d)
            ds.writer = None
            self.assertTrue(commit_episode_metadata(ds))
            self.assertEqual(meta.closes, 1)


class TestEpisodeLengths(unittest.TestCase):
    def test_reads_lengths_in_order(self):
        self.assertEqual(episode_lengths(_FakeMeta([650, 12, 7])), [650, 12, 7])

    def test_empty(self):
        self.assertEqual(episode_lengths(_FakeMeta([])), [])


class TestReadOnlyGuard(unittest.TestCase):
    def test_read_only_filesystem_raises_before_loading(self):
        # A read-only mount cannot be edited; the guard must fail fast with the
        # typed error, before any LeRobot import or dataset load.
        with mock.patch("common.recording.dataset_edit.os.access", return_value=False):
            with self.assertRaises(ReadOnlyDatasetError):
                delete_episodes_in_place("/mnt/ro/ds", "ds", [0], [])

    def test_empty_indices_rejected_before_writability_check(self):
        with self.assertRaises(ValueError):
            delete_episodes_in_place("/mnt/ro/ds", "ds", [], [])


class TestSurvivingIndices(unittest.TestCase):
    def test_hides_without_renumbering(self):
        # Unlike deletion_mapping, survivors KEEP their on-disk indices: the
        # video route still has to address the episode where it actually lives.
        self.assertEqual(surviving_indices(5, [2]), [0, 1, 3, 4])

    def test_nothing_marked_keeps_everything(self):
        self.assertEqual(surviving_indices(3, []), [0, 1, 2])

    def test_all_marked_leaves_nothing(self):
        self.assertEqual(surviving_indices(2, [0, 1]), [])


class TestSoftDeleteMarker(unittest.TestCase):
    def test_round_trip_sorted_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(write_soft_deleted(d, [5, 1, 5, 3]), [1, 3, 5])
            self.assertEqual(read_soft_deleted(d), [1, 3, 5])

    def test_absent_marker_reads_as_nothing_marked(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_soft_deleted(d), [])

    def test_corrupt_marker_reads_as_nothing_marked(self):
        # Never refuse to open a dataset over a broken marker: the episodes are
        # all still there, so the safe reading is "nothing was deleted".
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / SOFT_DELETE_REL
            path.parent.mkdir(parents=True)
            path.write_text("{not json at all")
            self.assertEqual(read_soft_deleted(d), [])

    def test_write_leaves_no_partial_file_behind(self):
        with tempfile.TemporaryDirectory() as d:
            write_soft_deleted(d, [2])
            extra = Path(d) / "extra"
            self.assertEqual(
                sorted(p.name for p in extra.iterdir()), ["soft_deleted.json"]
            )

    def test_marker_records_the_episode_list_as_json(self):
        with tempfile.TemporaryDirectory() as d:
            write_soft_deleted(d, [4, 0])
            data = json.loads((Path(d) / SOFT_DELETE_REL).read_text())
            self.assertEqual(data["episodes"], [0, 4])
            self.assertIn("updated", data)

    def test_clear_removes_marker_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            write_soft_deleted(d, [1])
            clear_soft_deleted(d)
            self.assertEqual(read_soft_deleted(d), [])
            clear_soft_deleted(d)  # must not raise when already gone

    def test_empty_write_clears_the_marks(self):
        with tempfile.TemporaryDirectory() as d:
            write_soft_deleted(d, [1, 2])
            self.assertEqual(write_soft_deleted(d, []), [])
            self.assertEqual(read_soft_deleted(d), [])


def _write_episode_meta(root: Path, rel: str, indices, lengths) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"episode_index": indices, "length": lengths})
    pq.write_table(table, path)


class TestReadEpisodeLengths(unittest.TestCase):
    def test_reads_lengths_across_several_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write_episode_meta(
                d, "meta/episodes/chunk-000/file-000.parquet", [0, 1], [10, 20]
            )
            _write_episode_meta(
                d, "meta/episodes/chunk-000/file-001.parquet", [2], [30]
            )
            lengths, bad = read_episode_lengths(d)
            self.assertEqual(lengths, {0: 10, 1: 20, 2: 30})
            self.assertEqual(bad, [])

    def test_one_corrupt_file_costs_only_its_own_episodes(self):
        # The whole point: an interrupted write must not make the dataset
        # unlistable, which is what going through HuggingFace datasets did.
        with tempfile.TemporaryDirectory() as d:
            _write_episode_meta(
                d, "meta/episodes/chunk-000/file-000.parquet", [0], [10]
            )
            bad_path = Path(d) / "meta/episodes/chunk-000/file-003.parquet"
            bad_path.write_text("Parquet magic bytes not found in footer")
            lengths, bad = read_episode_lengths(d)
            self.assertEqual(lengths, {0: 10})
            self.assertEqual(bad, ["meta/episodes/chunk-000/file-003.parquet"])

    def test_missing_meta_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_episode_lengths(d), ({}, []))

    def test_agrees_with_the_lerobot_metadata_helper(self):
        # Same numbers as episode_lengths(meta) would give, in list form.
        with tempfile.TemporaryDirectory() as d:
            _write_episode_meta(
                d, "meta/episodes/chunk-000/file-000.parquet", [0, 1, 2], [5, 6, 7]
            )
            lengths, _ = read_episode_lengths(d)
            self.assertEqual([lengths[i] for i in range(3)], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
