"""Unit tests for the pure curation helpers of common.recording.dataset_edit.

No LeRobot and no filesystem: exercises the deletion re-indexing arithmetic that
keeps our ``extra/`` side files aligned with LeRobot's renumbered dataset.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.recording.dataset_edit import (
    ReadOnlyDatasetError,
    delete_episodes_in_place,
    deletion_mapping,
    episode_lengths,
    extra_reindex_ops,
    new_episode_uid,
    read_episode_uid,
    write_episode_uid,
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


if __name__ == "__main__":
    unittest.main()
