"""Unit tests for the pure curation helpers of common.recording.dataset_edit.

No LeRobot and no filesystem: exercises the deletion re-indexing arithmetic that
keeps our ``extra/`` side files aligned with LeRobot's renumbered dataset.
"""

import unittest

from common.recording.dataset_edit import (
    deletion_mapping,
    episode_lengths,
    extra_reindex_ops,
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
        self.assertEqual(len(ops), 2)  # drift + sidecar only

    def test_batch_deleted_episodes_not_in_ops(self):
        ops = extra_reindex_ops(deletion_mapping(5, [1, 3]), ["d"])
        srcs = [s for s, _ in ops]
        self.assertFalse(any("000001" in s or "000003" in s for s in srcs))


class _FakeMeta:
    def __init__(self, lengths):
        self.total_episodes = len(lengths)
        self.episodes = [{"length": L} for L in lengths]


class TestEpisodeLengths(unittest.TestCase):
    def test_reads_lengths_in_order(self):
        self.assertEqual(episode_lengths(_FakeMeta([650, 12, 7])), [650, 12, 7])

    def test_empty(self):
        self.assertEqual(episode_lengths(_FakeMeta([])), [])


if __name__ == "__main__":
    unittest.main()
