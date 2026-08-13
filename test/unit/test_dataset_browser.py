"""Unit tests for the pure helpers of tool/dataset_browser.py.

No cv2 window and no hardware: the GUI calls all live inside ``main`` so the
module imports cleanly in CI, and these tests exercise only the deletion
re-indexing and the click hit-test.
"""

import unittest

from tool.dataset_browser import (
    deletion_mapping,
    episode_at_y,
    extra_reindex_ops,
    list_scroll_offset,
)


class TestDeletionMapping(unittest.TestCase):
    def test_middle_delete_renumbers_tail(self):
        # delete ep2 of 0..4 -> keep 0,1,3,4 renumbered 0,1,2,3
        self.assertEqual(deletion_mapping(5, [2]), {0: 0, 1: 1, 3: 2, 4: 3})

    def test_first_delete_shifts_all_down(self):
        self.assertEqual(deletion_mapping(3, [0]), {1: 0, 2: 1})

    def test_last_delete_leaves_prefix_unchanged(self):
        self.assertEqual(deletion_mapping(3, [2]), {0: 0, 1: 1})

    def test_multi_delete(self):
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

    def test_deleted_episode_not_in_ops(self):
        ops = extra_reindex_ops(deletion_mapping(3, [1]), ["d"])
        srcs = [s for s, _ in ops]
        self.assertFalse(any("000001" in s for s in srcs))  # ep1 was deleted


class TestEpisodeAtY(unittest.TestCase):
    def test_first_row(self):
        self.assertEqual(episode_at_y(y=44, n_visible=5, row_h=34, y0=44), 0)

    def test_second_row(self):
        self.assertEqual(episode_at_y(y=44 + 34, n_visible=5, row_h=34, y0=44), 1)

    def test_above_list_is_none(self):
        self.assertIsNone(episode_at_y(y=10, n_visible=5, row_h=34, y0=44))

    def test_below_last_row_is_none(self):
        self.assertIsNone(episode_at_y(y=44 + 34 * 5, n_visible=5, row_h=34, y0=44))


class TestListScrollOffset(unittest.TestCase):
    def test_no_scroll_when_all_fit(self):
        self.assertEqual(list_scroll_offset(selected=3, n=5, max_rows=10), 0)

    def test_centres_selection_when_overflowing(self):
        # 20 episodes, 10 visible, selected 12 -> first = 12 - 5 = 7
        self.assertEqual(list_scroll_offset(selected=12, n=20, max_rows=10), 7)

    def test_clamps_to_last_window(self):
        # near the end it stops so the window stays full
        self.assertEqual(list_scroll_offset(selected=19, n=20, max_rows=10), 10)

    def test_clamps_to_zero_near_start(self):
        self.assertEqual(list_scroll_offset(selected=1, n=20, max_rows=10), 0)


if __name__ == "__main__":
    unittest.main()
