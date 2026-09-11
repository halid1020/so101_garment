"""Unit tests for the collector's choice of where a dataset goes.

The settings module itself moved to actoris_harena and is tested there; what
is left here is tool/collect_dataset.py, which is this rig's entry point.
"""

import tempfile
import unittest
from pathlib import Path

from tool.collect_dataset import nearest_existing_ancestor


class TestNearestExistingAncestor(unittest.TestCase):
    def test_returns_path_itself_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(nearest_existing_ancestor(Path(tmp)), Path(tmp))

    def test_walks_up_to_existing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no" / "such" / "child"
            self.assertEqual(nearest_existing_ancestor(missing), Path(tmp))

    def test_backslash_escaped_path_resolves_above_mount(self):
        # A quoted path that kept its backslashes (a common shell mistake)
        # has no real component under the mount, so the nearest existing
        # ancestor is the mount root, not the intended directory.
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "Seagate\\ Portable\\ Drive" / "so101"
            self.assertEqual(nearest_existing_ancestor(bad), Path(tmp))


if __name__ == "__main__":
    unittest.main()
