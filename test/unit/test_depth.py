"""Unit tests for the depth writer and the DualDataManager depth API.

No RealSense hardware: fake uint16 frames drive DepthWriter, and depth is
pushed through DualDataManager's pure state API.
"""

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from common.data_manager_dual import DualDataManager
from common.recording.depth import DepthWriter


class TestDepthWriter(unittest.TestCase):
    def _frame(self, val: int) -> np.ndarray:
        return np.full((8, 12), val, dtype=np.uint16)

    def test_png16_roundtrip_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(3)
                w.add(0, {"central_depth": self._frame(1000)})
                w.add(1, {"central_depth": self._frame(40000)})  # >8-bit range
                w.end_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000003"
            f0 = base / "000000.png"
            f1 = base / "000001.png"
            self.assertTrue(f0.exists() and f1.exists())
            # Read back unchanged (lossless 16-bit).
            got = cv2.imread(str(f1), cv2.IMREAD_UNCHANGED)
            self.assertEqual(got.dtype, np.uint16)
            self.assertEqual(int(got[0, 0]), 40000)

    def test_abort_removes_episode_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(5)
                w.add(0, {"central_depth": self._frame(7)})
                w.abort_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000005"
            self.assertFalse(base.exists())

    def test_begin_wipes_reused_index(self):
        # A discarded episode's index is reused by the next take: begin must
        # start from a clean directory.
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(2)
                w.add(0, {"central_depth": self._frame(11)})
                w.add(1, {"central_depth": self._frame(12)})
                w.end_episode()
                # Reuse index 2 with a single frame; the stale 000001.png goes.
                w.begin_episode(2)
                w.add(0, {"central_depth": self._frame(99)})
                w.end_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000002"
            self.assertTrue((base / "000000.png").exists())
            self.assertFalse((base / "000001.png").exists())


class TestDataManagerDepth(unittest.TestCase):
    def test_set_get_and_age(self):
        dm = DualDataManager()
        self.assertIsNone(dm.get_depth_image("central_depth"))
        self.assertIsNone(dm.get_depth_image_age("central_depth"))
        depth = np.full((4, 5), 1234, dtype=np.uint16)
        t = time.monotonic()
        dm.set_depth_image(depth, "central_depth", t_capture=t)
        got = dm.get_depth_image("central_depth")
        self.assertEqual(got.dtype, np.uint16)
        self.assertEqual(int(got[0, 0]), 1234)
        age = dm.get_depth_image_age("central_depth", t + 0.02)
        self.assertAlmostEqual(age, 0.02, places=4)

    def test_get_depth_at_reports_drift(self):
        dm = DualDataManager()
        t = time.monotonic()
        dm.set_depth_image(np.zeros((2, 2), np.uint16), "d", t_capture=t)
        res = dm.get_depth_image_at("d", t + 0.005)
        self.assertIsNotNone(res)
        _img, drift = res
        self.assertAlmostEqual(drift, 0.005, places=4)

    def test_stored_copy_is_independent(self):
        dm = DualDataManager()
        depth = np.zeros((2, 2), np.uint16)
        dm.set_depth_image(depth, "d")
        depth[0, 0] = 5  # mutate caller's array
        self.assertEqual(int(dm.get_depth_image("d")[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
