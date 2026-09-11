"""Unit tests for the DualDataManager depth API.

The DepthWriter itself moved to actoris_harena and is tested there; what stays
here is how this rig's blackboard carries a depth frame.
"""

import time
import unittest

import numpy as np

from common.data_manager_dual import DualDataManager


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
