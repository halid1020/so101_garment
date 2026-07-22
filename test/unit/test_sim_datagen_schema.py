"""Unit tests for the sim-VLA seed protocol and observation schema.

Pure python / numpy (no MuJoCo): the disjoint train/val/eval seed pools and
the 12-D observation-state layout the collector and eval harness share.
"""

import unittest

import numpy as np

from common.recording.features import build_observation_state
from sim_datagen.seeds import EVAL_SEEDS, TRAIN_SEEDS, VAL_SEEDS


class TestSeedPools(unittest.TestCase):
    def test_pools_disjoint(self):
        train, val, ev = set(TRAIN_SEEDS), set(VAL_SEEDS), set(EVAL_SEEDS)
        self.assertEqual(train & val, set())
        self.assertEqual(train & ev, set())
        self.assertEqual(val & ev, set())

    def test_pool_sizes(self):
        # Long-run protocol: ~1000 train, 5-10 val, 30 eval.
        self.assertEqual(len(list(TRAIN_SEEDS)), 1000)
        self.assertGreaterEqual(len(list(VAL_SEEDS)), 5)
        self.assertLessEqual(len(list(VAL_SEEDS)), 10)
        self.assertEqual(len(list(EVAL_SEEDS)), 30)


class TestObservationSchema(unittest.TestCase):
    def test_state_layout(self):
        joints = np.arange(10, dtype=np.float64)  # left 0-4, right 5-9
        grip = {"left": 0.3, "right": 0.7}
        state = build_observation_state(joints, grip)
        self.assertEqual(state.shape, (12,))
        self.assertEqual(state.dtype, np.float32)
        # Layout: [left5, left_grip, right5, right_grip].
        np.testing.assert_allclose(state[0:5], joints[0:5])
        self.assertAlmostEqual(state[5], 0.3, places=6)
        np.testing.assert_allclose(state[6:11], joints[5:10])
        self.assertAlmostEqual(state[11], 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
