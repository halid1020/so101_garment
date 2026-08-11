"""Unit tests for the pure helpers of tool/replay_recording.py."""

import unittest
from pathlib import Path

import numpy as np

from tool.replay_recording import depth_png_path, frame_to_bgr, state12_to_side_dicts


class TestState12ToSideDicts(unittest.TestCase):
    def test_splits_body_and_gripper_per_side(self):
        # left body 0..4, left gripper 5, right body 6..10, right gripper 11
        vec = np.arange(12, dtype=float)
        d = state12_to_side_dicts(vec)
        self.assertEqual(d["left"]["shoulder_pan"], 0.0)
        self.assertEqual(d["left"]["wrist_roll"], 4.0)
        self.assertEqual(d["left"]["gripper"], 5.0)
        self.assertEqual(d["right"]["shoulder_pan"], 6.0)
        self.assertEqual(d["right"]["wrist_roll"], 10.0)
        self.assertEqual(d["right"]["gripper"], 11.0)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            state12_to_side_dicts(np.zeros(10))


class TestDepthPngPath(unittest.TestCase):
    def test_layout(self):
        p = depth_png_path(Path("/data/ds"), "central_depth", 3, 42)
        self.assertEqual(
            p,
            Path("/data/ds/extra/depth/central_depth/episode_000003/000042.png"),
        )


class TestFrameToBgr(unittest.TestCase):
    def test_float_chw_to_uint8_hwc_bgr(self):
        # CHW float in [0,1], pure red in RGB → BGR should be (0,0,255)
        chw = np.zeros((3, 4, 5), dtype=np.float32)
        chw[0] = 1.0  # R channel
        out = frame_to_bgr(chw)
        self.assertEqual(out.shape, (4, 5, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out[0, 0] == (0, 0, 255)))

    def test_uint8_hwc_passthrough_channel_swap(self):
        hwc = np.zeros((4, 5, 3), dtype=np.uint8)
        hwc[..., 2] = 255  # blue in RGB → BGR (255,0,0)
        out = frame_to_bgr(hwc)
        self.assertEqual(out.shape, (4, 5, 3))
        self.assertTrue(np.all(out[0, 0] == (255, 0, 0)))

    def test_grayscale_expands(self):
        gray = np.full((4, 5), 128, dtype=np.uint8)
        out = frame_to_bgr(gray)
        self.assertEqual(out.shape, (4, 5, 3))


if __name__ == "__main__":
    unittest.main()
