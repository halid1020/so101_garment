"""Unit tests for the pure helpers of tool/replay_recording.py."""

import unittest
from pathlib import Path

import numpy as np

from common.sensor_view import colourise_depth, depth_range_from_frame
from tool.replay_recording import depth_png_path, frame_to_bgr, state12_to_side_dicts


class TestDepthRangeFromFrame(unittest.TestCase):
    def test_ignores_zeros_and_orders_near_far(self):
        # metres 0.5..1.5 at 0.001 m/unit → units 500..1500, plus zero holes
        depth = np.zeros((10, 10), dtype=np.uint16)
        depth[:, :5] = np.linspace(500, 1500, 50).reshape(10, 5).astype(np.uint16)
        rng = depth_range_from_frame(depth, 0.001)
        assert rng is not None
        near, far = rng
        self.assertLess(near, far)
        self.assertGreater(near, 0.4)
        self.assertLess(far, 1.6)

    def test_all_zero_returns_none(self):
        self.assertIsNone(depth_range_from_frame(np.zeros((4, 4), dtype=np.uint16)))

    def test_flat_frame_widens_to_min_span(self):
        depth = np.full((8, 8), 1000, dtype=np.uint16)  # all 1.0 m
        near, far = depth_range_from_frame(depth, 0.001, min_span_m=0.2)
        self.assertAlmostEqual(far - near, 0.2, places=5)


class TestColouriseDepthBounds(unittest.TestCase):
    def test_shape_dtype_and_zero_stays_black(self):
        depth = np.full((6, 7), 1000, dtype=np.uint16)
        depth[0, 0] = 0  # no-return pixel
        out = colourise_depth(depth, 0.001, near_m=0.5, far_m=1.5)
        self.assertEqual(out.shape, (6, 7, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out[0, 0] == 0))


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
