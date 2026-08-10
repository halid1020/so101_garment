"""Unit tests for src/common/eval_video.py (no sim, no GPU)."""

import unittest

import numpy as np

from common.eval_video import CAMERA_ORDER, EvalVideoComposer


def _cameras(rng):
    return {
        name: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        for name in ("scene", "wrist_left", "wrist_right")
    }


class TestEvalVideoComposer(unittest.TestCase):
    def test_frame_shape_and_dtype(self):
        comp = EvalVideoComposer(fps=30, tile_wh=(320, 240), plot_h=368)
        rng = np.random.default_rng(0)
        overview = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        comp.add(_cameras(rng), overview, np.zeros(12), np.ones(12))
        frame = comp._frames[0]
        # width = tile_w * 4 panels; height = tile_h + plot_h
        self.assertEqual(frame.shape, (240 + 368, 320 * len(CAMERA_ORDER), 3))
        self.assertEqual(frame.dtype, np.uint8)
        comp.close()

    def test_accumulates_one_frame_per_add(self):
        comp = EvalVideoComposer(fps=30)
        rng = np.random.default_rng(1)
        overview = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        for k in range(5):
            comp.add(_cameras(rng), overview, np.full(12, k), np.full(12, k + 0.5))
        self.assertEqual(len(comp._frames), 5)
        comp.close()

    def test_missing_camera_becomes_black_tile(self):
        comp = EvalVideoComposer(fps=30, tile_wh=(80, 60), plot_h=160)
        rng = np.random.default_rng(2)
        overview = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        # omit wrist_right -> its tile must be zero-filled, not crash
        cams = _cameras(rng)
        del cams["wrist_right"]
        comp.add(cams, overview, np.zeros(12), np.zeros(12))
        self.assertEqual(len(comp._frames), 1)
        comp.close()


if __name__ == "__main__":
    unittest.main()
