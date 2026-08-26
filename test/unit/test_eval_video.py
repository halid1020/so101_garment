"""Unit tests for src/common/eval_video.py (no sim, no GPU)."""

import unittest

import numpy as np

from common.eval_video import CAMERA_ORDER, EvalVideoComposer, gif_frames


def _cameras(rng):
    return {
        name: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        for name in ("scene", "wrist_camera_left", "wrist_camera_right")
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
        # omit wrist_camera_right -> its tile must be zero-filled, not crash
        cams = _cameras(rng)
        del cams["wrist_camera_right"]
        comp.add(cams, overview, np.zeros(12), np.zeros(12))
        self.assertEqual(len(comp._frames), 1)
        comp.close()


if __name__ == "__main__":
    unittest.main()


class TestGifFrames(unittest.TestCase):
    """The GIF is the skimmable artefact: camera row only, thinned, smaller."""

    @staticmethod
    def frames(n, w=1280, tile_h=240, plot_h=368):
        rng = np.random.default_rng(1)
        return [
            rng.integers(0, 255, (tile_h + plot_h, w, 3), dtype=np.uint8)
            for _ in range(n)
        ]

    def test_the_signal_panel_is_dropped(self):
        out = gif_frames(self.frames(1), tile_h=240, stride=1, width=1280)
        self.assertEqual(out[0].shape[0], 240)

    def test_every_stride_th_frame_survives(self):
        out = gif_frames(self.frames(10), tile_h=240, stride=3)
        self.assertEqual(len(out), 4)  # 0, 3, 6, 9

    def test_the_aspect_ratio_is_preserved_when_downscaling(self):
        out = gif_frames(self.frames(1), tile_h=240, stride=1, width=640)
        self.assertEqual(out[0].shape[:2], (120, 640))

    def test_a_frame_already_small_enough_is_not_upscaled(self):
        out = gif_frames(self.frames(1, w=320), tile_h=240, stride=1, width=640)
        self.assertEqual(out[0].shape[:2], (240, 320))

    def test_a_stride_below_one_is_treated_as_one(self):
        out = gif_frames(self.frames(4), tile_h=240, stride=0)
        self.assertEqual(len(out), 4)

    def test_no_frames_means_no_gif_rather_than_a_crash(self):
        self.assertEqual(gif_frames([], tile_h=240), [])
