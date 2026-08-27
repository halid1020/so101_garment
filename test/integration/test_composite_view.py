"""Tiling four cameras into one image feature, on real encoded video.

The pure parts of a composite -- its grid, its feature entry, its derived
statistics -- are unit-tested. What cannot be tested there is the thing that
actually goes wrong: decoding four videos in lockstep and writing one. A tiling
that paired camera A's frame k with camera B's frame k+1, or that put the wrong
camera in a quadrant, would produce a dataset that loads, trains and teaches the
wrong thing, and nothing downstream would say so.

So this builds a real composite from real encoded frames and asks the only
question that settles it: does each quadrant look like ITS OWN camera? AV1 is
lossy and the tiles are scaled down, so the test is not equality -- it is that a
quadrant matches its own source far better than any other camera's, which no
misalignment or mis-ordering could produce.

Run:  PYTHONPATH=.:src python -m unittest test.integration.test_composite_view
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.recording.dataset_view import (
    CAMERA_PREFIX,
    ViewError,
    build_composite_video,
    composite_grid,
)

FRAMES = 12
SIZE = (64, 64)  # small and square, like the shipped one but quick to encode
SOURCE_SIZE = (48, 64)  # 4:3, like the rig's cameras


def write_video(path: Path, colours, size=SOURCE_SIZE) -> None:
    """One frame per colour, each a flat field so a quadrant is identifiable."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libsvtav1", rate=30)
    stream.height, stream.width = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "10"}
    for colour in colours:
        frame = np.zeros((*size, 3), dtype=np.uint8)
        frame[:, :] = colour
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, "rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def read_frames(path: Path, limit=None) -> "list[np.ndarray]":
    import av

    container = av.open(str(path))
    try:
        out = []
        for i, frame in enumerate(container.decode(video=0)):
            if limit is not None and i >= limit:
                break
            out.append(frame.to_ndarray(format="rgb24"))
        return out
    finally:
        container.close()


class TestCompositeVideo(unittest.TestCase):
    """Four sources, each a different colour that changes over time."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        # Each camera owns a channel pattern, and each frame a brightness, so a
        # quadrant identifies both which camera it came from and which frame.
        cls.palettes = [
            [(20 + 15 * t, 0, 0) for t in range(FRAMES)],
            [(0, 20 + 15 * t, 0) for t in range(FRAMES)],
            [(0, 0, 20 + 15 * t) for t in range(FRAMES)],
            [(20 + 15 * t, 20 + 15 * t, 0) for t in range(FRAMES)],
        ]
        cls.sources = []
        for i, palette in enumerate(cls.palettes):
            path = cls.tmp / f"cam{i}.mp4"
            write_video(path, palette)
            cls.sources.append(path)
        cls.out = cls.tmp / "quad.mp4"
        cls.written = build_composite_video(cls.sources, cls.out, SIZE)
        cls.frames = read_frames(cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def quadrant(self, frame, index):
        rows, cols = composite_grid(4)
        h, w = SIZE[0] // rows, SIZE[1] // cols
        r, c = divmod(index, cols)
        return frame[r * h : (r + 1) * h, c * w : (c + 1) * w]

    def test_every_frame_is_written(self):
        self.assertEqual(self.written, FRAMES)
        self.assertEqual(len(self.frames), FRAMES)

    def test_the_output_is_the_size_that_was_asked_for(self):
        self.assertEqual(self.frames[0].shape, (*SIZE, 3))

    def test_each_quadrant_holds_its_own_camera(self):
        # The failure this catches: a tiling in the wrong order, which trains a
        # policy on a picture of the rig that is not the rig.
        for index, palette in enumerate(self.palettes):
            patch = self.quadrant(self.frames[FRAMES // 2], index).astype(float)
            want = np.array(palette[FRAMES // 2], dtype=float)
            errors = [
                float(np.abs(patch - np.array(p[FRAMES // 2], dtype=float)).mean())
                for p in self.palettes
            ]
            self.assertEqual(
                int(np.argmin(errors)),
                index,
                f"quadrant {index} looks more like camera {int(np.argmin(errors))}",
            )
            self.assertLess(np.abs(patch.mean(axis=(0, 1)) - want).max(), 24)

    def test_a_quadrant_holds_its_own_frame_not_a_neighbour(self):
        # The failure this catches: sources decoded out of lockstep, which
        # pairs one camera's frame k with another's k+1 -- a dataset that loads
        # and trains and is quietly wrong about what happened when.
        index = 0
        palette = self.palettes[index]
        for k in (2, FRAMES // 2, FRAMES - 2):
            patch = self.quadrant(self.frames[k], index).astype(float)
            errors = {
                offset: float(
                    np.abs(
                        patch.mean(axis=(0, 1))
                        - np.array(palette[k + offset], dtype=float)
                    ).mean()
                )
                for offset in (-1, 0, 1)
            }
            self.assertEqual(min(errors, key=lambda o: errors[o]), 0, f"frame {k}")

    def test_sources_of_different_lengths_are_refused(self):
        # They cannot happen -- the recorder writes one frame per camera per
        # dataset frame -- but if they ever did, tiling them would silently
        # misalign every later frame rather than fail.
        short = self.tmp / "short.mp4"
        write_video(short, self.palettes[0][: FRAMES - 3])
        with self.assertRaises(ViewError) as caught:
            build_composite_video(
                [self.sources[0], self.sources[1], self.sources[2], short],
                self.tmp / "bad.mp4",
                SIZE,
            )
        self.assertIn("same number of frames", str(caught.exception))


class TestCompositeViewIsAnOrdinaryDataset(unittest.TestCase):
    """The metadata a composite view writes, against a real source dataset.

    Skipped unless one of this rig's tactile datasets is on the machine: the
    point is that LeRobot opens the result without being told anything about
    tiling, and only a real dataset can show that.
    """

    ROOTS = (
        Path("/mnt/seagate/so101/fold-short-tactile"),
        Path("/mnt/seagate/so101/fold-short-from-flattend-tactile"),
    )

    @classmethod
    def setUpClass(cls):
        cls.src = next((r for r in cls.ROOTS if (r / "meta/info.json").is_file()), None)
        if cls.src is None:
            raise unittest.SkipTest("no tactile dataset on this machine")

    def test_the_view_names_the_composite_and_gives_it_a_shape(self):
        from common.recording.dataset_view import filtered_info, split_selection

        info = json.loads((self.src / "meta" / "info.json").read_text())
        keep, composites = split_selection(info, ["central", "tactile_quad"])
        out = filtered_info(info, keep, composites)
        feature = out["features"][CAMERA_PREFIX + "tactile_quad"]
        self.assertEqual(feature["shape"][:2], [224, 224])
        self.assertNotIn(
            CAMERA_PREFIX + "left_arm_left_gripper",
            out["features"],
            "a composite's parts are consumed by it, not kept beside it",
        )


if __name__ == "__main__":
    unittest.main()
