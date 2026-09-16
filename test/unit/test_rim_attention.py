#!/usr/bin/env python
"""The rim measure, on maps whose answer is known by construction.

The measure exists to settle a hypothesis formed by looking at single frames,
so it has to be right in the two ways looking at frames is not: it must be
resolution-independent, and it must be reported against what a map with NO
preference would give.

Both of those were got wrong first. Counting the rim in feature-map CELLS made
diffusion look rim-obsessed at 1.8x uniform when it is only coarse -- its map is
6x8 where ACT's is 15x20, so "two cells in" is 40% of one image and 67% of the
other, and three cells in leaves a 6x8 map with no interior at all and a rim
share of exactly 100%.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tool"))

from rim_attention import per_camera, rim_mask  # noqa: E402


class RimMaskTest(unittest.TestCase):
    def test_a_flat_map_gives_back_the_band_itself(self):
        """The whole measure is a ratio against this, so it has to be exact."""
        for shape in ((15, 20), (6, 8), (7, 7)):
            for band in (0.1, 0.15, 0.2):
                with self.subTest(shape=shape, band=band):
                    mask = rim_mask(shape, band)
                    flat = np.ones(shape) / np.prod(shape)
                    self.assertAlmostEqual(flat[mask].sum(), mask.mean(), places=12)

    def test_a_band_the_coarse_grid_can_resolve_agrees_across_resolutions(self):
        """At 20% the 6x8 and 15x20 maps mean nearly the same region.

        A 6x8 grid's cells are 1/6 of the height, so it can only place a
        boundary in sixths. At a 20% band both grids take one row and one column
        from each side and the realised regions agree; that is the band the ACT
        and diffusion numbers may be compared at.
        """
        coarse = rim_mask((6, 8), 0.2).mean()
        fine = rim_mask((15, 20), 0.2).mean()
        self.assertLess(abs(coarse - fine), 1 / 6)

    def test_a_narrow_band_is_NOT_comparable_across_resolutions(self):
        """And at 10% they are not, which is a limit to report, not to hide.

        The 6x8 grid's first cell centre sits at 8.3%, inside a 10% band, so it
        takes a whole sixth of the image where the 15x20 grid takes a fifteenth.
        The ratio-against-flat normalisation keeps each side honest about its
        OWN map, but the two are then answering about different regions -- so a
        cross-family claim belongs at 20%, not at 10%.
        """
        coarse = rim_mask((6, 8), 0.1).mean()
        fine = rim_mask((15, 20), 0.1).mean()
        self.assertGreater(coarse - fine, 1 / 6)

    def test_counting_cells_would_not_be_resolution_independent(self):
        """The mistake, pinned so it cannot come back.

        Three cells in from a 6x8 map leaves nothing, so every map reads 100%
        rim however centred it is. This asserts the failure of the REJECTED
        approach, which is why it is written as an assertion about geometry
        rather than about the code.
        """
        self.assertEqual((6 - 2 * 3) * (8 - 2 * 3), 0)
        self.assertGreater((15 - 2 * 3) * (20 - 2 * 3), 0)


class PerCameraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self._make())

    def _make(self):
        import json
        import tempfile

        centre = np.zeros((15, 20))
        centre[7, 10] = 1.0
        rim = np.zeros((15, 20))
        rim[0, 0] = 1.0
        payload = {
            "episodes": {
                "0": {
                    "frames": [
                        {
                            "index": 0,
                            "gradcam": {
                                "centre_cam": centre.tolist(),
                                "rim_cam": rim.tolist(),
                            },
                        },
                    ]
                }
            }
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_a_map_with_all_its_mass_at_the_centre_reads_zero(self):
        rows = per_camera(str(self.tmp), 0.15)
        observed, _, count = rows["centre_cam"]
        self.assertEqual(observed, 0.0)
        self.assertEqual(count, 1)

    def test_a_map_with_all_its_mass_in_a_corner_reads_one(self):
        observed, _, _ = per_camera(str(self.tmp), 0.15)["rim_cam"]
        self.assertEqual(observed, 1.0)

    def test_an_empty_map_is_skipped_not_counted_as_centred(self):
        """A frame where Grad-CAM collected nothing must not pull the mean
        toward zero and read as a strongly centre-seeking policy."""
        import json
        import tempfile

        payload = {
            "episodes": {
                "0": {
                    "frames": [
                        {"index": 0, "gradcam": {"cam": np.zeros((15, 20)).tolist()}},
                    ]
                }
            }
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        try:
            self.assertEqual(per_camera(handle.name, 0.15), {})
        finally:
            Path(handle.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
