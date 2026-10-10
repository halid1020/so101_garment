#!/usr/bin/env python
"""Does the policy read the gel RIM? The supervisor's hypothesis, as a number.

Pooled over every analysed frame and all four tactile cameras, rather than read
off one frame -- which is how the hypothesis was formed and is not enough to
settle it.

Two things this gets right that the obvious version gets wrong:

* The rim is defined in NORMALISED coordinates, not in feature-map cells. ACT's
  Grad-CAM map is 15x20 and diffusion's is 6x8, so "the outer two cells" is 40%
  of one image and 67% of the other, and counting cells makes diffusion look
  rim-obsessed when it is only coarse. MEASURED: at three cells in, a 6x8 map
  has no interior left at all and reads 100% rim.
* Everything is reported against what a FLAT map would give at that same
  resolution. The absolute share is uninterpretable on its own -- an edge band
  covering half the image should collect half the mass from a model that has no
  preference whatever.
"""

import json

import numpy as np

TACTILE = (
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)


def rim_mask(shape, frac):
    """Cells whose centre lies within `frac` of any edge."""
    h, w = shape
    y = (np.arange(h) + 0.5) / h
    x = (np.arange(w) + 0.5) / w
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (yy < frac) | (yy > 1 - frac) | (xx < frac) | (xx > 1 - frac)


#: The 2x2 fingertip composite FastWAM reads, tiled row-major in this order
#: (``common/rig_profile.COMPOSITES``). Its map is cut back into four sensors, so
#: each sensor's own rim is measured, not the composite's outer edge.
COMPOSITE_TILES = {"tactile_quad": TACTILE}


def camera_maps(gradcam):
    """(camera, map) pairs, with a composite's map split into its tiles."""
    for camera, values in (gradcam or {}).items():
        cam_map = np.asarray(values, dtype=float)
        tiles = COMPOSITE_TILES.get(camera)
        if tiles is None:
            yield camera, cam_map
            continue
        h, w = cam_map.shape[0] // 2, cam_map.shape[1] // 2
        for index, name in enumerate(tiles):
            row, column = divmod(index, 2)
            yield name, cam_map[row * h : (row + 1) * h, column * w : (column + 1) * w]


def per_camera(path, frac):
    """{camera: (observed share, uniform share, n frames)}."""
    payload = json.load(open(path))
    seen = {}
    for episode in payload["episodes"].values():
        for frame in episode["frames"]:
            for camera, cam_map in camera_maps(frame.get("gradcam")):
                total = cam_map.sum()
                if total <= 0:
                    continue
                mask = rim_mask(cam_map.shape, frac)
                seen.setdefault(camera, []).append(
                    (cam_map[mask].sum() / total, mask.mean())
                )
    return {
        camera: (
            float(np.mean([o for o, _ in rows])),
            float(np.mean([u for _, u in rows])),
            len(rows),
        )
        for camera, rows in seen.items()
    }


def pooled(path, frac, cameras=TACTILE):
    rows = per_camera(path, frac)
    picked = [rows[c] for c in cameras if c in rows]
    if not picked:
        return float("nan"), float("nan"), 0
    return (
        float(np.mean([o for o, _, _ in picked])),
        float(np.mean([u for _, u, _ in picked])),
        sum(n for _, _, n in picked),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attributions", nargs="+", help="attribution.json files")
    parser.add_argument("--band", type=float, default=0.15, help="Edge band, 0-0.5")
    parser.add_argument("--all-cameras", action="store_true")
    args = parser.parse_args()

    cameras = None if args.all_cameras else TACTILE
    print(f"{'file':<34} {'rim share':>10} {'flat map':>10} {'ratio':>8} {'n':>7}")
    for path in args.attributions:
        rows = per_camera(path, args.band)
        picked = [rows[c] for c in (cameras or rows) if c in rows]
        if not picked:
            print(f"{path[-34:]:<34} {'no gradcam maps':>40}")
            continue
        observed = float(np.mean([o for o, _, _ in picked]))
        flat = float(np.mean([u for _, u, _ in picked]))
        n = sum(k for _, _, k in picked)
        print(
            f"{path[-34:]:<34} {observed:>9.1%} {flat:>10.1%} "
            f"{observed / flat:>8.2f} {n:>7}"
        )
    print("\nRatio > 1 means the map seeks the rim; < 1 means it seeks the centre.")


if __name__ == "__main__":
    main()
