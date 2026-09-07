#!/usr/bin/env python3
"""How much of a tactile frame is sensor, and how much is the light leaking in.

The crop fraction the cropped policies train at has to come from the data. This
measures the border rather than guessing at it, and prints the numbers the
choice rests on so the choice can be argued with.

    venv/bin/python tool/measure_tactile_border.py \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile \\
        --episodes 0-5

WHAT IT MEASURES, and why these two statistics. A vision-based tactile sensor's
useful signal is *deformation*: the gel moves when it touches something, so the
pixels that carry information are the ones whose value CHANGES over an episode.
The light leaking in at the boundary does the opposite -- it is bright and it is
temporally flat, because it tracks the room rather than the object. So:

* **temporal standard deviation** per pixel, over the frames sampled. High in
  the gel, low at the housing edge.
* **mean luminance** per pixel. High at the leak, mid-range in the gel.

A row or column is "border" when its temporal variation falls below a fraction
of the frame's own median -- relative, not absolute, because exposure differs
per camera and the four fingertips are not identically lit. The reported
fraction is the largest CENTRED box that excludes every such row and column,
rounded down to be safe, and it is reported per camera so a single bad sensor
is visible rather than averaged away.

Reports, never writes: the number goes into `So101ActCropConfig.tactile_crop`
by hand, because it is a decision about an experiment and deserves to be made
once and written down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so101_policies.common.tactile import TACTILE_CAMERAS  # noqa: E402


def profiles(frames: np.ndarray) -> "dict[str, np.ndarray]":
    """Per-row and per-column temporal variation and brightness.

    ``frames`` is ``(T, H, W, 3)`` in [0, 1]. Reduced to one number per row and
    per column because that is the shape of the decision -- a centred crop takes
    whole rows and columns off, so a per-pixel map would have to be reduced to
    this anyway, and reducing first makes the numbers printable.
    """
    grey = frames.mean(axis=-1)
    variation = grey.std(axis=0)
    brightness = grey.mean(axis=0)
    return {
        "row_var": variation.mean(axis=1),
        "col_var": variation.mean(axis=0),
        "row_lum": brightness.mean(axis=1),
        "col_lum": brightness.mean(axis=0),
    }


def keep_fraction(profile: np.ndarray, floor: float) -> float:
    """The largest centred fraction whose every line is above ``floor``.

    Walks in from both ends while the line is quiet, then takes the SMALLER of
    the two margins, since the crop is centred and cannot be lopsided.
    """
    n = len(profile)
    low = 0
    while low < n and profile[low] < floor:
        low += 1
    high = n - 1
    while high > low and profile[high] < floor:
        high -= 1
    if low >= high:
        return 1.0
    margin = max(low, n - 1 - high)
    return max(0.0, (n - 2 * margin) / n)


def measure(frames: np.ndarray, quiet: float) -> "dict[str, float]":
    """One camera's answer, with the evidence beside it."""
    prof = profiles(frames)
    row_floor = float(np.median(prof["row_var"])) * quiet
    col_floor = float(np.median(prof["col_var"])) * quiet
    height = keep_fraction(prof["row_var"], row_floor)
    width = keep_fraction(prof["col_var"], col_floor)
    edge = min(len(prof["row_var"]), len(prof["col_var"])) // 20 or 1
    return {
        "keep_height": height,
        "keep_width": width,
        "keep": min(height, width),
        # The evidence for the claim that the border is bright and flat. If
        # these two do not separate, the crop is not justified by this dataset
        # and the fraction should stay 1.0 rather than being invented.
        "border_variation": float(np.mean(prof["row_var"][:edge])),
        "centre_variation": float(np.median(prof["row_var"])),
        "border_luminance": float(np.mean(prof["row_lum"][:edge])),
        "centre_luminance": float(np.median(prof["row_lum"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="A LeRobotDataset root")
    parser.add_argument("--episodes", default="0-5", help="e.g. 0-5 or 0,3,7")
    parser.add_argument("--every", type=int, default=10, help="Sample 1 frame in N")
    parser.add_argument(
        "--cameras",
        default=",".join(TACTILE_CAMERAS),
        help="Comma list of camera short names to measure",
    )
    parser.add_argument(
        "--quiet",
        type=float,
        default=0.35,
        help="A line is border when its temporal variation is below this "
        "fraction of the frame's median (default 0.35)",
    )
    parser.add_argument("--json", default=None, help="Also write the numbers here")
    args = parser.parse_args()

    from common.analysis.sources import DatasetSource

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    source = DatasetSource(args.dataset, args.episodes, args.every, cameras)
    print(f"📁 {source.describe()}")

    stacks: "dict[str, list[np.ndarray]]" = {c: [] for c in cameras}
    for episode in source.episodes:
        for row in source.rows(episode):
            for camera, frame in row["images"].items():
                if camera in stacks:
                    stacks[camera].append(np.asarray(frame, dtype=np.float32))

    results = {}
    for camera in cameras:
        if not stacks[camera]:
            print(f"⚠️  {camera}: no frames")
            continue
        frames = np.stack(stacks[camera])
        if frames.shape[1] in (1, 3):  # (T, C, H, W) -> (T, H, W, C)
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.max() > 1.5:
            frames = frames / 255.0
        results[camera] = measure(frames, args.quiet)
        r = results[camera]
        print(
            f"  {camera:26s} keep {r['keep']:.2f} "
            f"(h {r['keep_height']:.2f} w {r['keep_width']:.2f})  "
            f"variation border {r['border_variation']:.4f} vs centre "
            f"{r['centre_variation']:.4f}  luminance "
            f"{r['border_luminance']:.3f} vs {r['centre_luminance']:.3f}"
        )

    if results:
        # The tightest camera decides: one fraction is trained, and a crop that
        # keeps a leak on one sensor has not solved the problem it exists for.
        chosen = min(r["keep"] for r in results.values())
        print()
        print(f"👉 tactile_crop = {chosen:.2f}   (the tightest of the cameras)")
        separated = all(
            r["border_variation"] < r["centre_variation"] for r in results.values()
        )
        if not separated:
            print(
                "⚠️  the border is NOT quieter than the centre on every camera. "
                "This dataset does not support a crop; leave tactile_crop at 1.0 "
                "rather than cropping on the strength of a figure."
            )
    if args.json and results:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"📝 {args.json}")


if __name__ == "__main__":
    main()
