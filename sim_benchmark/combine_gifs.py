#!/usr/bin/env python3
"""Stitch per-method episode GIFs into one side-by-side comparison GIF.

Usage:
    python sim_benchmark/combine_gifs.py \
        outputs/teleop_benchmark_gifs/circle_r5cm_{pink_full,pink_relaxed,dls,mink,scipy_ls}.gif \
        -o outputs/teleop_benchmark_gifs/compare_circle_r5cm.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def load_frames(path: Path) -> list[Image.Image]:
    im = Image.open(path)
    frames = []
    for k in range(im.n_frames):
        im.seek(k)
        frames.append(im.convert("RGB").copy())
    return frames


def combine(
    paths: list[Path], out: Path, fps: float = 10.0, panel_width: int = 0
) -> None:
    clips = [load_frames(p) for p in paths]
    if panel_width:
        clips = [
            [
                f.resize(
                    (panel_width, int(f.height * panel_width / f.width)),
                    Image.LANCZOS,
                )
                for f in c
            ]
            for c in clips
        ]
    n = max(len(c) for c in clips)
    # Pad shorter clips by holding their last frame.
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    w = sum(c[0].width for c in clips)
    h = max(c[0].height for c in clips)

    frames = []
    for k in range(n):
        row = Image.new("RGB", (w, h))
        x = 0
        for c in clips:
            row.paste(c[k], (x, 0))
            x += c[k].width
        frames.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"saved {out} ({n} frames, {w}x{h})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gifs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--panel-width",
        type=int,
        default=0,
        help="Resize each source clip to this width (0 = keep original)",
    )
    args = parser.parse_args()
    combine(args.gifs, args.out, args.fps, args.panel_width)


if __name__ == "__main__":
    main()
