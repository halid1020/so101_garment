#!/usr/bin/env python
"""Framewise attribution, several policies at once, as a video.

The still figures in the report pool over frames, and pooling hides the thing
worth seeing: what a policy leans on CHANGES within an episode, and the tactile
channels matter around contact and almost nowhere else. A mean over an episode
reports a small tactile share and says nothing about when it was large.

Every policy is drawn in the SAME frame from the SAME recording, because that is
the only arrangement in which the comparison is honest -- separate videos played
side by side drift apart, and a reader cannot tell whether two panels differ
because the policies differ or because the moments do.

Two rules this enforces rather than trusts:

* **A cropped policy's panel is drawn from the CROPPED image.** It is the image
  that policy receives; showing it the full frame would illustrate attention to
  pixels it never saw.
* **A policy with no spatial feature map is refused from a Grad-CAM video**, by
  name, rather than drawn as an empty panel. One of the three families here has
  none, and an empty panel in a row of full ones reads as a policy that attends
  to nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
#: Proprioception, then the overhead camera, then the fingertips. Fixed, so a
#: bar does not change meaning between frames or between panels.
STREAM_ORDER = (
    "state",
    "central",
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)
STREAM_COLOUR = {
    "state": "#5b4b9e",
    "central": "#2a78d6",
}
TACTILE_COLOUR = "#eb6834"


class NoSuchMethod(RuntimeError):
    """This attribution run does not carry the method the video needs.

    Deliberately its own type. The caller has to tell "the method does not apply
    to this architecture" apart from "the run did not include it", and both
    arrive here as a missing key.
    """


def load_arm(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def frames_of(run: dict, episode: int) -> "list[dict]":
    """The per-frame records for one episode, or nothing if it was not covered."""
    episodes = run.get("episodes") or {}
    return (episodes.get(str(episode)) or {}).get("frames") or []


def shares_at(frame: dict) -> "dict[str, float]":
    """The occlusion share of each stream on one frame."""
    streams = ((frame.get("occlusion") or {}).get("streams")) or {}
    return {name: float(value.get("share", 0.0)) for name, value in streams.items()}


def maps_at(run: dict, episode: int, index: int) -> "dict | None":
    """The gradient maps for one arm at one instant, or nothing."""
    frame = next((f for f in frames_of(run, episode) if f["index"] == index), None)
    return (frame or {}).get("gradcam")


def require_gradcam(name: str, run: dict, episode: int) -> None:
    """Refuse, by name, a policy whose run carries no gradient maps."""
    for frame in frames_of(run, episode):
        if frame.get("gradcam"):
            return
    raise NoSuchMethod(
        f"{name} has no Grad-CAM maps for episode {episode}. Either the method "
        "does not apply to this architecture -- a token model has no spatial "
        "feature map to attribute to -- or the run did not request it. It is "
        "left OUT of the video rather than drawn empty, because an empty panel "
        "beside full ones reads as a policy attending to nothing."
    )


def common_frames(runs: "dict[str, dict]", episode: int) -> "list[int]":
    """Frame indices every arm scored, so each video frame compares like with like.

    An arm that stopped early must not silently shorten the others, nor be
    carried forward on a stale panel: the video covers the intersection and the
    caller is told what it lost.
    """
    sets = [set(f["index"] for f in frames_of(run, episode)) for run in runs.values()]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def draw_shares(axis, shares: "dict[str, float]", label: str) -> None:
    """One policy's stream shares at one instant, as fixed-order bars.

    The axis is pinned to the unit interval and the order never changes, so the
    height of a bar means the same thing in every frame of the video and in
    every panel of a frame. An autoscaled axis would make every policy look
    equally decisive and every moment look equally eventful.
    """
    names = [n for n in STREAM_ORDER if n in shares]
    values = [shares[n] for n in names]
    colours = [STREAM_COLOUR.get(n, TACTILE_COLOUR) for n in names]
    positions = np.arange(len(names))
    axis.barh(positions, values, color=colours, height=0.68)
    axis.set_yticks(positions)
    axis.set_yticklabels(
        ["proprioception" if n == "state" else n.replace("_", " ") for n in names],
        fontsize=7,
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("share of what moved the plan", fontsize=7, color=MUTED)
    axis.set_title(label, fontsize=9, color=INK)
    axis.grid(axis="x", alpha=0.3, color=GRID)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=7)


def draw_map(axis, image: np.ndarray, cam: "np.ndarray | None", label: str) -> None:
    """A camera image with its gradient map over it, as deviation from uniform.

    The map is shown relative to its own mean rather than raw, because a raw map
    is dominated by its overall magnitude, which differs between policies and
    carries no comparative meaning. What a reader should see is where this
    policy looks MORE than it looks everywhere else.
    """
    axis.imshow(image)
    if cam is not None:
        grid = np.asarray(cam, dtype=float)
        if grid.mean() != 0:
            grid = grid / grid.mean()
        axis.imshow(
            grid,
            cmap="inferno",
            alpha=0.45,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
    axis.set_title(label, fontsize=8, color=INK)
    axis.axis("off")


def crop_for(run: dict) -> "tuple[tuple, tuple] | None":
    """The tactile crop this arm was trained with, read from its checkpoint.

    Returned rather than guessed, and read from the checkpoint the attribution
    run actually scored, because a crop drawn from anywhere else would be a
    picture of a different policy. An arm whose config names no crop returns
    nothing, which is the uncropped case.
    """
    checkpoint = run.get("checkpoint")
    if not checkpoint:
        return None
    config = Path(checkpoint) / "config.json"
    if not config.exists():
        return None
    body = json.loads(config.read_text())
    fraction = body.get("tactile_crop")
    if not fraction:
        return None
    cameras = tuple(body.get("tactile_cameras") or ())
    return tuple(fraction), cameras


def apply_crop(image: np.ndarray, crop) -> np.ndarray:
    """What the policy was shown: centre-cropped and resized back.

    Delegates to the policy package's own crop rather than repeating the
    arithmetic here. A second implementation that drifted by a pixel would
    illustrate attention to pixels the policy never saw, which is the one thing
    this video exists to avoid.
    """
    import torch
    from actoris_harena.policies.common.tactile import crop_and_restore

    chw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    restored = crop_and_restore(chw, crop)
    return restored.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()


def images_for(root: str, indices: "list[int]", cameras: "list[str]") -> dict:
    """Every camera image the video needs, keyed by frame index then camera.

    The gradient maps are stored without the frames they were computed from, so
    the recording has to be read again. Read once here and reused across arms:
    the arms differ in what they attend to, not in what they were shown.
    """
    from actoris_harena.analysis.sources import DatasetSource, _to_hwc_uint8

    source = DatasetSource(root=root, every=1)
    keys = [k for k in source.keys if k.split(".")[-1] in cameras]
    out: dict = {}
    for index in indices:
        sample = source.dataset[index]
        out[index] = {k.split(".")[-1]: _to_hwc_uint8(sample[k]) for k in keys}
    return out


def render_gradcam_frame(runs, episode, index, out_path, images, crops) -> Path:
    """One video frame: every arm down the page, every camera across it.

    Laid out arm-per-row so a reader compares a policy against its cropped twin
    by looking down a column, which is the comparison the report is about.
    """
    cameras = sorted(
        {c for run in runs.values() for c in (maps_at(run, episode, index) or {})},
        key=lambda c: (c != "central", c),
    )
    rows, columns = len(runs), max(1, len(cameras))
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 2.1, rows * 1.85), squeeze=False
    )
    for row, (name, run) in enumerate(runs.items()):
        grids = maps_at(run, episode, index) or {}
        crop = crops.get(name)
        for column, camera in enumerate(cameras):
            axis = axes[row][column]
            picture = (images.get(index) or {}).get(camera)
            if picture is None:
                axis.axis("off")
                continue
            if crop and camera in crop[1]:
                picture = apply_crop(picture, crop[0])
            grid = grids.get(camera)
            draw_map(
                axis,
                picture,
                np.asarray(grid, dtype=float) if grid is not None else None,
                camera.replace("_", " ") if row == 0 else "",
            )
            if column == 0:
                axis.text(
                    -0.06,
                    0.5,
                    name,
                    transform=axis.transAxes,
                    rotation=90,
                    fontsize=8,
                    color=INK,
                    ha="right",
                    va="center",
                )
    figure.suptitle(f"episode {episode}, frame {index}", fontsize=10, color=MUTED)
    figure.tight_layout(rect=(0.02, 0, 1, 0.95))
    figure.savefig(out_path, dpi=110)
    plt.close(figure)
    return out_path


def tile_shape(count: int) -> "tuple[int, int]":
    """Rows and columns for `count` panels, wide rather than tall.

    A video is watched on a screen that is wider than it is high, so panels go
    across before they go down.
    """
    if count <= 0:
        raise ValueError("a video with no panels is not a video")
    columns = min(count, 3)
    rows = int(np.ceil(count / columns))
    return rows, columns


def render_shares_frame(runs, episode, index, out_path) -> Path:
    """One video frame: every arm's shares at one instant, side by side."""
    rows, columns = tile_shape(len(runs))
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 3.6, rows * 2.3), squeeze=False
    )
    flat = [a for row in axes for a in row]
    for axis in flat:
        axis.axis("off")
    for axis, (name, run) in zip(flat, runs.items()):
        axis.axis("on")
        frame = next((f for f in frames_of(run, episode) if f["index"] == index), None)
        draw_shares(axis, shares_at(frame) if frame else {}, name)
    figure.suptitle(f"episode {episode}, frame {index}", fontsize=10, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_path, dpi=110)
    plt.close(figure)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=attribution.json",
        help="repeatable; one panel per arm, drawn in the order given",
    )
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--mode", choices=("shares", "gradcam"), default="shares")
    parser.add_argument(
        "--dataset",
        default="",
        help="dataset root; required by --mode gradcam, which draws the maps "
        "over the frames they came from",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--still",
        type=int,
        default=None,
        help="render ONE frame index to --out as a picture instead of a video, "
        "so the report shows the same panels at full quality rather than a "
        "frame recovered from a compressed stream",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs: "dict[str, dict]" = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--arm wants NAME=path, got {entry!r}")
        runs[name] = load_arm(Path(path))
    if not runs:
        raise SystemExit("no --arm given")

    if args.mode == "gradcam":
        for name, run in list(runs.items()):
            try:
                require_gradcam(name, run, args.episode)
            except NoSuchMethod as problem:
                print(f"  dropping {name}: {problem}")
                del runs[name]
        if not runs:
            raise SystemExit("no arm has Grad-CAM maps for this episode")

    indices = common_frames(runs, args.episode)
    if not indices:
        raise SystemExit(
            f"no frame of episode {args.episode} was scored by every arm; "
            "the video would compare different moments"
        )
    for name, run in runs.items():
        covered = len(frames_of(run, args.episode))
        if covered != len(indices):
            print(f"  note: {name} scored {covered} frames, video uses {len(indices)}")

    images: dict = {}
    crops: dict = {}
    if args.mode == "gradcam":
        if not args.dataset:
            raise SystemExit(
                "--dataset is required for a Grad-CAM video: the maps are stored "
                "without the frames they were computed from, so the recording has "
                "to be read again to draw them over anything"
            )
        crops = {n: crop_for(r) for n, r in runs.items()}
        cameras = sorted(
            {
                c
                for r in runs.values()
                for c in (maps_at(r, args.episode, indices[0]) or {})
            }
        )
        # A still needs one frame decoded, not sixty.
        wanted = [args.still] if args.still is not None else indices
        images = images_for(args.dataset, wanted, cameras)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.still is not None:
        if args.still not in indices:
            nearest = min(indices, key=lambda i: abs(i - args.still))
            raise SystemExit(
                f"frame {args.still} was not scored by every arm; the nearest "
                f"that was is {nearest}"
            )
        if args.mode == "gradcam":
            render_gradcam_frame(runs, args.episode, args.still, out, images, crops)
        else:
            render_shares_frame(runs, args.episode, args.still, out)
        print(f"\n✅ {out}  (frame {args.still})")
        return 0

    import imageio.v2 as imageio

    scratch = out.parent / f".{out.stem}_frames"
    scratch.mkdir(exist_ok=True)
    written = []
    for position, index in enumerate(indices):
        path = scratch / f"{position:05d}.png"
        if args.mode == "gradcam":
            render_gradcam_frame(runs, args.episode, index, path, images, crops)
        else:
            render_shares_frame(runs, args.episode, index, path)
        written.append(path)
        print(f"\r  {position + 1}/{len(indices)} frames", end="", flush=True)
    print()

    with imageio.get_writer(out, fps=args.fps) as writer:
        for path in written:
            writer.append_data(imageio.imread(path))
    for path in written:
        path.unlink()
    scratch.rmdir()
    print(f"✅ {out}  ({len(written)} frames, {len(runs)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
