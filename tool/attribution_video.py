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
    parser.add_argument("--fps", type=int, default=4)
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

    import imageio.v2 as imageio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / f".{out.stem}_frames"
    scratch.mkdir(exist_ok=True)
    written = []
    for position, index in enumerate(indices):
        path = scratch / f"{position:05d}.png"
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
