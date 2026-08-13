"""Replay a recorded episode in the exact live-collection view.

Given a dataset and an episode id, this reopens the SAME window the operator
watched while collecting — the camera tiles (colour plus the colourised central
depth) over the two per-arm joint columns — but stepping through the recorded
frames instead of live sensors. Column one shows the recorded measured state,
column two the recorded command, so a review looks identical to collection with
no live drift line. The composited view can also be written to an mp4.

The layout is produced by ``common.sensor_view.compose_sensor_view_frame``, the
same function the live monitor uses, so the replay and the live view cannot
drift apart.

Usage:

    # Window playback:
    venv/bin/python tool/replay_recording.py \\
        --dir /media/hdd/so101 --name towel_fold --episode 0

    # Save to mp4 (add --no-view to skip the window, e.g. headless):
    venv/bin/python tool/replay_recording.py --dir /media/hdd/so101 \\
        --name towel_fold --episode 0 --mp4 episode0.mp4

    --dataset-root DIR --repo-id ID   # instead of --dir/--name
    --fps N       # playback/output rate override (default: the dataset's)
    --loop        # repeat until q/Esc (window only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np

from common.sensor_view import (
    ViewPanel,
    colourise_depth,
    compose_sensor_view_frame,
    depth_range_from_frame,
)

_IMAGE_PREFIX = "observation.images."


def state12_to_side_dicts(vec12) -> dict:
    """Split a 12-channel state/action vector into per-side joint dicts.

    Layout (``common.recording.features.STATE_NAMES``): left five body joints
    then the left gripper, then the right arm. Returns ``{"left": {...},
    "right": {...}}`` with each side's five joints plus its gripper. Pure —
    unit-tested.
    """
    from common.sensor_view import _vec5_to_dict

    v = np.asarray(vec12, dtype=float).reshape(-1)
    if v.shape[0] != 12:
        raise ValueError(f"expected a 12-channel vector, got {v.shape[0]}")
    return {
        "left": _vec5_to_dict(v[0:5], float(v[5])),
        "right": _vec5_to_dict(v[6:11], float(v[11])),
    }


def depth_png_path(root: Path, depth_name: str, episode: int, frame_idx: int) -> Path:
    """Path to the PNG16 depth frame the recorder wrote (see recording.depth)."""
    return (
        Path(root)
        / "extra"
        / "depth"
        / depth_name
        / f"episode_{episode:06d}"
        / f"{frame_idx:06d}.png"
    )


def frame_to_bgr(image) -> np.ndarray:
    """A decoded LeRobot image tensor/array → contiguous BGR uint8 (H, W, 3).

    Accepts CHW or HWC, float in [0, 1] or uint8, RGB channel order (the
    dataset's convention). Pure — unit-tested.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _resolve_root_repo(args) -> tuple[Path, str]:
    if args.dataset_root:
        root = Path(args.dataset_root).expanduser()
        repo_id = args.repo_id or root.name
    else:
        if not (args.dir and args.name):
            raise SystemExit("❌ give --dir and --name, or --dataset-root [--repo-id]")
        root = Path(args.dir).expanduser() / args.name
        repo_id = args.repo_id or args.name
    if not root.exists():
        raise SystemExit(f"❌ dataset not found at {root}")
    return root, repo_id


def _load_realsense(root: Path) -> tuple[str | None, float]:
    """(depth_name, depth_scale) from meta/realsense.json, or (None, 0.001)."""
    p = root / "meta" / "realsense.json"
    if not p.is_file():
        return None, 0.001
    meta = json.loads(p.read_text())
    return meta.get("depth_name"), float(meta.get("depth_scale_m_per_unit") or 0.001)


def load_depth_range(root, depth_name, episode, depth_scale):
    """Lock the depth colour range to the episode's first depth frame.

    Returns ``(near_m, far_m)`` from ``depth_range_from_frame`` on frame 0's
    PNG16 so replay scales depth exactly like the live view did during
    collection, or ``None`` if the frame is missing/empty (then the default
    metric window is used).
    """
    dp = depth_png_path(root, depth_name, episode, 0)
    if not dp.is_file():
        return None
    depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    return None if depth is None else depth_range_from_frame(depth, depth_scale)


def build_frame(
    ds, i, episode, camera_names, depth_name, depth_scale, root, depth_range=None
):
    """Composite one recorded frame into the live-view layout (BGR)."""
    from tool.test_sensor_rates import _camera_short_label

    item = ds[i]
    panels: list[ViewPanel] = []
    for key, name in camera_names:
        img = item.get(key)
        bgr = None if img is None else frame_to_bgr(img)
        hw = (bgr.shape[0], bgr.shape[1]) if bgr is not None else (480, 640)
        panels.append(
            ViewPanel(
                label=_camera_short_label(name),
                image_bgr=bgr,
                fallback_hw=hw,
                line1=_camera_short_label(name),
            )
        )
    if depth_name is not None:
        dp = depth_png_path(root, depth_name, episode, i)
        depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if dp.is_file() else None
        if depth is None:
            depth_bgr = None
        elif depth_range is not None:
            depth_bgr = colourise_depth(
                depth, depth_scale, depth_range[0], depth_range[1]
            )
        else:
            depth_bgr = colourise_depth(depth, depth_scale)
        panels.append(
            ViewPanel(
                label=_camera_short_label(depth_name),
                image_bgr=depth_bgr,
                fallback_hw=(480, 640),
                line1=_camera_short_label(depth_name),
            )
        )
    col1 = state12_to_side_dicts(item["observation.state"])
    col2 = state12_to_side_dicts(item["action"])
    return compose_sensor_view_frame(panels, col1, col2, "cmd", joint_strip=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", help="Collection directory (with --name)")
    parser.add_argument("--name", help="Dataset name (with --dir)")
    parser.add_argument(
        "--dataset-root", help="Dataset directory (instead of --dir/--name)"
    )
    parser.add_argument("--repo-id", help="Dataset repo id (defaults to the name)")
    parser.add_argument(
        "--episode", type=int, required=True, help="Episode id to replay"
    )
    parser.add_argument("--mp4", help="Also write the view to this mp4 path")
    parser.add_argument("--fps", type=int, default=None, help="Playback/output rate")
    parser.add_argument(
        "--loop", action="store_true", help="Repeat until q/Esc (window)"
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Do not open a window (mp4 only)"
    )
    args = parser.parse_args()

    if args.no_view and not args.mp4:
        raise SystemExit("❌ --no-view needs --mp4 (nothing to show or save otherwise)")

    root, repo_id = _resolve_root_repo(args)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[args.episode])
    n = len(ds)
    if n == 0:
        raise SystemExit(f"❌ episode {args.episode} has no frames")
    fps = args.fps or int(ds.meta.fps)
    camera_names = [(k, k[len(_IMAGE_PREFIX) :]) for k in ds.meta.camera_keys]
    depth_name, depth_scale = _load_realsense(root)
    depth_range = (
        None
        if depth_name is None
        else load_depth_range(root, depth_name, args.episode, depth_scale)
    )
    print(
        f"▶ replaying episode {args.episode}: {n} frames @ {fps} fps, "
        f"cameras={[n for _, n in camera_names]}"
        + (f", depth={depth_name}" if depth_name else "")
    )

    window = None if args.no_view else "replay"
    delay = max(1, int(1000 / fps))
    mp4_frames: list[np.ndarray] = []
    quit_requested = False
    while True:
        for i in range(n):
            frame = build_frame(
                ds,
                i,
                args.episode,
                camera_names,
                depth_name,
                depth_scale,
                root,
                depth_range,
            )
            if args.mp4:
                mp4_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if window is not None:
                cv2.imshow(window, frame)
                if (cv2.waitKey(delay) & 0xFF) in (27, ord("q")):
                    quit_requested = True
                    break
        if quit_requested or not (args.loop and window is not None):
            break
    if window is not None:
        cv2.destroyAllWindows()

    if args.mp4:
        import imageio.v2 as imageio

        out = Path(args.mp4)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, mp4_frames, fps=fps)
        print(f"  💾 wrote {out} ({len(mp4_frames)} frames)")


if __name__ == "__main__":
    main()
