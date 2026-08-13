"""Interactive browser for a collected LeRobot dataset (cv2 HighGUI).

One window with three parts: an episode LIST strip on the left, the composited
sensor-view frame on the right (the SAME layout the operator watched while
collecting, produced by ``tool/replay_recording.build_frame``), and a draggable
"frame" scrub bar. Click an episode to play it; drag the scrub bar to seek;
space toggles play/pause. With ``--allow-delete`` a flawed episode can be
removed on the drive — the dataset and its side files are rewritten with the
remaining episodes renumbered, so the result stays a consistent dataset.

Deletion is destructive and is therefore OFF unless ``--allow-delete`` is
passed; even then it asks for an on-screen confirmation. Because v3.0 datasets
pack several episodes into shared chunk files, a real per-episode delete goes
through LeRobot's ``delete_episodes`` (it re-encodes any video segment that
mixes kept and deleted episodes) and we re-index our own side files
(``extra/`` depth, sidecar and drift) with the same old->new mapping.

Usage:

    venv/bin/python tool/dataset_browser.py --dir /media/hdd/so101 --name towel_fold
    venv/bin/python tool/dataset_browser.py --dataset-root /media/hdd/so101/towel_fold
    venv/bin/python tool/dataset_browser.py --dir /media/hdd/so101 --name towel_fold \\
        --allow-delete            # enable episode deletion (asks to confirm)

Controls: click an episode to select/play · space play/pause · drag the frame
bar to scrub · d arm delete (with --allow-delete) · y/n confirm/cancel · q/Esc.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import]
import numpy as np

from tool.replay_recording import (
    _load_realsense,
    _resolve_root_repo,
    build_frame,
    load_depth_range,
    saved_episode_count,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_IMAGE_PREFIX = "observation.images."
_LIST_W = 260
_ROW_H = 34
_LIST_Y0 = 44


# ── Pure helpers (unit-tested; no cv2 window, no hardware) ────────────────────


def deletion_mapping(total: int, to_delete: "list[int]") -> "dict[int, int]":
    """``{old_index: new_index}`` for the episodes KEPT after a deletion.

    Kept episodes stay in ascending order and are renumbered 0..M-1, exactly
    matching LeRobot's own re-indexing in ``delete_episodes`` so our side files
    line up with the rewritten dataset. Deleted indices are absent from the map.
    Pure.
    """
    dead = set(to_delete)
    kept = [i for i in range(total) if i not in dead]
    return {old: new for new, old in enumerate(kept)}


def extra_reindex_ops(
    mapping: "dict[int, int]", depth_names: "list[str]"
) -> "list[tuple[str, str]]":
    """``(src_rel, dst_rel)`` moves that re-index our ``extra/`` side files.

    For every kept episode ``old -> new`` this maps the per-episode drift and
    sidecar parquet and each depth stream's per-episode directory from its old
    six-digit index to its new one. Paths are relative to the dataset root so a
    caller joins them with the source and destination roots; missing sources are
    skipped by the caller. Pure.
    """
    ops: list[tuple[str, str]] = []
    for old, new in sorted(mapping.items()):
        ops.append((f"extra/drift_{old:06d}.parquet", f"extra/drift_{new:06d}.parquet"))
        ops.append(
            (f"extra/episode_{old:06d}.parquet", f"extra/episode_{new:06d}.parquet")
        )
        for dname in depth_names:
            ops.append(
                (
                    f"extra/depth/{dname}/episode_{old:06d}",
                    f"extra/depth/{dname}/episode_{new:06d}",
                )
            )
    return ops


def episode_at_y(y: int, n_visible: int, row_h: int = _ROW_H, y0: int = _LIST_Y0):
    """Visible row under a click ``y`` (0-based), or ``None`` outside the rows.

    Returns the row within the currently displayed window; the caller adds the
    scroll offset to get the episode index. Pure.
    """
    if y < y0:
        return None
    row = (y - y0) // row_h
    if 0 <= row < n_visible:
        return int(row)
    return None


def list_scroll_offset(selected: int, n: int, max_rows: int) -> int:
    """First visible episode so ``selected`` stays on screen. Pure."""
    if n <= max_rows:
        return 0
    return max(0, min(selected - max_rows // 2, n - max_rows))


# ── Deletion (I/O; orchestrates LeRobot delete + side-file re-index + swap) ───


def delete_episode_in_place(
    root: Path, repo_id: str, index: int, depth_names: "list[str]"
) -> int:
    """Delete episode ``index`` on the drive and renumber the rest. Returns N-1.

    Builds a re-indexed copy in a sibling temp dir (LeRobot's ``delete_episodes``
    for data/videos/meta, plus our ``extra/`` side files and the two extra meta
    JSONs it does not know about), then atomically swaps it in. The original is
    moved aside first and only removed once the swap succeeds, so a failure
    partway never loses the source dataset.
    """
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    ds = LeRobotDataset(repo_id, root=root)
    total = ds.meta.total_episodes
    if total <= 1:
        raise SystemExit("cannot delete the only episode in a dataset")

    mapping = deletion_mapping(total, [index])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    temp = root.parent / f"{root.name}.tmp-{stamp}"

    delete_episodes(ds, [index], output_dir=temp, repo_id=repo_id)

    # LeRobot rewrites meta/ from scratch, so our two dataset-level JSONs are not
    # carried over — copy them across.
    for fn in ("realsense.json", "action_space.json"):
        src = root / "meta" / fn
        if src.is_file():
            (temp / "meta").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, temp / "meta" / fn)

    # Re-index our side files (extra/) into the temp dataset per the mapping.
    for src_rel, dst_rel in extra_reindex_ops(mapping, depth_names):
        src = root / src_rel
        if not src.exists():
            continue
        dst = temp / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    trash = root.parent / f"{root.name}.trash-{stamp}"
    os.replace(root, trash)  # atomic within the filesystem
    os.replace(temp, root)
    shutil.rmtree(trash, ignore_errors=True)
    return total - 1


# ── List rendering (array ops; no window) ─────────────────────────────────────


def render_episode_list(
    lengths: "list[int]",
    selected: int,
    height: int,
    first_visible: int,
    max_rows: int,
    delete_armed: bool,
    allow_delete: bool,
) -> np.ndarray:
    """The left LIST strip (BGR), highlighting the selected episode."""
    strip = np.zeros((height, _LIST_W, 3), dtype=np.uint8)
    cv2.putText(
        strip, f"episodes ({len(lengths)})", (8, 26), _FONT, 0.6, (200, 200, 200), 1
    )
    for row in range(min(max_rows, len(lengths) - first_visible)):
        k = first_visible + row
        y = _LIST_Y0 + row * _ROW_H
        if k == selected:
            cv2.rectangle(
                strip, (4, y - 2), (_LIST_W - 4, y + _ROW_H - 8), (60, 60, 60), -1
            )
        col = (0, 255, 0) if k == selected else (180, 180, 180)
        cv2.putText(strip, f"ep {k}  {lengths[k]}f", (10, y + 20), _FONT, 0.6, col, 1)
    hint = "d: delete" if allow_delete else "delete off"
    cv2.putText(strip, hint, (8, height - 12), _FONT, 0.5, (120, 120, 120), 1)
    if delete_armed:
        cv2.putText(
            strip, "DELETE? y/n", (8, height - 34), _FONT, 0.55, (0, 180, 255), 2
        )
    return strip


# ── Interactive app ───────────────────────────────────────────────────────────


def _episode_lengths(meta) -> "list[int]":
    return [int(meta.episodes[k]["length"]) for k in range(meta.total_episodes)]


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
        "--allow-delete",
        action="store_true",
        help="Enable deleting an episode on the drive (destructive; asks to confirm)",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Window max width")
    parser.add_argument("--max-height", type=int, default=720, help="Window max height")
    args = parser.parse_args()

    root, repo_id = _resolve_root_repo(args)

    # Purely local browsing; never reach out to the Hub (an incomplete dataset
    # would otherwise surface a misleading 401 on the bare dataset name).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if saved_episode_count(root) == 0:
        raise SystemExit(
            f"❌ {root} has no saved episodes yet — nothing to browse. "
            "(Record and save at least one episode first.)"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    lengths = _episode_lengths(meta)
    depth_name, depth_scale = _load_realsense(root)
    depth_names = [depth_name] if depth_name else []

    window = "dataset browser"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    # Shared UI state; the mouse callback only records a raw click for the loop.
    ui: dict = {"click": None}

    def _on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["click"] = (x, y)

    cv2.setMouseCallback(window, _on_mouse)

    # Per-episode playback state, reloaded whenever the selection changes.
    open_ds: dict[str, Any] = {
        "episode": -1,
        "ds": None,
        "n": 0,
        "cams": [],
        "range": None,
    }

    def _load_episode(ep: int) -> None:
        ds = LeRobotDataset(repo_id, root=root, episodes=[ep])
        open_ds["episode"] = ep
        open_ds["ds"] = ds
        open_ds["n"] = len(ds)
        open_ds["cams"] = [(k, k[len(_IMAGE_PREFIX) :]) for k in ds.meta.camera_keys]
        open_ds["range"] = (
            None
            if depth_name is None
            else load_depth_range(root, depth_name, ep, depth_scale)
        )
        cv2.setTrackbarMax("frame", window, max(open_ds["n"] - 1, 1))
        cv2.setTrackbarPos("frame", window, 0)

    selected = 0
    frame_idx = 0
    playing = True
    delete_armed = False
    tb_pos = 0
    fps = int(meta.fps) or 30
    delay = max(1, int(1000 / fps))

    cv2.createTrackbar("frame", window, 0, max(lengths[0] - 1, 1), lambda _v: None)
    _load_episode(0)

    while True:
        # 1) Consume a pending click: an episode row, else ignored.
        if ui["click"] is not None:
            cx, cy = ui["click"]
            ui["click"] = None
            if cx < _LIST_W:
                max_rows = max(1, (args.max_height - _LIST_Y0 - 50) // _ROW_H)
                first = list_scroll_offset(selected, len(lengths), max_rows)
                row = episode_at_y(cy, min(max_rows, len(lengths) - first))
                if row is not None:
                    selected = first + row
                    delete_armed = False

        # 2) Selection changed -> (re)load that episode from the start.
        if selected != open_ds["episode"]:
            _load_episode(selected)
            frame_idx, tb_pos, playing = 0, 0, True

        # 3) Scrub bar: a user drag (pos != our last programmatic set) seeks+pauses.
        pos = cv2.getTrackbarPos("frame", window)
        if pos != tb_pos:
            frame_idx = min(max(pos, 0), max(open_ds["n"] - 1, 0))
            tb_pos = pos
            playing = False

        # 4) Compose the list strip beside the recorded frame.
        frame = build_frame(
            open_ds["ds"],
            frame_idx,
            selected,
            open_ds["cams"],
            depth_name,
            depth_scale,
            root,
            open_ds["range"],
        )
        max_rows = max(1, (frame.shape[0] - _LIST_Y0 - 50) // _ROW_H)
        first = list_scroll_offset(selected, len(lengths), max_rows)
        strip = render_episode_list(
            lengths,
            selected,
            frame.shape[0],
            first,
            max_rows,
            delete_armed,
            args.allow_delete,
        )
        cv2.putText(
            frame,
            f"ep {selected}  frame {frame_idx + 1}/{open_ds['n']}"
            + ("  [PAUSED]" if not playing else ""),
            (8, frame.shape[0] - 12),
            _FONT,
            0.6,
            (255, 255, 255),
            1,
        )
        if delete_armed:
            cv2.putText(
                frame,
                f"delete episode {selected}? press y to confirm, n to cancel",
                (8, 28),
                _FONT,
                0.7,
                (0, 180, 255),
                2,
            )
        from tool.test_sensor_rates import _fit_to_screen

        canvas = _fit_to_screen(
            np.hstack([strip, frame]), args.max_width, args.max_height
        )
        cv2.imshow(window, canvas)

        # 5) Advance playback (unless paused) and reflect it on the scrub bar.
        if playing and open_ds["n"] > 0:
            frame_idx = (frame_idx + 1) % open_ds["n"]
            tb_pos = frame_idx
            cv2.setTrackbarPos("frame", window, frame_idx)

        # 6) Keys.
        key = cv2.waitKey(delay) & 0xFF
        if key == ord(" "):
            playing = not playing
        elif key == ord("d") and args.allow_delete and len(lengths) > 1:
            delete_armed = True
            playing = False
        elif key == ord("n"):
            delete_armed = False
        elif key == ord("y") and delete_armed:
            delete_armed = False
            print(f"🗑️  deleting episode {selected} …")
            new_total = delete_episode_in_place(root, repo_id, selected, depth_names)
            meta = LeRobotDatasetMetadata(repo_id, root=root)
            lengths = _episode_lengths(meta)
            selected = min(selected, new_total - 1)
            open_ds["episode"] = -1  # force reload of the now-renumbered selection
            print(f"   ✅ {new_total} episode(s) remain")
        elif key in (27, ord("q")):
            if delete_armed:
                delete_armed = False  # Esc cancels an armed delete first
            else:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
