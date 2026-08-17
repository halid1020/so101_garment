"""Curation helpers for a collected LeRobot v3.0 dataset (edit/delete on disk).

Deleting a recorded episode is not a file removal: v3.0 packs several episodes
into shared chunk files, so a real per-episode delete goes through LeRobot's
``delete_episodes`` (it re-encodes any video segment that mixes kept and deleted
episodes and renumbers the survivors 0..M-1). Our own side files under
``extra/`` (per-episode depth directories, sidecar and drift parquet) and the
two dataset-level meta JSONs LeRobot does not know about are re-indexed here with
the same old->new mapping, then swapped in atomically.

The pure helpers (``deletion_mapping``, ``extra_reindex_ops``) are unit-tested;
``delete_episodes_in_place`` performs the file operations. Shared by the web
review tool and any other curation entry point so the re-indexing lives once.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any


class ReadOnlyDatasetError(OSError):
    """Raised when a delete is attempted on a read-only filesystem.

    Deletion rewrites the dataset in place (a sibling temp dir plus atomic
    renames in the parent directory), so it needs write access to the dataset's
    parent. On a read-only mount that is impossible; callers surface this as a
    clean message instead of a mid-operation ``OSError``.
    """


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


def episode_lengths(meta: Any) -> "list[int]":
    """Per-episode frame counts from a ``LeRobotDatasetMetadata``. Pure-ish."""
    return [int(meta.episodes[k]["length"]) for k in range(meta.total_episodes)]


def delete_episodes_in_place(
    root: Path, repo_id: str, indices: "list[int]", depth_names: "list[str]"
) -> int:
    """Delete ``indices`` on the drive and renumber the rest. Returns the new N.

    Builds a re-indexed copy in a sibling temp dir (LeRobot's ``delete_episodes``
    for data/videos/meta, plus our ``extra/`` side files and the two extra meta
    JSONs it does not carry over), then atomically swaps it in. The original is
    moved aside first and only removed once the swap succeeds, so a failure
    partway never loses the source dataset. Accepts one or many indices so a
    batch delete is a single re-encode + re-index + swap.
    """
    root = Path(root)
    indices = sorted({int(i) for i in indices})
    if not indices:
        raise ValueError("no episodes given to delete")

    # The rewrite creates a sibling temp dir and renames within the parent, so
    # a read-only mount cannot be edited. Fail early with a clear message rather
    # than partway through LeRobot's re-encode (and before loading the dataset).
    if not os.access(root.parent, os.W_OK):
        raise ReadOnlyDatasetError(
            f"dataset is on a read-only filesystem ({root.parent}) — remount "
            "read-write to delete episodes"
        )

    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    total = ds.meta.total_episodes
    if indices[0] < 0 or indices[-1] >= total:
        raise ValueError(f"episode index out of range 0..{total - 1}: {indices}")
    if len(indices) >= total:
        raise ValueError("cannot delete all episodes in a dataset")

    mapping = deletion_mapping(total, indices)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    temp = root.parent / f"{root.name}.tmp-{stamp}"

    delete_episodes(ds, indices, output_dir=temp, repo_id=repo_id)

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
    return total - len(indices)
