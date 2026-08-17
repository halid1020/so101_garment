"""Curation helpers for a collected LeRobot v3.0 dataset (edit/delete on disk).

Deleting a recorded episode is not a file removal: v3.0 packs several episodes
into shared chunk files, so a real per-episode delete goes through LeRobot's
``delete_episodes`` (it re-encodes any video segment that mixes kept and deleted
episodes and renumbers the survivors 0..M-1). Our own side files under
``extra/`` (per-episode depth directories, sidecar and drift parquet) and the
two dataset-level meta JSONs LeRobot does not know about are re-indexed here with
the same old->new mapping, then swapped in atomically.

Because that rewrite touches the whole dataset, it costs tens of seconds even
for a single episode, which is far too slow to sit behind a click while an
operator reviews a session. Deletion is therefore two steps. Marking is
instant: the reviewer's decision goes into a small marker file
(``extra/soft_deleted.json``) and the marked episodes are hidden from the review
tool immediately. Compaction performs the real rewrite once for the whole batch.

The marked episodes remain on disk until compaction, so anything that CONSUMES a
dataset must treat a non-empty marker as "not ready": training on a dataset with
pending marks would train on the very episodes the reviewer threw away. Use
``read_soft_deleted`` to check.

The pure helpers (``deletion_mapping``, ``extra_reindex_ops``,
``surviving_indices``) are unit-tested; ``delete_episodes_in_place`` and
``compact_dataset`` perform the file operations. Shared by the web review tool
and any other curation entry point so the re-indexing lives once.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

# Marker file listing episodes the reviewer has deleted but that are still on
# disk. Lives inside the dataset so it travels with it (and so a reviewer cannot
# lose the decision by restarting the tool).
SOFT_DELETE_REL = "extra/soft_deleted.json"


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


def surviving_indices(total: int, marked: "list[int]") -> "list[int]":
    """Episode indices still visible after ``marked`` are hidden. Pure.

    Unlike ``deletion_mapping`` this does NOT renumber: a soft delete only hides
    episodes, so the survivors keep the on-disk indices that the video and
    per-episode routes need. Renumbering happens once, at compaction.
    """
    dead = set(marked)
    return [i for i in range(total) if i not in dead]


def read_soft_deleted(root: Path) -> "list[int]":
    """Episodes marked for deletion but still on disk. Never raises.

    A missing, empty or corrupt marker reads as "nothing marked": losing a
    reviewer's marks is far better than refusing to open the dataset, and the
    episodes themselves are still intact either way.
    """
    path = Path(root) / SOFT_DELETE_REL
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return sorted({int(i) for i in data.get("episodes", [])})
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def write_soft_deleted(root: Path, indices: "list[int]") -> "list[int]":
    """Replace the marker with ``indices`` (sorted, de-duplicated). Returns it.

    Writes via a temp file and an atomic rename inside ``extra/`` so an
    interrupted write cannot leave a half-written marker. Needs only the dataset
    to be writable, not its parent, so marking works on mounts where the
    in-place rewrite of ``delete_episodes_in_place`` could not run.
    """
    root = Path(root)
    clean = sorted({int(i) for i in indices})
    path = root / SOFT_DELETE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    payload = {"episodes": clean, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return clean


def clear_soft_deleted(root: Path) -> None:
    """Remove the marker file, if present."""
    path = Path(root) / SOFT_DELETE_REL
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def compact_dataset(root: Path, repo_id: str, depth_names: "list[str]") -> int:
    """Really delete every marked episode and clear the marker. Returns the new N.

    This is the expensive half of the two-step delete: it re-encodes the video
    segments that mixed kept and deleted episodes and renumbers the survivors, so
    it is run once for a whole batch of marks rather than per episode.
    """
    marked = read_soft_deleted(root)
    if not marked:
        return saved_episode_total(root)
    remaining = delete_episodes_in_place(root, repo_id, marked, depth_names)
    # The rewrite produced a fresh dataset directory, so the marker is already
    # gone with the old one; clear defensively in case a future rewrite copies
    # extra/ wholesale.
    clear_soft_deleted(root)
    return remaining


def saved_episode_total(root: Path) -> int:
    """Episode count recorded in the dataset's own metadata (0 if unreadable)."""
    try:
        with open(Path(root) / "meta" / "info.json", "r") as f:
            return int(json.load(f).get("total_episodes", 0))
    except (OSError, ValueError, TypeError):
        return 0


def episode_lengths(meta: Any) -> "list[int]":
    """Per-episode frame counts from a ``LeRobotDatasetMetadata``. Pure-ish."""
    return [int(meta.episodes[k]["length"]) for k in range(meta.total_episodes)]


def read_episode_lengths(root: Path) -> "tuple[dict[int, int], list[str]]":
    """``({episode_index: length}, [unreadable files])`` straight from the parquet.

    Building a ``LeRobotDatasetMetadata`` just to list episodes is expensive (it
    loads the episode metadata through HuggingFace ``datasets``: measured at
    ~4.9 s for 58 episodes, against ~0.3 s here) and it is all-or-nothing -- one
    truncated parquet raises and no episode can be listed at all. Reading the
    columns directly is both quicker and per-file fault tolerant, which is what a
    review tool needs: a half-written file from an interrupted session should cost
    the episodes in THAT file, not the whole session.

    Returns the lengths it could read and the files it could not, so the caller
    can tell the operator which part of the dataset is damaged.
    """
    root = Path(root)
    lengths: dict[int, int] = {}
    bad: list[str] = []
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.is_dir():
        return lengths, bad
    import pyarrow.parquet as pq  # local: keeps the import off pure-helper users

    for path in sorted(meta_dir.rglob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["episode_index", "length"])
            data = table.to_pydict()
            for index, length in zip(data["episode_index"], data["length"]):
                lengths[int(index)] = int(length)
        except Exception:
            # Any failure to read is the same outcome for the caller: these
            # episodes' lengths are unknown. Keep going with the rest.
            bad.append(str(path.relative_to(root)))
    return lengths, bad


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
