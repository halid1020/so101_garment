"""Camera-ablation VIEWS of a collected dataset.

An ablation asks what a policy loses when a camera is taken away. The honest
way to answer it is to train on the same episodes, the same frames and the same
actions, changing only which cameras the policy can see -- so the datasets that
differ must differ in nothing else.

Copying a dataset per camera set would do that and is wasteful: the videos are
99% of the bytes (525 MB of 543 MB for ``cube-pnp-new``) and they are exactly
the part that is shared. A **view** is a directory that LeRobot opens as an
ordinary dataset while owning almost nothing: the metadata is rewritten to name
only the cameras being kept, and each kept camera's video directory is a
symlink back to the source. A three-camera dataset therefore yields three views
for a few megabytes.

Rewriting the metadata rather than overriding the policy's inputs is what makes
this cheap at training time too. ``LeRobotDatasetMetadata.video_keys`` reads the
feature table in ``meta/info.json``, so a camera that is not named there is
never decoded -- and ``dataset_to_policy_features`` then derives exactly the
inputs the policy should see, with no ``--policy.input_features`` override to
keep in step with the data. MEASURED on this laptop, batch 8 with 4 workers:
127 ms/batch for three cameras, 56 ms for one.

Views also normalise the task string, which matters for one policy and not the
others. ``cube-pnp-new`` carries two spellings of one instruction -- "place it
on the plate" and "place it one the plate" -- because the string was retyped
partway through collection, leaving a minority of episodes labelled with the
typo. ACT and Diffusion ignore language entirely, but pi0.5 is conditioned on
it, and that minority would otherwise teach it a second instruction for one
task. The source dataset is never modified: the correction lives in the view,
where it is one rewritten column.

What a view holds, and why:

  * ``videos/<kept camera>/`` -- a symlink to the source. The bytes are shared.
  * ``meta/`` and ``data/`` -- REAL files, rewritten. They are small (under
    4 MB together here), and both carry per-camera and per-task columns that
    have to agree with ``info.json`` or the dataset describes itself wrongly.
  * ``extra/`` -- absent. The sidecar streams are a recording artefact; nothing
    in training reads them.

Building is atomic and idempotent, because the cluster builds views from
several array tasks at once: each build lands in a temporary directory and is
renamed into place, and a task that loses the race keeps the winner's view.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

CAMERA_PREFIX = "observation.images."
_VIDEO_DTYPES = ("video", "image")

# Camera names are long because they say which arm they ride on; a directory
# name repeating them in full is unreadable. Only the redundant middle word is
# dropped, so `wrist_camera_left` -> `wrist_left` stays unambiguous.
_SLUG_ELISIONS = (("wrist_camera_", "wrist_"),)


# pi0.5 was pretrained with three fixed camera slots under openpi's names, and a
# finetune reaches them by renaming rather than by re-deriving features: each
# slot carries what it learned about that viewpoint, so a rig camera should land
# on the slot that means the same thing. A slot left unmapped is not an error --
# pi0.5 fills it with a padded image and a zero mask, which is exactly how an
# ablated camera should read to the model.
PI05_SLOTS = {
    "central": "observation.images.base_0_rgb",
    "wrist_camera_left": "observation.images.left_wrist_0_rgb",
    "wrist_camera_right": "observation.images.right_wrist_0_rgb",
}

# The slots themselves, in the order an unnamed camera is given one. The
# overhead slot first because it is the one whose viewpoint is least
# arm-specific, then the two wrists left-to-right.
PI05_SLOT_ORDER = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)

# Short names accepted in a run matrix's `slots` column, so a row can say
# `central=base,left_arm_left_gripper=left_wrist` instead of repeating the
# openpi feature keys.
PI05_SLOT_ALIASES = {
    "base": "observation.images.base_0_rgb",
    "left_wrist": "observation.images.left_wrist_0_rgb",
    "right_wrist": "observation.images.right_wrist_0_rgb",
}

# Named COMPOSITES tile several cameras into ONE image feature.
#
# A policy whose architecture fixes the number of views cannot simply be given
# more of them: FastWAM concatenates its image features into a single frame of
# `policy.image_size`, so five 640x480 cameras have nowhere to go. Tiling the
# four fingertip cameras into one square feature spends one view on all four
# instead of losing three of them, and keeps every tactile signal in front of
# the model.
#
# The grid is row-major and must be square-ish: four cameras tile 2x2, each
# quadrant a quarter of the output's width and height.
COMPOSITES: "dict[str, tuple[str, ...]]" = {
    "tactile_quad": (
        "left_arm_left_gripper",
        "left_arm_right_gripper",
        "right_arm_left_gripper",
        "right_arm_right_gripper",
    ),
}


class ViewError(Exception):
    """A view was asked for that the source dataset cannot provide."""


# --------------------------------------------------------------------------
# pure: what the metadata says, and what it should say
# --------------------------------------------------------------------------
def camera_keys(info: dict) -> list[str]:
    """The dataset's camera feature keys, in the order ``info.json`` lists them."""
    return [
        key
        for key, ft in (info.get("features") or {}).items()
        if ft.get("dtype") in _VIDEO_DTYPES
    ]


def short_name(key: str) -> str:
    """``observation.images.central`` -> ``central``."""
    return key[len(CAMERA_PREFIX) :] if key.startswith(CAMERA_PREFIX) else key


def available_camera_names(info: dict) -> list[str]:
    """The dataset's camera SHORT names, plus every composite it could build.

    A composite is offered only when the dataset records all of its parts, so a
    run matrix naming one is judged against what this dataset can actually
    produce rather than against a list of names that exist somewhere.
    """
    have = [short_name(k) for k in camera_keys(info)]
    known = set(have)
    return have + [
        name for name, parts in COMPOSITES.items() if known.issuperset(parts)
    ]


def resolve_cameras(info: dict, names: Iterable[str]) -> list[str]:
    """Camera names (short or full, any order) -> full keys in dataset order.

    ``all`` selects every camera. Resolving through the dataset rather than
    trusting the caller is what turns a typo in a run matrix into a message
    naming the cameras that do exist, instead of a view that silently trains on
    fewer inputs than the experiment intended.
    """
    available = camera_keys(info)
    if not available:
        raise ViewError("dataset has no camera features")

    wanted: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            wanted.extend(available)
            continue
        key = name if name.startswith(CAMERA_PREFIX) else CAMERA_PREFIX + name
        if key not in available:
            raise ViewError(
                f"no camera {name!r} in this dataset; it has: "
                f"{', '.join(short_name(k) for k in available)}"
            )
        wanted.append(key)

    if not wanted:
        raise ViewError("no cameras selected")
    # Dataset order, duplicates dropped: the same set asked for in two orders is
    # one view, not two.
    return [key for key in available if key in set(wanted)]


def view_slug(info: dict, keep: Sequence[str]) -> str:
    """A short, stable directory name for a camera set (``all``, ``central+wrist_left``)."""
    available = camera_keys(info)
    if list(keep) == available:
        return "all"
    parts = []
    for key in keep:
        name = short_name(key)
        for prefix, replacement in _SLUG_ELISIONS:
            if name.startswith(prefix):
                name = replacement + name[len(prefix) :]
        parts.append(name)
    return "+".join(parts)


def filtered_info(info: dict, keep: Sequence[str]) -> dict:
    """``info.json`` with every camera feature except ``keep`` removed."""
    drop = set(camera_keys(info)) - set(keep)
    out = dict(info)
    out["features"] = {k: v for k, v in info["features"].items() if k not in drop}
    return out


def filtered_stats(stats: dict, keep: Sequence[str], cameras: Sequence[str]) -> dict:
    """``stats.json`` with the dropped cameras' entries removed."""
    drop = set(cameras) - set(keep)
    return {k: v for k, v in stats.items() if k not in drop}


def dropped_meta_columns(
    columns: Iterable[str], keep: Sequence[str], cameras: Sequence[str]
) -> list[str]:
    """Episode-metadata columns belonging to cameras that this view drops.

    The per-episode table carries a block of columns per camera (``videos/<key>/…``
    for where its frames live, ``stats/<key>/…`` for its pixel statistics). They
    must go with the feature, or the view claims statistics for a camera it does
    not have.
    """
    drop = set(cameras) - set(keep)
    return [
        col
        for col in columns
        if any(
            col.startswith(f"videos/{k}/") or col.startswith(f"stats/{k}/")
            for k in drop
        )
    ]


def task_remap(
    tasks: Sequence[str],
    counts: dict[int, int] | None = None,
    canonical: str | None = None,
) -> tuple[str, dict[int, int]]:
    """Collapse a dataset's task strings onto one, keeping positions valid.

    Returns the canonical string and a mapping from each old ``task_index`` to
    its new one. The reader resolves a task POSITIONALLY
    (``meta.tasks.iloc[task_index].name``), so the rewritten table and the
    remapped column have to move together -- which is exactly why this returns
    both and the caller applies them in one pass.

    With no ``canonical`` given, the spelling covering the most FRAMES wins.
    Position is no guide: task indices are handed out in first-seen order, and
    on ``cube-pnp-new`` index 0 is the typo that 9 episodes carry while index 1
    is the spelling the other 109 use. Weight of evidence is what identifies the
    instruction the operator meant; order identifies only which was typed first.
    """
    if not tasks:
        raise ViewError("dataset has no tasks")
    if canonical is not None:
        chosen = canonical
    elif counts:
        weights = counts
        chosen = tasks[max(range(len(tasks)), key=lambda i: weights.get(i, 0))]
    else:
        chosen = tasks[0]
    return chosen, {old: 0 for old in range(len(tasks))}


def pi05_rename_map(
    keep: Sequence[str], slots: "dict[str, str] | None" = None
) -> dict[str, str]:
    """This view's cameras -> the pi0.5 slots they should be fed into.

    Only cameras the view actually has are mapped. The slots left over stay
    empty on purpose: that is how a camera is ablated for a policy whose
    architecture has a fixed number of views.

    A camera whose viewpoint matches one of pi0.5's own -- an overhead view, a
    wrist -- takes that slot by NAME, so it inherits what the base learned about
    it. A camera with no counterpart (this rig's fingertip cameras have none:
    pi0.5 was never pretrained on a gel image) takes the next slot still free,
    in ``PI05_SLOT_ORDER``. That is a weaker claim than the named case and is
    worth saying out loud, but it is the right one: the alternative was refusing
    to train pi0.5 on a tactile dataset at all.

    ``slots`` overrides both, and is how a run matrix pins the assignment for an
    ablation. Its values may be the openpi feature keys or the short aliases in
    ``PI05_SLOT_ALIASES``.
    """
    out: "dict[str, str]" = {}
    if len(keep) > len(PI05_SLOT_ORDER):
        raise ViewError(
            f"pi0.5 has {len(PI05_SLOT_ORDER)} image slots but this view has "
            f"{len(keep)} cameras ({', '.join(short_name(k) for k in keep)}); "
            "build a view with fewer, or map them in the slots column"
        )

    if slots:
        for camera, slot in slots.items():
            key = camera if camera.startswith(CAMERA_PREFIX) else CAMERA_PREFIX + camera
            if key not in list(keep):
                raise ViewError(
                    f"slots names camera {short_name(key)!r}, which this view "
                    f"does not have: {', '.join(short_name(k) for k in keep)}"
                )
            full = PI05_SLOT_ALIASES.get(slot, slot)
            if full not in PI05_SLOT_ORDER:
                raise ViewError(
                    f"{slot!r} is not a pi0.5 slot; want one of "
                    f"{', '.join(sorted(PI05_SLOT_ALIASES))}"
                )
            out[key] = full
        return out

    taken: "set[str]" = set()
    unnamed: "list[str]" = []
    for key in keep:
        named = PI05_SLOTS.get(short_name(key))
        if named is None:
            unnamed.append(key)
            continue
        out[key] = named
        taken.add(named)
    free = [s for s in PI05_SLOT_ORDER if s not in taken]
    for key, slot in zip(unnamed, free):
        out[key] = slot
    # zip stops at the shorter list, so a camera left without a slot means the
    # named ones already filled them -- which the length check above cannot
    # catch, because it counts cameras rather than collisions.
    if len(out) < len(keep):
        raise ViewError(
            "pi0.5's slots are already taken by the named cameras; map them "
            f"explicitly in the slots column ({', '.join(sorted(PI05_SLOT_ALIASES))})"
        )
    return out


# --------------------------------------------------------------------------
# building a view on disk
# --------------------------------------------------------------------------
def _rewrite_parquet_dir(
    src_dir: Path,
    dst_dir: Path,
    drop_columns: Sequence[str] = (),
    task_index_map: dict[int, int] | None = None,
    task_strings: dict[int, str] | None = None,
) -> None:
    """Copy a tree of parquet files, dropping columns and remapping task indices.

    Types are preserved by editing the arrow table in place rather than taking a
    pandas round trip, which would silently widen ints and turn the fixed-size
    lists the metadata uses into something LeRobot reads back differently.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    for src in sorted(src_dir.rglob("*.parquet")):
        table = pq.read_table(src)
        drop = [c for c in drop_columns if c in table.column_names]
        if drop:
            table = table.drop(drop)
        if task_index_map and "task_index" in table.column_names:
            col = table.column("task_index")
            mapped = pa.array(
                [
                    None if v is None else task_index_map.get(v, v)
                    for v in col.to_pylist()
                ],
                type=col.type,
            )
            table = table.set_column(
                table.column_names.index("task_index"), "task_index", mapped
            )
        if task_strings is not None and "tasks" in table.column_names:
            col = table.column("tasks")
            canonical = task_strings[0]
            mapped = pa.array(
                [
                    None if v is None else [canonical for _ in v]
                    for v in col.to_pylist()
                ],
                type=col.type,
            )
            table = table.set_column(table.column_names.index("tasks"), "tasks", mapped)
        out = dst_dir / src.relative_to(src_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)


def _write_view(
    src: Path, work: Path, keep: Sequence[str], canonical_task: str | None
) -> None:
    import pandas as pd

    info = json.loads((src / "meta" / "info.json").read_text())
    cameras = camera_keys(info)

    (work / "meta").mkdir(parents=True)
    info_out = filtered_info(info, keep)

    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        (work / "meta" / "stats.json").write_text(
            json.dumps(filtered_stats(stats, keep, cameras), indent=4)
        )

    # Tasks: one canonical string, and the index map that keeps the data honest.
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_list = list(tasks_df.index)
    index_map: dict[int, int] | None = None
    task_strings: dict[int, str] | None = None
    if canonical_task is not None or len(task_list) > 1:
        chosen, index_map = task_remap(
            task_list, _task_frame_counts(src / "data"), canonical_task
        )
        task_strings = {0: chosen}
        new_tasks = pd.DataFrame(
            {"task_index": [0]}, index=pd.Index([chosen], name="task")
        )
        new_tasks.to_parquet(work / "meta" / "tasks.parquet")
        info_out["total_tasks"] = 1
    else:
        shutil.copy2(src / "meta" / "tasks.parquet", work / "meta" / "tasks.parquet")

    (work / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))

    _rewrite_parquet_dir(
        src / "meta" / "episodes",
        work / "meta" / "episodes",
        drop_columns=dropped_meta_columns(
            _all_columns(src / "meta" / "episodes"), keep, cameras
        ),
        task_index_map=index_map,
        task_strings=task_strings,
    )
    _rewrite_parquet_dir(src / "data", work / "data", task_index_map=index_map)

    # The bytes that matter are shared, not copied.
    (work / "videos").mkdir()
    for key in keep:
        (work / "videos" / key).symlink_to((src / "videos" / key).resolve())


def _task_frame_counts(data_dir: Path) -> dict[int, int]:
    """How many frames each ``task_index`` covers -- only that column is read."""
    import pyarrow.parquet as pq

    counts: dict[int, int] = {}
    for path in sorted(data_dir.rglob("*.parquet")):
        for value in (
            pq.read_table(path, columns=["task_index"]).column("task_index").to_pylist()
        ):
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
    return counts


def _all_columns(parquet_dir: Path) -> list[str]:
    import pyarrow.parquet as pq

    for path in sorted(parquet_dir.rglob("*.parquet")):
        return list(pq.read_schema(path).names)
    return []


def build_view(
    src: Path | str,
    dst: Path | str,
    cameras: Iterable[str],
    canonical_task: str | None = None,
    force: bool = False,
) -> Path:
    """Create (or reuse) a camera view of ``src`` at ``dst``.

    Idempotent: an existing view is kept unless ``force``. The build is atomic,
    so a reader never sees a half-written dataset and concurrent builders cannot
    corrupt each other -- the loser of a race discards its own work.
    """
    src, dst = Path(src), Path(dst)
    if not (src / "meta" / "info.json").is_file():
        raise ViewError(f"no dataset at {src}")

    if dst.exists() and not force:
        return dst
    if dst.exists() and force:
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    keep = resolve_cameras(info, cameras)

    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        _write_view(src, work / "view", keep, canonical_task)
        try:
            os.replace(work / "view", dst)
        except OSError:
            # Another task finished this same view first. Its copy is as good as
            # ours by construction, so keep it.
            if not dst.exists():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dst
