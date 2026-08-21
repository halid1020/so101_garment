"""Dataset browsing, playback and episode curation for the rig console.

This is the browser tool that used to be ``tool/dataset_web.py``, moved here
unchanged when the console grew its other panes. It serves the DATASET list
(every dataset under the collection directory with at least one saved episode),
the RECORDING list for a selected dataset, and the sensor VIEW of a selected
recording.

The view plays the recorded camera files themselves, side by side and on one
clock, with the episode's joint values drawn beside them. Nothing is decoded or
encoded to answer a click: the recorded videos already hold the frames a
reviewer wants, the browser can decode them, and each episode is a timestamp
window inside a shared file. So opening a recording costs a range request rather
than a render. The composited layout the operator watched while collecting is
still built on demand (``render_episode_mp4``) for the cases direct playback
cannot cover -- a dataset recording depth, which is stored as per-frame images
rather than a playable stream, or a browser without AV1 -- and for downloading
an episode as a single file.

Episode deletion happens in two steps, because really removing one episode from
a v3.0 dataset re-encodes and renumbers the whole thing and takes far too long
to sit behind a click. Deleting MARKS the episode: the row disappears at once
and the decision is recorded in the dataset, reversibly. "Remove for good" then
compacts: one rewrite for the whole batch, via
``common.recording.dataset_edit.compact_dataset``.

Until a dataset is compacted its marked episodes are still on disk, so anything
that trains on the dataset would still see them. The pane says so, and the
real-VLA training scripts refuse to start while marks are pending.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from aiohttp import web  # type: ignore[import]

# This pane only ever reads datasets from the local drive. Without this, a
# dataset whose metadata is momentarily incomplete makes LeRobot fall back to a
# Hub lookup on the bare dataset name and report an offline-mode or 401 failure,
# which says nothing about the real problem. Set before any LeRobot import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from common.recording.dataset_check import (
    DatasetDamaged,
    dataset_integrity,
    repair_phantom_episodes,
)
from common.recording.dataset_edit import (
    ReadOnlyDatasetError,
    compact_dataset,
    read_episode_lengths,
    read_soft_deleted,
    saved_episode_total,
    surviving_indices,
    writability_problem,
    write_soft_deleted,
)
from common.web.jobs import finish, new_job, refuse_while_busy
from common.web.lifecycle import directory_size, is_working_dir, read_dataset_meta
from common.web.util import in_executor
from tool.replay_recording import _load_realsense, load_depth_range, saved_episode_count

# How much of an episode's tail the live viewer leaves unplayed. Episodes are
# packed end to end in a shared video file, so the frames immediately after one
# are the next recording's, and three decoders kept in step by seeking can stray
# across that seam for a frame at a time -- which reads as the end of a recording
# flickering against the start of the next one. Stopping a little short keeps
# every displayed frame unambiguously inside the episode. This trims the VIEW
# only: the rendered mp4 and the download stay complete, so nothing is lost from
# the record itself, only from what autoplay runs through.
_VIEW_END_MARGIN_S = 0.1


# ── Dataset discovery + rendering (blocking; run in a thread executor) ─────────


def list_datasets(root: Path) -> "list[dict]":
    """Every immediate sub-directory of ``root``, with what it holds.

    The episode count shown is what SURVIVES: episodes already marked for
    deletion are excluded, so the list agrees with the recording pane beside it.

    Unlike the standalone browser this once was, a directory with no saved
    episodes is still listed (flagged ``stillborn``): a session that quit before
    recording leaves one behind, and being able to see and delete it here is the
    point of managing the drive from the console. The console's own working
    directories are the exception -- they are debris, not datasets.
    """
    from common.recording.dataset_read import camera_label

    out: list[dict] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or is_working_dir(child.name):
            continue
        n = saved_episode_count(child)
        pending = len([i for i in read_soft_deleted(child) if i < n])
        meta = read_dataset_meta(child)
        out.append(
            {
                "name": child.name,
                "episodes": n - pending,
                "pending": pending,
                "saved": n,
                "fps": meta["fps"],
                "robot_type": meta["robot_type"],
                "streams": [camera_label(k) for k in meta["streams"]],
                "tasks": meta["tasks"],
                "bytes": directory_size(child),
                "stillborn": n == 0,
                "damaged": meta["fps"] is None,
            }
        )
    return out


def dataset_root(root: Path, name: str) -> Path:
    """Resolve + validate a dataset directory under ``root`` (no traversal)."""
    if "/" in name or name in ("", ".", ".."):
        raise web.HTTPBadRequest(text="bad dataset name")
    path = root / name
    if not path.is_dir() or saved_episode_count(path) == 0:
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    return path


def episodes_of(root: Path, name: str) -> dict:
    """The visible recordings plus how many are marked for deletion.

    Marked episodes are hidden but NOT renumbered: they still occupy their
    on-disk index until compaction, and the video route needs that index.

    Reads the episode metadata directly (see ``read_episode_lengths``) so that a
    damaged dataset still lists: an episode whose length could not be read is
    listed with an unknown length rather than taking the whole pane down.
    """
    path = dataset_root(root, name)
    total = saved_episode_total(path)
    lengths, bad = read_episode_lengths(path)
    if not total:
        total = (max(lengths) + 1) if lengths else 0
    marked = [i for i in read_soft_deleted(path) if 0 <= i < total]
    visible = surviving_indices(total, marked)
    # An episode with no length is either a metadata file that would not read
    # (``damaged``) or one the dataset counts and never wrote. The two look
    # identical in the list and need opposite remedies, so the pane is told
    # which. This costs nothing: the episodes without a row are what the lengths
    # already read say they are. Whether such an episode is truly empty or has
    # frames nobody indexed is settled by the repair route, which reads the data.
    missing = [i for i in range(total) if i not in lengths]
    return {
        "episodes": [{"index": k, "length": lengths.get(k)} for k in visible],
        "pending": len(marked),
        "total": total,
        "damaged": bad,
        "integrity": {
            "ok": not missing,
            "repairable": bool(missing),
            "summary": (
                ""
                if not missing
                else f"episode(s) {', '.join(str(i) for i in missing[:6])}"
                + ("" if len(missing) <= 6 else f" (+{len(missing) - 6} more)")
                + " are counted by the dataset but have no recording behind them"
            ),
            "phantom": missing,
        },
    }


def _cache_path(cache_dir: Path, root: Path, name: str, episode: int) -> Path:
    """mp4 cache path keyed by the dataset's info.json mtime (any edit busts it).

    The collection directory can be changed while the console runs, so the drive
    is part of the key too: two drives may each hold a ``cube-pnp``, and neither
    may ever be shown the other's rendered episode.
    """
    info = dataset_root(root, name) / "meta" / "info.json"
    stamp = int(info.stat().st_mtime) if info.is_file() else 0
    drive = hashlib.md5(str(Path(root).resolve()).encode()).hexdigest()[:8]
    return cache_dir / drive / name / f"ep_{episode:06d}_{stamp}.mp4"


def _dataset_fps(path: Path) -> int:
    """The dataset's recorded frame rate, from its own declaration."""
    import json

    try:
        with open(path / "meta" / "info.json", "r") as f:
            return int(json.load(f).get("fps", 30))
    except (OSError, ValueError, TypeError, AttributeError):
        return 30


def render_episode_mp4(
    root: Path, name: str, episode: int, out_path: Path, fps_override: "int | None"
) -> Path:
    """Render one episode's composited sensor view to ``out_path`` (cached).

    Reads the episode directly from the files that hold it -- one sequential
    decode pass per camera plus the two joint columns out of the data parquet --
    rather than addressing frames one at a time through the dataset API. The
    layout it composites is unchanged; only the cost is. Measured on a
    fourteen-second, three-camera episode: about five seconds all told, against
    roughly thirty before, of which twenty-four were decoding alone.
    """
    import imageio.v2 as imageio  # type: ignore[import]
    from cv2 import COLOR_BGR2RGB, cvtColor  # type: ignore[import]

    from common.recording.dataset_read import (
        camera_label,
        decode_episode_frames,
        read_episode_joints,
        read_episode_row,
        video_keys,
    )
    from common.sensor_view import ViewPanel, compose_sensor_view_frame
    from tool.replay_recording import state12_to_side_dicts
    from tool.test_sensor_rates import _camera_short_label

    if out_path.is_file():
        return out_path
    path = dataset_root(root, name)
    row = read_episode_row(path, episode)
    if row is None:
        raise web.HTTPConflict(
            text=(
                f"episode {episode} of '{name}' has no readable metadata yet — it "
                "is still being recorded, or the session that recorded it was "
                "interrupted before this episode was committed"
            )
        )
    keys = video_keys(path)
    fps = _dataset_fps(path)
    state, action = read_episode_joints(path, row)
    # One sequential pass per camera, the three in parallel: decoding is the
    # dominant cost and the passes are independent.
    with ThreadPoolExecutor(max_workers=max(len(keys), 1)) as pool:
        streams = list(
            pool.map(lambda k: decode_episode_frames(path, row, k, fps), keys)
        )
    # The episode's own declared length is the number of frames it has; the
    # others are a floor under it in case a stream came up short. This used to
    # rely on the joint columns being shorter than the decoded streams, which was
    # true only because the decode was picking up one frame too many.
    declared = int(row.get("length") or 0)
    lengths = [len(state), len(action)] + [len(s) for s in streams]
    n = min(lengths + ([declared] if declared else []))
    if n == 0:
        raise web.HTTPNotFound(text=f"episode {episode} has no frames")

    depth_name, depth_scale = _load_realsense(path)
    depth_range = (
        None
        if depth_name is None
        else load_depth_range(path, depth_name, episode, depth_scale)
    )
    frames = []
    for i in range(n):
        panels = [
            ViewPanel(
                label=_camera_short_label(camera_label(key)),
                image_bgr=streams[c][i],
                fallback_hw=(streams[c][i].shape[0], streams[c][i].shape[1]),
                line1=_camera_short_label(camera_label(key)),
            )
            for c, key in enumerate(keys)
        ]
        if depth_name is not None:
            panels.append(
                _depth_panel(path, depth_name, depth_scale, episode, i, depth_range)
            )
        frame = compose_sensor_view_frame(
            panels,
            state12_to_side_dicts(state[i]),
            state12_to_side_dicts(action[i]),
            "cmd",
            joint_strip=None,
        )
        frames.append(cvtColor(frame, COLOR_BGR2RGB))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.mp4")
    # ultrafast: this is a review proxy, not an archive — the frames it shows are
    # already stored losslessly enough in the dataset's own videos, so spending
    # encoder time on a smaller file only makes the reviewer wait. The composed
    # layout is 1280x700, which the default macro-block size does not divide, so
    # it used to be silently stretched to 704; 4 divides both and keeps the
    # rendered geometry exactly as composed.
    imageio.mimsave(
        tmp,
        frames,
        fps=fps_override or _dataset_fps(path),
        codec="libx264",
        macro_block_size=4,
        output_params=["-preset", "ultrafast", "-crf", "28"],
    )
    os.replace(tmp, out_path)
    return out_path


def _depth_panel(
    path: Path, depth_name: str, depth_scale: float, episode: int, i: int, depth_range
):
    """One depth tile, read from its per-frame image (depth is not a video)."""
    import cv2  # type: ignore[import]

    from common.sensor_view import ViewPanel, colourise_depth
    from tool.replay_recording import depth_png_path
    from tool.test_sensor_rates import _camera_short_label

    dp = depth_png_path(path, depth_name, episode, i)
    depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if dp.is_file() else None
    if depth is None:
        depth_bgr = None
    elif depth_range is not None:
        depth_bgr = colourise_depth(depth, depth_scale, depth_range[0], depth_range[1])
    else:
        depth_bgr = colourise_depth(depth, depth_scale)
    return ViewPanel(
        label=_camera_short_label(depth_name),
        image_bgr=depth_bgr,
        fallback_hw=(480, 640),
        line1=_camera_short_label(depth_name),
    )


# ── Background pre-rendering (opt-in) ─────────────────────────────────────────
#
# Warming the composited videos ahead of the click made sense while every click
# needed one. It no longer does: the view plays the recorded files directly, so
# the composited video is a fallback, and rendering a whole session of them costs
# real CPU for something most reviews never open. Worse, a review often happens
# while the next session is being collected on the same machine, where that CPU
# is the encoder's and the cameras'. So this is off unless asked for, which is
# worth doing when the composited view IS the primary one -- a dataset recording
# depth, or a browser without AV1.
#
# When it does run, it runs newest first (the end of a session is what an operator
# reviews) on a SEPARATE single-thread executor. Separate matters: sharing the
# request executor would let a queue of renders block listing and deleting.


def _prerender_one(cache_dir: Path, root: Path, name: str, episode: int) -> None:
    """Render one episode into the cache, ignoring failures (it is a prefetch)."""
    try:
        out = _cache_path(cache_dir, root, name, episode)
        if out.is_file():
            return
        render_episode_mp4(root, name, episode, out, None)
    except Exception:
        # A prefetch must never take the server down or spam the log; the
        # interactive request will surface any real problem.
        pass


def _queue_prerender(app: web.Application, name: str, episodes: "list[int]") -> None:
    """Schedule cache warming for ``episodes``, replacing any earlier queue."""
    if not app["prerender"]:
        return
    for task in app["prerender_tasks"]:
        task.cancel()
    app["prerender_tasks"] = []
    loop = asyncio.get_running_loop()

    async def run() -> None:
        for episode in reversed(episodes):  # newest first: the likeliest click
            await loop.run_in_executor(
                app["prerender_executor"],
                partial(_prerender_one, app["cache_dir"], app["root"], name, episode),
            )

    app["prerender_tasks"] = [loop.create_task(run())]


# ── HTTP handlers ────────────────────────────────────────────────────────────


async def handle_datasets(request: web.Request) -> web.Response:
    app = request.app
    data = await in_executor(app, list_datasets, app["root"])
    return web.json_response(data)


async def handle_episodes(request: web.Request) -> web.Response:
    app = request.app
    name = request.match_info["name"]
    data = await in_executor(app, episodes_of, app["root"], name)
    # Selecting a dataset is the moment we learn which videos the operator is
    # about to click, so start building them now instead of at the first click.
    _queue_prerender(app, name, [e["index"] for e in data["episodes"]])
    return web.json_response(data)


async def handle_video(request: web.Request) -> web.StreamResponse:
    app = request.app
    name = request.match_info["name"]
    episode = int(request.match_info["episode"])
    out = await in_executor(
        app, _cache_path, app["cache_dir"], app["root"], name, episode
    )
    await in_executor(
        app, render_episode_mp4, app["root"], name, episode, out, app["fps"]
    )
    return web.FileResponse(out)


def episode_playback(root: Path, name: str, episode: int) -> dict:
    """Everything the browser needs to play one episode without a render.

    The recorded videos are already the frames a reviewer wants to see, and the
    browser can decode them itself, so the fastest possible answer is to hand
    over where they are rather than to build a new video out of them. Each entry
    names the stream, the file to fetch, and the window inside that file this
    episode occupies -- several episodes share one video file, so the window is
    what turns a shared file into one recording. That window is reported
    INCLUSIVE of its last frame, unlike the metadata it comes from, whose end is
    the next recording's first frame -- a player that stops on the recorded end
    shows a frame belonging to the following episode. The joint columns come
    along as numbers for the page to draw beside the video, rounded to the
    precision the live view displayed them at, together with the motion
    derivatives (see ``motion_payload``) the dataset does not store.
    """
    from common.recording.dataset_read import (
        camera_label,
        playable_window,
        read_episode_joints,
        read_episode_row,
        video_keys,
    )

    path = dataset_root(root, name)
    row = read_episode_row(path, episode)
    if row is None:
        raise web.HTTPConflict(
            text=(
                f"episode {episode} of '{name}' has no readable metadata yet — it "
                "is still being recorded, or the session that recorded it was "
                "interrupted before this episode was committed"
            )
        )
    fps = _dataset_fps(path)
    streams = []
    for key in video_keys(path):
        # The INCLUSIVE window: "to" here is the last frame this episode owns,
        # deliberately one frame short of the metadata field of the same name,
        # whose end is exclusive and belongs to the next recording. Sending it
        # this way keeps the frame-period arithmetic out of the browser. The
        # tail margin comes off here too, so the browser needs to know nothing
        # about why (see _VIEW_END_MARGIN_S).
        start, end = playable_window(row, key, fps)
        end = max(end - _VIEW_END_MARGIN_S, start)
        streams.append(
            {
                "key": key,
                "label": camera_label(key),
                "url": f"/api/datasets/{name}/episodes/{episode}/video/{key}",
                "from": start,
                "to": end,
            }
        )
    state, action = read_episode_joints(path, row)
    depth_name, _ = _load_realsense(path)
    return {
        "episode": episode,
        "fps": fps,
        "streams": streams,
        # Velocity and acceleration are not recorded: the dataset stores joint
        # positions, and whether a demonstration was smooth is a property of
        # their derivatives. They are computed here, once, alongside the
        # end-effector motion the recording has no stream for at all.
        **motion_payload(state, fps),
        # Depth is stored as per-frame images rather than a playable stream, so a
        # dataset carrying it cannot be shown this way and falls back to the
        # server-composited view.
        "has_depth": depth_name is not None,
        "state": [[round(v, 1) for v in frame] for frame in state.tolist()],
        "action": [[round(v, 1) for v in frame] for frame in action.tolist()],
    }


def _round_rows(array, digits: int) -> "list[list[float]]":
    """``(N, C)`` array → nested lists, rounded, with no non-finite values.

    Rounding is what keeps the payload small; the finiteness pass is what keeps
    it *valid*, because Python's JSON encoder writes a bare ``NaN`` that the
    browser's parser rejects — one bad sample would otherwise take down the
    whole view rather than one number.
    """
    import numpy as np

    clean = np.nan_to_num(
        np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    return [[round(v, digits) for v in row] for row in clean.tolist()]


def _round_series(array, digits: int) -> "list[float]":
    """One ``(N,)`` series, rounded and finite (see ``_round_rows``)."""
    import numpy as np

    return [row[0] for row in _round_rows(np.asarray(array).reshape(-1, 1), digits)]


def motion_payload(state, fps: float) -> dict:
    """Joint and end-effector rates for one episode, in the units it displays.

    The peaks travel with the series because the page scales each plot to its
    own signal: a wrist joint and a shoulder joint do not share a range, and one
    shared axis would flatten every wrist plot into a line.
    """
    from common.recording.episode_motion import JOINT_UNITS, episode_motion

    motion = episode_motion(state, fps)
    payload: dict = {
        "joint_vel": _round_rows(motion["joint_vel"], 2),
        "joint_acc": _round_rows(motion["joint_acc"], 0),
        "joint_units": list(JOINT_UNITS),
        "ee": None,
        "ee_error": motion["ee_error"],
    }
    peaks: dict = {
        "joint_vel": [round(v, 2) for v in _peaks(motion["joint_vel"])],
        "joint_acc": [round(v) for v in _peaks(motion["joint_acc"])],
        "ee": None,
    }
    if motion["ee"] is not None:
        digits = {"v": 3, "a": 2, "w": 1, "alpha": 0}
        payload["ee"] = {
            side: {k: _round_series(series[k], d) for k, d in digits.items()}
            for side, series in motion["ee"].items()
        }
        peaks["ee"] = {
            side: {
                k: round(float(max(series[k], default=0.0)), digits[k]) for k in digits
            }
            for side, series in payload["ee"].items()
        }
    payload["peaks"] = peaks
    return payload


def _peaks(array) -> "list[float]":
    """Largest absolute value of each column, or zeros for an empty episode."""
    import numpy as np

    clean = np.nan_to_num(
        np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    if clean.size == 0:
        return [0.0] * clean.shape[1]
    return [float(v) for v in np.abs(clean).max(axis=0)]


def episode_video_file(root: Path, name: str, episode: int, key: str) -> Path:
    """The recorded video file holding ``episode``'s frames for one camera."""
    from common.recording.dataset_read import (
        episode_video_path,
        read_episode_row,
        video_keys,
    )

    path = dataset_root(root, name)
    if key not in video_keys(path):
        raise web.HTTPNotFound(text=f"'{name}' has no camera stream '{key}'")
    row = read_episode_row(path, episode)
    if row is None:
        raise web.HTTPConflict(
            text=f"episode {episode} of '{name}' is not readable yet"
        )
    video = episode_video_path(path, row, key)
    if not video.is_file():
        raise web.HTTPNotFound(text=f"{video.name} is missing from the dataset")
    return video


async def handle_playback(request: web.Request) -> web.Response:
    app = request.app
    data = await in_executor(
        app,
        episode_playback,
        app["root"],
        request.match_info["name"],
        int(request.match_info["episode"]),
    )
    return web.json_response(data)


async def handle_stream(request: web.Request) -> web.StreamResponse:
    """Serve a recorded video file as it is. No decode, no encode, no cache.

    ``FileResponse`` answers range requests, and the recorder writes these files
    with their index at the front, so the browser can seek straight to the
    episode's window instead of pulling the whole file first.
    """
    app = request.app
    video = await in_executor(
        app,
        episode_video_file,
        app["root"],
        request.match_info["name"],
        int(request.match_info["episode"]),
        request.match_info["key"],
    )
    return web.FileResponse(video)


def _mark_deleted(root: Path, name: str, indices: "list[int]") -> dict:
    """Add ``indices`` to the dataset's delete marker. Cheap: one small write."""
    path = dataset_root(root, name)
    marked = sorted(set(read_soft_deleted(path)) | set(indices))
    try:
        write_soft_deleted(path, marked)
    except OSError as exc:
        raise ReadOnlyDatasetError(
            f"cannot mark episodes: {writability_problem(path) or exc}"
        )
    return {"pending": len(marked)}


async def handle_delete(request: web.Request) -> web.Response:
    """Mark episodes deleted. Returns immediately; the rewrite waits for compact."""
    app = request.app
    name = request.match_info["name"]
    body = await request.json()
    indices = [int(i) for i in body.get("episodes", [])]
    if not indices:
        raise web.HTTPBadRequest(text="no episodes given")
    try:
        result = await in_executor(app, _mark_deleted, app["root"], name, indices)
    except ReadOnlyDatasetError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response(result)


async def handle_restore(request: web.Request) -> web.Response:
    """Un-mark every episode marked for deletion (nothing has been removed yet)."""
    app = request.app
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    try:
        await in_executor(app, write_soft_deleted, path, [])
    except OSError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"pending": 0})


async def handle_compact(request: web.Request) -> web.Response:
    """Really remove the marked episodes -- a JOB: it rewrites the whole dataset.

    Every episode of every kept video file is re-encoded and renumbered, which
    takes minutes on a session-sized dataset and much longer on a merged one, so
    this returns a job for the dock rather than holding the request open. A
    failure is reported there too, instead of as a request that dies after half
    an hour.
    """
    app = request.app
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    refuse_while_busy(app)
    marked = await in_executor(app, read_soft_deleted, path)
    if not marked:
        raise web.HTTPBadRequest(text="no episodes are marked for deletion")
    depth_name, _ = await in_executor(app, _load_realsense, path)
    depth_names = [depth_name] if depth_name else []
    problem = await in_executor(app, writability_problem, Path(app["root"]))
    if problem:
        raise web.HTTPConflict(text=f"cannot rewrite the dataset: {problem}")

    job = new_job(
        app,
        "compact",
        name,
        episodes=len(marked),
        message=f"removing {len(marked)} recording(s) from {name}",
    )

    def run() -> None:
        try:
            total = compact_dataset(path, name, depth_names)
        except BaseException as exc:  # noqa: B036 - reported, not swallowed
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        finish(job, "done", f"{name} now holds {total} recording(s)")

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


async def handle_repair(request: web.Request) -> web.Response:
    """Drop episodes the dataset counts but never wrote -- a JOB, like compaction.

    It rewrites every episode row and every data file (an index column each) and
    renames the side files, which is fast next to a re-encode but far too slow to
    hold a request open on a session-sized dataset.
    """
    app = request.app
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    refuse_while_busy(app)
    report = await in_executor(app, dataset_integrity, path)
    if report["ok"]:
        raise web.HTTPBadRequest(text=f"{name} has nothing to repair")
    if not report["repairable"]:
        raise web.HTTPBadRequest(text=report["summary"])
    problem = await in_executor(app, writability_problem, Path(app["root"]))
    if problem:
        raise web.HTTPConflict(text=f"cannot rewrite the dataset: {problem}")
    depth_name, _ = await in_executor(app, _load_realsense, path)
    depth_names = [depth_name] if depth_name else []

    dropped = len(report["phantom"])
    job = new_job(
        app,
        "repair",
        name,
        episodes=dropped,
        message=f"dropping {dropped} empty episode slot(s) from {name}",
    )

    def run() -> None:
        try:
            result = repair_phantom_episodes(path, depth_names)
        except (DatasetDamaged, OSError, ValueError) as exc:
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        finish(
            job,
            "done",
            f"{name} now holds {result['episodes']} recording(s); "
            f"{result['renumbered']} were renumbered",
        )

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


def add_dataset_routes(app: web.Application) -> None:
    """Register the browse / playback / episode-curation routes."""
    app.add_routes(
        [
            web.get("/api/datasets", handle_datasets),
            web.get("/api/datasets/{name}/episodes", handle_episodes),
            web.get("/api/datasets/{name}/episodes/{episode}.mp4", handle_video),
            web.get(
                "/api/datasets/{name}/episodes/{episode}/playback", handle_playback
            ),
            web.get(
                "/api/datasets/{name}/episodes/{episode}/video/{key}", handle_stream
            ),
            web.post("/api/datasets/{name}/delete", handle_delete),
            web.post("/api/datasets/{name}/restore", handle_restore),
            web.post("/api/datasets/{name}/compact", handle_compact),
            web.post("/api/datasets/{name}/repair", handle_repair),
        ]
    )
