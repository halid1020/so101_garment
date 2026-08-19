"""Localhost web browser for collected LeRobot datasets (aiohttp).

Serves a small single-page app on ``127.0.0.1:<port>`` with three panes: the
DATASET list (every dataset under ``--dir`` with at least one saved episode),
the RECORDING list for the selected dataset (with checkboxes for batch delete
and a per-row delete), and the sensor VIEW of the selected recording.

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

Deletion is OFF unless ``--allow-delete`` is passed (a delete request returns 403
otherwise), and it happens in two steps, because really removing one episode from
a v3.0 dataset re-encodes and renumbers the whole thing and takes far too long to
sit behind a click. Deleting MARKS the episode: the row disappears at once and
the decision is recorded in the dataset, reversibly. "Remove for good" then
compacts: one rewrite for the whole batch, via
``common.recording.dataset_edit.compact_dataset``.

Until a dataset is compacted its marked episodes are still on disk, so anything
that trains on the dataset would still see them. The pane says so, and the
real-VLA training scripts refuse to start while marks are pending.

Usage:

    venv/bin/python tool/dataset_web.py --dir /media/hdd/so101
    venv/bin/python tool/dataset_web.py --dir /media/hdd/so101 --port 8000 --allow-delete

Then open http://127.0.0.1:8000/ in a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from aiohttp import web  # type: ignore[import]

# This tool only ever reads datasets from the local drive. Without this, a
# dataset whose metadata is momentarily incomplete makes LeRobot fall back to a
# Hub lookup on the bare dataset name and report an offline-mode or 401 failure,
# which says nothing about the real problem. Set before any LeRobot import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
from tool.replay_recording import _load_realsense, load_depth_range, saved_episode_count

_IMAGE_PREFIX = "observation.images."


# ── Dataset discovery + rendering (blocking; run in a thread executor) ─────────


def list_datasets(root: Path) -> "list[dict]":
    """Every immediate sub-directory of ``root`` that is a non-empty dataset.

    The count shown is what SURVIVES: episodes already marked for deletion are
    excluded, so the list agrees with the recording pane beside it.
    """
    out: list[dict] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        n = saved_episode_count(child)
        if n > 0:
            pending = len([i for i in read_soft_deleted(child) if i < n])
            out.append(
                {"name": child.name, "episodes": n - pending, "pending": pending}
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
    return {
        "episodes": [{"index": k, "length": lengths.get(k)} for k in visible],
        "pending": len(marked),
        "total": total,
        "damaged": bad,
    }


def _cache_path(cache_dir: Path, root: Path, name: str, episode: int) -> Path:
    """mp4 cache path keyed by the dataset's info.json mtime (any edit busts it)."""
    info = dataset_root(root, name) / "meta" / "info.json"
    stamp = int(info.stat().st_mtime) if info.is_file() else 0
    return cache_dir / name / f"ep_{episode:06d}_{stamp}.mp4"


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


# ── HTTP handlers ─────────────────────────────────────────────────────────────


async def _in_executor(app: web.Application, fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app["executor"], partial(fn, *args))


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=_INDEX_HTML, content_type="text/html")


async def handle_datasets(request: web.Request) -> web.Response:
    app = request.app
    data = await _in_executor(app, list_datasets, app["root"])
    return web.json_response(data)


async def handle_episodes(request: web.Request) -> web.Response:
    app = request.app
    name = request.match_info["name"]
    data = await _in_executor(app, episodes_of, app["root"], name)
    # Selecting a dataset is the moment we learn which videos the operator is
    # about to click, so start building them now instead of at the first click.
    _queue_prerender(app, name, [e["index"] for e in data["episodes"]])
    return web.json_response(data)


async def handle_video(request: web.Request) -> web.StreamResponse:
    app = request.app
    name = request.match_info["name"]
    episode = int(request.match_info["episode"])
    out = await _in_executor(
        app, _cache_path, app["cache_dir"], app["root"], name, episode
    )
    await _in_executor(
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
    precision the live view displayed them at.
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
        # this way keeps the frame-period arithmetic out of the browser.
        start, end = playable_window(row, key, fps)
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
        # Depth is stored as per-frame images rather than a playable stream, so a
        # dataset carrying it cannot be shown this way and falls back to the
        # server-composited view.
        "has_depth": depth_name is not None,
        "state": [[round(v, 1) for v in frame] for frame in state.tolist()],
        "action": [[round(v, 1) for v in frame] for frame in action.tolist()],
    }


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
    data = await _in_executor(
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
    video = await _in_executor(
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
    if not app["allow_delete"]:
        raise web.HTTPForbidden(text="deletion disabled; restart with --allow-delete")
    name = request.match_info["name"]
    body = await request.json()
    indices = [int(i) for i in body.get("episodes", [])]
    if not indices:
        raise web.HTTPBadRequest(text="no episodes given")
    try:
        result = await _in_executor(app, _mark_deleted, app["root"], name, indices)
    except ReadOnlyDatasetError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response(result)


async def handle_restore(request: web.Request) -> web.Response:
    """Un-mark every episode marked for deletion (nothing has been removed yet)."""
    app = request.app
    if not app["allow_delete"]:
        raise web.HTTPForbidden(text="deletion disabled; restart with --allow-delete")
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    try:
        await _in_executor(app, write_soft_deleted, path, [])
    except OSError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"pending": 0})


async def handle_compact(request: web.Request) -> web.Response:
    """Really remove the marked episodes. Slow: rewrites and renumbers the dataset."""
    app = request.app
    if not app["allow_delete"]:
        raise web.HTTPForbidden(text="deletion disabled; restart with --allow-delete")
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    depth_name, _ = await _in_executor(app, _load_realsense, path)
    depth_names = [depth_name] if depth_name else []
    try:
        new_total = await _in_executor(app, compact_dataset, path, name, depth_names)
    except ReadOnlyDatasetError as exc:
        raise web.HTTPConflict(text=str(exc))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"episodes": new_total, "pending": 0})


def _preinit_tqdm_lock() -> None:
    """Create tqdm's class lock up front, to stop concurrent loads racing on it.

    LeRobot's dataset loading reaches HuggingFace ``datasets``, whose
    ``thread_map`` wraps work in tqdm's ``ensure_lock``. That helper deletes
    ``tqdm._lock`` again if it was absent when it started, so two loads running
    together in a thread pool both find it absent, both try to delete it, and the
    second raises ``AttributeError: type object 'tqdm' has no attribute '_lock'``
    -- which surfaced as a 500 on an otherwise fine dataset. Creating the lock
    once means it is never absent, so it is never deleted.
    """
    try:
        from tqdm import tqdm

        tqdm.set_lock(tqdm.get_lock())
    except Exception:
        pass


def build_app(args: argparse.Namespace) -> web.Application:
    _preinit_tqdm_lock()
    app = web.Application(client_max_size=1024)
    app["root"] = Path(args.dir).expanduser()
    app["allow_delete"] = bool(args.allow_delete)
    app["fps"] = args.fps
    app["executor"] = ThreadPoolExecutor(max_workers=2)
    # Cache warming runs on its own single thread so it can never delay a click.
    app["prerender"] = bool(getattr(args, "prerender", False))
    app["prerender_executor"] = ThreadPoolExecutor(max_workers=1)
    app["prerender_tasks"] = []
    cache = os.environ.get("SO101_OUTPUT_DIR", "outputs")
    app["cache_dir"] = Path(cache).expanduser() / "dataset_web_cache"
    app.add_routes(
        [
            web.get("/", handle_index),
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
        ]
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir", required=True, help="Collection directory holding the datasets"
    )
    parser.add_argument("--port", type=int, default=8000, help="Localhost port")
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="Enable episode deletion (destructive; renumbers the survivors)",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="Playback fps override (default dataset)"
    )
    parser.add_argument(
        "--prerender",
        action="store_true",
        help="Build the composited episode videos in the background while "
        "browsing. Only worth it when the composited view is the one you "
        "actually watch (a dataset with depth, or a browser without AV1); "
        "playback of the recorded streams needs no rendering at all",
    )
    args = parser.parse_args()

    # Purely local; never reach out to the Hub for an incomplete dataset.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    app = build_app(args)
    print(f"📺 dataset browser on http://127.0.0.1:{args.port}/  (dir: {args.dir})")
    if args.allow_delete:
        print("⚠️  --allow-delete: episode deletion is ENABLED (renumbers survivors)")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 dataset browser</title>
<style>
  :root { color-scheme: light dark; --bd:#8883; --sel:#3b82f6; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui,sans-serif; display:flex; height:100vh; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em; opacity:.6;
       margin:0; padding:10px 12px; border-bottom:1px solid var(--bd); }
  .col { display:flex; flex-direction:column; border-right:1px solid var(--bd);
         overflow:hidden; }
  #datasets { width:220px; } #episodes { width:260px; }
  #viewer { flex:1; padding:12px; overflow:auto; }
  ul { list-style:none; margin:0; padding:0; overflow:auto; flex:1; }
  li { padding:8px 12px; cursor:pointer; border-bottom:1px solid var(--bd);
       display:flex; gap:8px; align-items:center; }
  li:hover { background:#8881; }
  li.sel { background:var(--sel); color:#fff; }
  .grow { flex:1; } .muted { opacity:.6; font-size:12px; }
  .bar { padding:8px 12px; border-bottom:1px solid var(--bd); display:flex;
         gap:8px; align-items:center; }
  button { font:inherit; padding:4px 10px; border:1px solid var(--bd);
           border-radius:6px; background:#8881; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  button.danger { color:#dc2626; border-color:#dc262688; }
  video { width:100%; max-height:calc(100vh - 90px); background:#000; border-radius:8px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
           gap:8px; }
  .tile { margin:0; } .tile video { max-height:38vh; }
  .tile figcaption { font-size:11px; opacity:.6; padding:2px 0; }
  .viewbar { border:0; padding:8px 0; }
  #joints { border-collapse:collapse; font-variant-numeric:tabular-nums;
            font-size:12px; }
  #joints th, #joints td { padding:2px 10px; text-align:right;
                           border-bottom:1px solid var(--bd); }
  #joints th:first-child { text-align:left; opacity:.6; font-weight:400; }
  #scrub { accent-color:var(--sel); }
  .del { margin-left:auto; opacity:.5; } .del:hover { opacity:1; }
  .pending { background:#f59e0b22; font-size:12px; }
  .damaged { background:#dc262622; font-size:12px; }
  .busy { opacity:.6; pointer-events:none; }
</style></head>
<body>
  <div class="col" id="datasets"><h2>Datasets</h2><ul id="ds-list"></ul></div>
  <div class="col" id="episodes">
    <h2 id="ep-title">Recordings</h2>
    <div class="bar">
      <label><input type="checkbox" id="all"> all</label>
      <button id="del-sel" class="danger" disabled>Delete selected</button>
    </div>
    <div class="bar damaged" id="damaged-bar" hidden>
      <span class="grow" id="damaged-text"></span>
    </div>
    <div class="bar pending" id="pending-bar" hidden>
      <span class="grow" id="pending-text"></span>
      <button id="restore">Restore</button>
      <button id="compact">Remove for good</button>
    </div>
    <ul id="ep-list"></ul>
  </div>
  <div id="viewer"><p class="muted">Select a recording to play its sensor view.</p></div>
<script>
const $ = (s) => document.querySelector(s);
let curDataset = null, curEpisode = null, allowDelete = false;

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

async function loadDatasets() {
  const list = await j('/api/datasets');
  const ul = $('#ds-list'); ul.innerHTML = '';
  for (const d of list) {
    const li = document.createElement('li');
    const mark = d.pending ? ` <span class="muted">(${d.pending}✗)</span>` : '';
    li.innerHTML = `<span class="grow">${d.name}</span>
                    <span class="muted">${d.episodes}</span>${mark}`;
    li.onclick = () => selectDataset(d.name, li);
    if (d.name === curDataset) li.classList.add('sel');
    ul.appendChild(li);
  }
  if (!list.length) ul.innerHTML = '<li class="muted">no datasets found</li>';
}

async function selectDataset(name, li) {
  curDataset = name; curEpisode = null;
  document.querySelectorAll('#ds-list li').forEach(x => x.classList.remove('sel'));
  if (li) li.classList.add('sel');
  $('#ep-title').textContent = name;
  await loadEpisodes();
  $('#viewer').innerHTML = '<p class="muted">Select a recording.</p>';
}

function showDamaged(files) {
  const bar = $('#damaged-bar');
  bar.hidden = !(files && files.length);
  if (!bar.hidden) $('#damaged-text').textContent =
    `${files.length} unreadable metadata file(s) — some episode lengths are `
    + `unknown ("?f"). The episodes themselves may still play. (${files.join(', ')})`;
}

function showPending(n) {
  $('#pending-bar').hidden = !n;
  if (n) $('#pending-text').textContent =
    `${n} marked for deletion — still on disk, so NOT yet excluded from training.`;
}

async function loadEpisodes() {
  const data = await j(`/api/datasets/${curDataset}/episodes`);
  const eps = data.episodes;
  showPending(data.pending);
  showDamaged(data.damaged);
  const ul = $('#ep-list'); ul.innerHTML = '';
  for (const e of eps) {
    const li = document.createElement('li');
    li.dataset.idx = e.index;
    const len = (e.length === null || e.length === undefined) ? '?' : e.length;
    li.innerHTML =
      `<input type="checkbox" class="pick" onclick="event.stopPropagation()">
       <span class="grow">episode ${e.index}</span>
       <span class="muted">${len}f</span>` +
      (allowDelete ? `<span class="del" title="delete">🗑</span>` : '');
    li.onclick = () => selectEpisode(e.index, li);
    if (allowDelete) li.querySelector('.del').onclick = (ev) => {
      ev.stopPropagation(); doDelete([e.index]);
    };
    li.querySelector('.pick').onchange = updateSelCount;
    ul.appendChild(li);
  }
  $('#all').checked = false; updateSelCount();
}

function picked() {
  return [...document.querySelectorAll('#ep-list .pick')]
    .filter(c => c.checked).map(c => +c.closest('li').dataset.idx);
}
function updateSelCount() {
  $('#del-sel').disabled = !allowDelete || picked().length === 0;
}

// Playing an episode does not need a rendered video: the recorded camera files
// are already the frames to show, so they are played where they lie and the
// joint columns are drawn beside them from the same numbers the live view
// displayed. That makes the first click as fast as the browser can start a
// video, instead of as slow as compositing one. The recorded mp4 is still
// rendered on demand for a dataset this cannot cover (depth is stored as images,
// not as a playable stream) and for downloading an episode.
const JOINTS = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
let sync = null;  // the running viewer, so a new selection can stop the old one

function selectEpisode(idx, li) {
  curEpisode = idx;
  document.querySelectorAll('#ep-list li').forEach(x => x.classList.remove('sel'));
  if (li) li.classList.add('sel');
  if (sync) { sync.stop(); sync = null; }
  $('#viewer').innerHTML = '<p class="muted">opening…</p>';
  openEpisode(curDataset, idx);
}

function renderedFallback(name, idx, why) {
  $('#viewer').innerHTML =
    `<video controls autoplay muted src="/api/datasets/${name}/episodes/${idx}.mp4"></video>
     <p class="muted">episode ${idx} — ${why} Showing the composited view, which is
     rendered on first play.</p>`;
}

async function openEpisode(name, idx) {
  let info;
  try {
    info = await j(`/api/datasets/${name}/episodes/${idx}/playback`);
  } catch (e) {
    $('#viewer').innerHTML = `<p class="muted">${e.message}</p>`;
    return;
  }
  if (name !== curDataset || idx !== curEpisode) return;  // a newer click won
  if (info.has_depth || !info.streams.length) {
    renderedFallback(name, idx, 'this dataset records a depth stream.');
    return;
  }
  const probe = document.createElement('video');
  if (!probe.canPlayType('video/mp4; codecs="av01.0.05M.08"')) {
    renderedFallback(name, idx, 'this browser cannot decode AV1.');
    return;
  }
  const tiles = info.streams.map((s, i) =>
    `<figure class="tile"><video id="v${i}" muted preload="metadata" src="${s.url}"></video>
     <figcaption>${s.label}</figcaption></figure>`).join('');
  $('#viewer').innerHTML = `
    <div class="tiles">${tiles}</div>
    <div class="bar viewbar">
      <button id="play">▶︎ play</button>
      <input id="scrub" type="range" min="0" max="1000" value="0" class="grow">
      <span class="muted" id="clock">0.00 s</span>
      <a id="dl" class="muted" href="/api/datasets/${name}/episodes/${idx}.mp4"
         download>download mp4</a>
    </div>
    <table id="joints"></table>
    <p class="muted">episode ${idx} — ${info.streams.length} streams playing from the
    recorded files, joint values beside them.</p>`;
  sync = startSync(info);
}

function jointRows(info, frame) {
  const cell = (v) => `<td>${v === undefined ? '--' : v.toFixed(1)}</td>`;
  const state = info.state[frame] || [], action = info.action[frame] || [];
  let html = '<tr><th></th><th colspan="2">left</th><th colspan="2">right</th></tr>' +
             '<tr><th></th><th>state</th><th>cmd</th><th>state</th><th>cmd</th></tr>';
  JOINTS.forEach((jn, k) => {
    html += `<tr><th>${jn}</th>${cell(state[k])}${cell(action[k])}` +
            `${cell(state[k + 6])}${cell(action[k + 6])}</tr>`;
  });
  return html;
}

// One stream is the clock; the others are told where to be. Each is seeked to
// its own window inside its own file, because the streams are cut at their own
// frame boundaries and so do not share a zero.
function startSync(info) {
  const videos = info.streams.map((s, i) => document.querySelector('#v' + i));
  const span = Math.max(info.streams[0].to - info.streams[0].from, 1e-6);
  const master = videos[0], base = info.streams[0].from;
  let stopped = false, playing = false;
  const at = () => Math.min(Math.max(master.currentTime - base, 0), span);

  const seek = (t) => info.streams.forEach((s, i) => {
    const want = s.from + Math.min(t, s.to - s.from);
    if (Math.abs(videos[i].currentTime - want) > 0.04) videos[i].currentTime = want;
  });
  // Land exactly on each stream's last frame, ignoring the tolerance the drift
  // correction uses: that tolerance is wider than a frame period, so it would
  // decline to undo an overshoot of the very size we are here to undo.
  const settle = () => info.streams.forEach((s, i) => {
    videos[i].currentTime = s.to;
  });
  const paint = () => {
    if (stopped) return;
    const t = at();
    document.querySelector('#clock').textContent = t.toFixed(2) + ' s';
    document.querySelector('#scrub').value = Math.round((t / span) * 1000);
    document.querySelector('#joints').innerHTML =
      jointRows(info, Math.min(Math.round(t * info.fps), info.state.length - 1));
    // Stop on this episode's last frame. This has to be driven from the frame
    // loop rather than from the video's own timeupdate event, which browsers
    // throttle to about four times a second: a quarter of a second is seven
    // frames at the dataset rate, so a check driven by it sails well past the
    // end and into the next recording before it fires. Overshoot is then undone
    // rather than merely stopped, so the frame left on screen is this
    // episode's last and not whatever the decoder had reached.
    if (playing && master.currentTime >= info.streams[0].to - 1e-3) {
      pause();
      settle();
    }
    // Drift correction: playback rates differ slightly between streams, so the
    // followers are nudged back whenever they fall more than a frame behind.
    if (playing) {
      info.streams.forEach((s, i) => {
        if (i === 0) return;
        const want = s.from + Math.min(t, s.to - s.from);
        if (Math.abs(videos[i].currentTime - want) > 0.04) videos[i].currentTime = want;
      });
    }
    requestAnimationFrame(paint);
  };
  const play = () => {
    playing = true;
    document.querySelector('#play').textContent = '❚❚ pause';
    videos.forEach(v => v.play().catch(() => {}));
  };
  const pause = () => {
    playing = false;
    document.querySelector('#play').textContent = '▶︎ play';
    videos.forEach(v => v.pause());
  };
  document.querySelector('#play').onclick = () => (playing ? pause() : play());
  document.querySelector('#scrub').oninput = (e) => seek((e.target.value / 1000) * span);
  // Start where the episode starts, not where its file does.
  let ready = 0;
  videos.forEach((v, i) => v.addEventListener('loadedmetadata', () => {
    v.currentTime = info.streams[i].from;
    if (++ready === videos.length) play();
  }, { once: true }));
  requestAnimationFrame(paint);
  return { stop: () => { stopped = true; videos.forEach(v => v.pause()); } };
}

// Marking is cheap, so the rows go immediately and the request follows. No
// confirm dialog: a mark is reversible with Restore until it is compacted.
async function doDelete(indices) {
  if (!allowDelete) return;
  const dead = new Set(indices);
  for (const li of document.querySelectorAll('#ep-list li')) {
    if (dead.has(+li.dataset.idx)) li.remove();
  }
  if (dead.has(curEpisode)) {
    curEpisode = null;
    $('#viewer').innerHTML = '<p class="muted">Deleted. Select a recording.</p>';
  }
  updateSelCount();
  try {
    const r = await j(`/api/datasets/${curDataset}/delete`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({episodes: indices}),
    });
    showPending(r.pending);
    loadDatasets();
  } catch (e) {
    alert('delete failed: ' + e.message);
    await loadEpisodes();  // the optimistic removal was wrong; resync
  }
}

// Compaction is the slow half: it rewrites and renumbers the dataset, so it
// blocks the pane and every visible index changes afterwards.
async function doCompact() {
  if (!confirm(`Permanently remove the marked episode(s) from ${curDataset}?\n\n`
      + `This rewrites the dataset, renumbers the survivors and cannot be `
      + `undone. It may take a while.`)) return;
  const bar = $('#pending-bar');
  bar.classList.add('busy');
  $('#pending-text').textContent = 'Removing… rewriting the dataset, please wait.';
  try {
    await j(`/api/datasets/${curDataset}/compact`, {method: 'POST'});
    curEpisode = null;
    $('#viewer').innerHTML = '<p class="muted">Removed. Select a recording.</p>';
  } catch (e) {
    alert('compact failed: ' + e.message);
  } finally {
    bar.classList.remove('busy');
    await loadDatasets(); await loadEpisodes();
  }
}

async function doRestore() {
  try {
    await j(`/api/datasets/${curDataset}/restore`, {method: 'POST'});
  } catch (e) { alert('restore failed: ' + e.message); }
  await loadDatasets(); await loadEpisodes();
}

$('#all').onchange = (e) => {
  document.querySelectorAll('#ep-list .pick').forEach(c => c.checked = e.target.checked);
  updateSelCount();
};
$('#del-sel').onclick = () => doDelete(picked());
$('#compact').onclick = doCompact;
$('#restore').onclick = doRestore;

// Probe whether deletion is enabled (a disabled server 403s the delete route).
fetch('/api/datasets/__probe__/delete', {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: '{}'})
  .then(r => { allowDelete = (r.status !== 403); loadDatasets(); })
  .catch(() => loadDatasets());
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
