"""Localhost web browser for collected LeRobot datasets (aiohttp).

Serves a small single-page app on ``127.0.0.1:<port>`` with three panes: the
DATASET list (every dataset under ``--dir`` with at least one saved episode),
the RECORDING list for the selected dataset (with checkboxes for batch delete
and a per-row delete), and the sensor-view VIDEO of the selected recording. The
video is the SAME composited layout the operator watched while collecting
(rendered by ``tool/replay_recording.build_frame``) written to an mp4 and shown
in an HTML5 ``<video controls>`` element, so the browser's own draggable scrub
bar seeks through the episode.

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
from functools import partial
from pathlib import Path

from aiohttp import web  # type: ignore[import]

from common.recording.dataset_edit import (
    ReadOnlyDatasetError,
    compact_dataset,
    read_episode_lengths,
    read_soft_deleted,
    saved_episode_total,
    surviving_indices,
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


def render_episode_mp4(
    root: Path, name: str, episode: int, out_path: Path, fps_override: "int | None"
) -> Path:
    """Render one episode's composited sensor view to ``out_path`` (cached)."""
    import imageio.v2 as imageio  # type: ignore[import]
    from cv2 import COLOR_BGR2RGB, cvtColor  # type: ignore[import]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tool.replay_recording import build_frame

    if out_path.is_file():
        return out_path
    path = dataset_root(root, name)
    # pyav backend: recorded videos are AV1 and torchcodec's AV1 seeking
    # mis-lands on frames (FrameTimestampError); pyav decodes them reliably.
    ds = LeRobotDataset(name, root=path, episodes=[episode], video_backend="pyav")
    n = len(ds)
    if n == 0:
        raise web.HTTPNotFound(text=f"episode {episode} has no frames")
    fps = fps_override or int(ds.meta.fps)
    camera_names = [(k, k[len(_IMAGE_PREFIX) :]) for k in ds.meta.camera_keys]
    depth_name, depth_scale = _load_realsense(path)
    depth_range = (
        None
        if depth_name is None
        else load_depth_range(path, depth_name, episode, depth_scale)
    )
    frames = []
    for i in range(n):
        frame = build_frame(
            ds, i, episode, camera_names, depth_name, depth_scale, path, depth_range
        )
        frames.append(cvtColor(frame, COLOR_BGR2RGB))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.mp4")
    imageio.mimsave(tmp, frames, fps=fps)
    os.replace(tmp, out_path)
    return out_path


# ── Background pre-rendering ──────────────────────────────────────────────────
#
# Rendering an episode's composited view takes seconds, and it used to happen on
# the first click. The reviewer's next click is highly predictable, though: it is
# one of the episodes just listed. So the videos are built ahead of time, newest
# first (the end of a session is what an operator reviews), on a SEPARATE
# single-thread executor. Separate matters: sharing the request executor would
# let a queue of renders block listing and deleting.


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


def _mark_deleted(root: Path, name: str, indices: "list[int]") -> dict:
    """Add ``indices`` to the dataset's delete marker. Cheap: one small write."""
    path = dataset_root(root, name)
    marked = sorted(set(read_soft_deleted(path)) | set(indices))
    try:
        write_soft_deleted(path, marked)
    except OSError as exc:
        raise ReadOnlyDatasetError(
            f"cannot mark episodes in {path}: {exc}. Remount the drive "
            "read-write to curate this dataset."
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
    from concurrent.futures import ThreadPoolExecutor

    _preinit_tqdm_lock()
    app = web.Application(client_max_size=1024)
    app["root"] = Path(args.dir).expanduser()
    app["allow_delete"] = bool(args.allow_delete)
    app["fps"] = args.fps
    app["executor"] = ThreadPoolExecutor(max_workers=2)
    # Cache warming runs on its own single thread so it can never delay a click.
    app["prerender"] = not getattr(args, "no_prerender", False)
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
        "--no-prerender",
        action="store_true",
        help="Do not build episode videos in the background (they are then "
        "rendered on the first click, as before)",
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

function selectEpisode(idx, li) {
  curEpisode = idx;
  document.querySelectorAll('#ep-list li').forEach(x => x.classList.remove('sel'));
  if (li) li.classList.add('sel');
  $('#viewer').innerHTML =
    `<video controls autoplay muted src="/api/datasets/${curDataset}/episodes/${idx}.mp4"></video>
     <p class="muted">episode ${idx} — drag the scrub bar to seek. First load renders the mp4.</p>`;
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
