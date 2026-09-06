"""One browser console for the rig: browse datasets, collect, bind signals.

Serves a small single-page app on ``127.0.0.1:<port>`` with four tabs.

DATASETS is the episode browser that used to live in ``tool/dataset_web.py``:
the datasets under ``--dir``, the recordings inside the selected one, and a
sensor view that plays the recorded camera files side by side on one clock.
Whole datasets can also be created, renamed, merged and deleted here, so the
collection drive is managed from the same place it is reviewed.

COLLECT starts and watches one collection session; SIGNALS binds the rig's
devices to the stream names the recorder writes; TRAINING sends a collected
dataset to a GPU machine and watches what it does there.

Deleting is always available -- see ``common.web.datasets_api`` for what marking
and compaction mean -- and every irreversible one asks first in the browser.

Usage:

    venv/bin/python tool/rig_web.py --dir /media/hdd/so101
    venv/bin/python tool/rig_web.py --dir /media/hdd/so101 --port 8000

Then open http://127.0.0.1:8000/ in a browser.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp  # type: ignore[import]
from aiohttp import web  # type: ignore[import]

# The console only ever reads datasets from the local drive. Without this, a
# dataset whose metadata is momentarily incomplete makes LeRobot fall back to a
# Hub lookup on the bare dataset name and report an offline-mode or 401 failure,
# which says nothing about the real problem. Set before any LeRobot import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# OpenCV logs a WARNING for every capture node it opens by name ("backend is
# generally available but can't be used to capture by name"), which on a rig with
# seven cameras and their metadata nodes buries the console's own output at every
# device scan. The message says nothing an operator can act on -- the open either
# works or is reported by us. Must be set before cv2 is first imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from common.web.datasets_api import add_dataset_routes
from common.web.jobs import add_job_routes
from common.web.lifecycle_api import add_lifecycle_routes
from common.web.projects_api import add_project_routes
from common.web.roots_api import (
    add_root_routes,
    initial_root,
    root_required,
    unmount_own,
)
from common.web.sensors_api import add_sensor_routes
from common.web.session import PreviewArms, PreviewCameras, SessionSupervisor
from common.web.session_api import add_session_routes
from common.web.training_api import add_training_routes
from common.web.util import preinit_tqdm_lock, revalidate_assets

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "common" / "web" / "static"


async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


async def _open_client(app: web.Application) -> None:
    """One client session for talking to a running collection session's monitor."""
    app["http"] = aiohttp.ClientSession()


async def _release_devices(app: web.Application) -> None:
    """Let go of every camera and bus, first thing in the shutdown.

    This runs on ``on_shutdown``, before aiohttp closes its connections, rather
    than in the cleanup that follows: a capture thread sits inside a blocking
    V4L2 read, so releasing early is what lets the in-flight frame requests
    finish promptly instead of being torn down mid-handler -- which is where
    Ctrl+C used to produce a wall of aiohttp InvalidStateError tracebacks.

    A collection session is left alone: it is a separate process with the
    dataset open, and closing the console must not end it, or an operator would
    lose a recording by restarting a web page. It is stopped from the Collect
    tab, or with its own quit key.
    """
    app["preview"].stop()
    app["arms"].stop()
    app["sensors_probe"].close()


async def _close_session(app: web.Application) -> None:
    """Close the client session and release a directory the console mounted.

    A remote directory the console mounted IS released: it exists only for this
    console, and leaving it behind would strand a dead FUSE mount.
    """
    await app["http"].close()
    unmount_own(app)


async def handle_console(request: web.Request) -> web.Response:
    """What the front-end needs to describe itself. The drive may not be chosen."""
    root = request.app["root"]
    return web.json_response({"root": None if root is None else str(root)})


def build_app(args: argparse.Namespace) -> web.Application:
    preinit_tqdm_lock()
    app = web.Application(
        client_max_size=1024, middlewares=[revalidate_assets, root_required]
    )
    outputs = Path(os.environ.get("SO101_OUTPUT_DIR", "outputs")).expanduser()
    app["roots_file"] = outputs / "rig_web_roots.json"
    app["mount_dir"] = Path(
        getattr(args, "mount_dir", None) or outputs / "rig_web_mounts"
    ).expanduser()
    # The directory can be chosen (and changed) in the page, so it may be unset
    # at start-up: the dataset routes refuse until one is picked.
    app["root"] = initial_root(args.dir, app["roots_file"])
    app["fps"] = args.fps
    app["executor"] = ThreadPoolExecutor(max_workers=2)
    # Cache warming runs on its own single thread so it can never delay a click.
    app["prerender"] = bool(getattr(args, "prerender", False))
    app["prerender_executor"] = ThreadPoolExecutor(max_workers=1)
    app["prerender_tasks"] = []
    # Whole-dataset work (a merge, or freeing a deleted dataset's bytes) runs
    # one at a time on its own thread, so it can never queue behind a click or
    # start twice.
    app["job_executor"] = ThreadPoolExecutor(max_workers=1)
    app["jobs"] = {}
    # The collection session (a subprocess) and the console's own idle previews
    # of the cameras and the follower arms. Only the session or the previews
    # ever hold a device, never both.
    # The supervisor's root follows the console's; with none chosen yet the
    # placeholder is never used, because starting a session needs one.
    app["session"] = SessionSupervisor(
        app["root"] or Path.cwd(), monitor_port=getattr(args, "monitor_port", 8766)
    )
    app["preview"] = PreviewCameras()
    app["arms"] = PreviewArms()
    app.on_startup.append(_open_client)
    app.on_shutdown.append(_release_devices)
    app.on_cleanup.append(_close_session)
    app["cache_dir"] = outputs / "rig_web_cache"
    # What this console has launched on a training machine. The runs themselves
    # live on that machine; this is only the list of them, so a corrupt or
    # missing file costs the page its history and nothing else.
    app["training_file"] = outputs / "training_runs.yaml"
    # Which runs the operator considers one experiment. Separate from the file
    # above because a project groups DISCOVERED runs -- nineteen of the run
    # directories on these machines were started from a terminal and have no
    # launch record to hang a field on.
    app["projects_file"] = outputs / "training_projects.yaml"
    app.add_routes(
        [
            web.get("/", handle_index),
            web.get("/api/console", handle_console),
            web.static("/static", STATIC_DIR),
        ]
    )
    add_root_routes(app)
    add_job_routes(app)
    add_dataset_routes(app)
    add_lifecycle_routes(app)
    add_session_routes(app)
    add_sensor_routes(app)
    add_training_routes(app)
    add_project_routes(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Collection directory holding the datasets. Optional: without it "
        "the console opens on the last one it was pointed at, and it can be "
        "changed (or mounted over SSH) from the page at any time",
    )
    parser.add_argument("--port", type=int, default=8000, help="Localhost port")
    parser.add_argument(
        "--fps", type=int, default=None, help="Playback fps override (default dataset)"
    )
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=8766,
        help="Loopback port the collection session serves its live view on; "
        "the console proxies it (change it only if the port is taken)",
    )
    parser.add_argument(
        "--mount-dir",
        default=None,
        help="Where remote collection directories are mounted "
        "(default $SO101_OUTPUT_DIR/rig_web_mounts)",
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
    where = app["root"] or "no directory yet — choose one in the page"
    print(f"🖥️  rig console on http://127.0.0.1:{args.port}/  (dir: {where})")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
