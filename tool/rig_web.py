"""One browser console for the rig: browse datasets, collect, assign sensors.

Serves a small single-page app on ``127.0.0.1:<port>`` with three tabs.

DATASETS is the episode browser that used to live in ``tool/dataset_web.py``:
the datasets under ``--dir``, the recordings inside the selected one, and a
sensor view that plays the recorded camera files side by side on one clock.
Whole datasets can also be created, renamed, merged and deleted here, so the
collection drive is managed from the same place it is reviewed.

COLLECT and SENSORS are placeholders in this change set; live view, session
control and device assignment follow.

Episode deletion (and dataset deletion) is OFF unless ``--allow-delete`` is
passed -- see ``common.web.datasets_api`` for what marking and compaction mean.

Usage:

    venv/bin/python tool/rig_web.py --dir /media/hdd/so101
    venv/bin/python tool/rig_web.py --dir /media/hdd/so101 --port 8000 --allow-delete

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

from common.web.datasets_api import add_dataset_routes
from common.web.lifecycle_api import add_lifecycle_routes
from common.web.sensors_api import add_sensor_routes
from common.web.session import PreviewCameras, SessionSupervisor
from common.web.session_api import add_session_routes
from common.web.util import preinit_tqdm_lock

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "common" / "web" / "static"


async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


async def _open_client(app: web.Application) -> None:
    """One client session for talking to a running collection session's monitor."""
    app["http"] = aiohttp.ClientSession()


async def _close_session(app: web.Application) -> None:
    """Release the cameras the preview holds; leave a collection session alone.

    A session is a separate process with the dataset open: closing the console
    must not end it, or an operator would lose a recording by restarting a web
    page. It is stopped from the Collect tab, or with its own quit key.
    """
    app["preview"].stop()
    app["sensors_probe"].close()
    await app["http"].close()


async def handle_console(request: web.Request) -> web.Response:
    """What the front-end needs to describe itself: the drive and the mode."""
    app = request.app
    return web.json_response(
        {"root": str(app["root"]), "allow_delete": bool(app["allow_delete"])}
    )


def build_app(args: argparse.Namespace) -> web.Application:
    preinit_tqdm_lock()
    app = web.Application(client_max_size=1024)
    app["root"] = Path(args.dir).expanduser()
    app["allow_delete"] = bool(args.allow_delete)
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
    # The collection session (a subprocess) and the console's own idle camera
    # preview. Only one of the two ever holds a device.
    app["session"] = SessionSupervisor(
        app["root"], monitor_port=getattr(args, "monitor_port", 8766)
    )
    app["preview"] = PreviewCameras()
    app.on_startup.append(_open_client)
    app.on_cleanup.append(_close_session)
    cache = os.environ.get("SO101_OUTPUT_DIR", "outputs")
    app["cache_dir"] = Path(cache).expanduser() / "rig_web_cache"
    app.add_routes(
        [
            web.get("/", handle_index),
            web.get("/api/console", handle_console),
            web.static("/static", STATIC_DIR),
        ]
    )
    add_dataset_routes(app)
    add_lifecycle_routes(app)
    add_session_routes(app)
    add_sensor_routes(app)
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
        help="Enable episode and dataset deletion (destructive)",
    )
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
    print(f"🖥️  rig console on http://127.0.0.1:{args.port}/  (dir: {args.dir})")
    if args.allow_delete:
        print("⚠️  --allow-delete: episode and dataset deletion is ENABLED")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
