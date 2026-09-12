#!/usr/bin/env python3
"""This rig's devices, for the shared console, in this repo's interpreter.

The console (``actoris-harena console``) is shared between robots and imports no
hardware code: it cannot, because this rig needs feetech and mujoco while a UR3e
needs ur-rtde, and the two environments cannot be installed together. So
anything that opens a device runs here instead, started as a subprocess with
``venv/bin/python`` as ``rig.yaml`` declares, and answers over loopback.

The protocol is the one the collection session already uses -- a JSON status and
MJPEG streams on ``127.0.0.1:<port>`` -- so the console proxies from an agent
exactly as it already proxies from a running session, and there is one way to
read a camera rather than two.

What lives here is only what needs a device:

    GET /status                what this rig is, which streams exist, what is open
    GET /cameras               the preview's stream names and what it could not open
    GET /cameras/<name>        MJPEG, one stream
    POST /cameras/start        open the assigned cameras (body: {"specs": [[name, dev]]})
    POST /cameras/stop         release them
    GET /arms                  the follower joints, torque off
    POST /arms/start           begin reading them
    POST /arms/stop            release them

Everything else the console shows -- datasets, training, the ablation views --
it does itself, because none of that needs this rig to be plugged in.

Bound to 127.0.0.1 only. This opens motor buses and cameras on request with no
authentication, which is safe on loopback and would not be anywhere else.

Usage (the console does this for you):

    venv/bin/python tool/rig_agent.py --port 8781
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

# Same reasons as tool/rig_web.py: keep LeRobot offline so an incomplete dataset
# cannot become a Hub lookup, and quiet OpenCV's per-node warning on a rig with
# seven cameras. Both must precede the first import of either library.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from actoris_harena.recording.monitor_wire import (  # noqa: E402
    BOUNDARY,
    encode_jpeg,
    mjpeg_part,
)
from aiohttp import web  # noqa: E402

from common.robot_schema import SCHEMA  # noqa: E402
from common.web.sensors import ArmProbe  # noqa: E402
from common.web.session import PreviewArms, PreviewCameras  # noqa: E402

#: How long a stalled MJPEG stream waits before sending the frame it has again.
#: Short enough that a browser tab does not look frozen, long enough that a
#: camera delivering at 30 fps is not re-encoded for nothing.
_FRAME_PERIOD_S = 1.0 / 15.0


def build_app() -> web.Application:
    """The agent's routes, over one set of devices.

    One preview and one probe per process, deliberately: a camera can be opened
    once, and a second reader of a motor bus is a second writer waiting to
    happen. The console starts one agent per rig and keeps it.
    """
    app = web.Application()
    cameras = PreviewCameras()
    arms = PreviewArms()
    probe = ArmProbe()

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "rig": "so101",
                "schema": {
                    "limbs": list(SCHEMA.limbs),
                    "body_joints": list(SCHEMA.body_joints),
                    "state_dim": SCHEMA.state_dim,
                    "gripper_columns": list(SCHEMA.gripper_columns),
                },
                "cameras": {
                    "running": cameras.running(),
                    "streams": cameras.stream_names() if cameras.running() else [],
                    "missing": cameras.missing(),
                },
                "arms": {"running": arms.running()},
            }
        )

    async def cameras_get(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "running": cameras.running(),
                "streams": cameras.stream_names() if cameras.running() else [],
                "missing": cameras.missing(),
            }
        )

    async def cameras_start(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        specs = body.get("specs")
        pairs = [(str(a), str(b)) for a, b in specs] if specs else None
        opened = await asyncio.get_running_loop().run_in_executor(
            None, cameras.start, pairs
        )
        return web.json_response({"opened": opened, "missing": cameras.missing()})

    async def cameras_stop(_request: web.Request) -> web.Response:
        await asyncio.get_running_loop().run_in_executor(None, cameras.stop)
        return web.json_response({"running": cameras.running()})

    async def camera_stream(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        try:
            while True:
                frame = cameras.frame(name)
                if frame is not None:
                    await response.write(mjpeg_part(encode_jpeg(frame)))
                await asyncio.sleep(_FRAME_PERIOD_S)
        except (ConnectionResetError, asyncio.CancelledError):
            # The browser closed the tab. Not an error, and not worth a line in
            # a log an operator is reading for real faults.
            pass
        return response

    async def arms_get(_request: web.Request) -> web.Response:
        return web.json_response(
            {"running": arms.running(), "snapshot": arms.snapshot()}
        )

    async def arms_start(_request: web.Request) -> web.Response:
        started = await asyncio.get_running_loop().run_in_executor(None, arms.start)
        return web.json_response({"started": started})

    async def arms_stop(_request: web.Request) -> web.Response:
        await asyncio.get_running_loop().run_in_executor(None, arms.stop)
        return web.json_response({"running": arms.running()})

    async def shutdown(_app: web.Application) -> None:
        """Release every device on the way out, including on a crash.

        A camera left open survives this process and the next session cannot
        have it; a bus left open holds a serial port. Both are worse than a
        slow exit.
        """
        for release in (cameras.stop, arms.stop, probe.close):
            try:
                release()
            except Exception:  # noqa: BLE001 - a failed release must not mask the rest
                pass

    app.router.add_get("/status", status)
    app.router.add_get("/cameras", cameras_get)
    app.router.add_post("/cameras/start", cameras_start)
    app.router.add_post("/cameras/stop", cameras_stop)
    app.router.add_get("/cameras/{name}", camera_stream)
    app.router.add_get("/arms", arms_get)
    app.router.add_post("/arms/start", arms_start)
    app.router.add_post("/arms/stop", arms_stop)
    app.on_cleanup.append(shutdown)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, required=True, help="loopback port to serve on"
    )
    args = parser.parse_args()
    print(f"🔌 rig agent for so101 on http://127.0.0.1:{args.port}")
    web.run_app(build_app(), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
