"""Routes for the Collect tab: readiness, the live view, and the session.

The console never touches a device while a session runs. What it does instead
is proxy: the recorder serves its own monitor on loopback (see
``common.recording.monitor_server``), and these routes forward the live frames,
the status and the two allowed key presses to it. The browser therefore talks to
one origin and is told "no session" rather than shown a dead image when there is
none.

While nothing is recording the console may open the assigned cameras and the
follower buses itself, so the same tiles answer "is this camera pointing where I
think" and the same joint table answers "do both arms report" before a session
starts. Both previews are released before any session is launched.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import aiohttp  # type: ignore[import]
from aiohttp import web  # type: ignore[import]

from common.recording.controls import control_steps
from common.recording.monitor_server import encode_jpeg, mjpeg_part
from common.web.session import resolve_plan
from common.web.util import in_executor

# The live view is polled by an image element, so a slow or absent monitor must
# fail fast rather than hold the browser's connection open.
MONITOR_TIMEOUT_S = 3.0


def monitor_url(app: web.Application, path: str) -> str:
    return f"http://127.0.0.1:{app['session'].monitor_port}{path}"


async def monitor_get(app: web.Application, path: str) -> "dict[str, Any] | None":
    """Read JSON from the running session's monitor, or ``None`` if it is not up."""
    try:
        timeout = aiohttp.ClientTimeout(total=MONITOR_TIMEOUT_S)
        async with app["http"].get(monitor_url(app, path), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


def _recording_config() -> "dict[str, Any]":
    from common.config_parser import load_recording_config

    return load_recording_config()


# ── Readiness and the collection form ────────────────────────────────────────


async def handle_collect_config(request: web.Request) -> web.Response:
    """What the new-dataset form offers: this machine's cameras and defaults.

    The control steps for both input modes come with it, so the form can say
    what the operator is about to need -- a headset, or the keyboard at the rig
    -- before a session exists to ask.
    """
    app = request.app
    config = await in_executor(app, _recording_config)
    cameras = config.get("cameras") or {}
    return web.json_response(
        {
            "cameras": [
                {"name": name, "enabled": bool(cam["enabled"])}
                for name, cam in sorted(cameras.items())
            ],
            "realsense_rgb_name": (config.get("realsense") or {}).get("rgb_name"),
            "fps": config.get("fps"),
            "controls": {mode: control_steps(mode) for mode in ("quest", "leader")},
        }
    )


def _preflight(hardware: bool, dataset_dir: Path) -> "list[dict[str, Any]]":
    from tool.collect_preflight import run_checks

    return [
        {"name": c.name, "level": c.level, "detail": c.detail}
        for c in run_checks(hardware=hardware, dataset_dir=dataset_dir)
    ]


async def handle_preflight(request: web.Request) -> web.Response:
    """The rig-readiness table. Hardware checks are skipped while a session runs.

    A session holds every camera and both arm buses, so probing them from here
    would either fail or, worse, disturb a recording. The file-level checks
    still answer, which is what an operator wants mid-session anyway (disk).
    """
    app = request.app
    hardware = not app["session"].running() and not app["preview"].running()
    checks = await in_executor(app, _preflight, hardware, Path(app["root"]))
    return web.json_response({"hardware": hardware, "checks": checks})


# ── The session ──────────────────────────────────────────────────────────────


async def _plan_for(request: web.Request, body: "dict[str, Any]") -> "dict[str, Any]":
    app = request.app
    config = await in_executor(app, _recording_config)
    return await in_executor(
        app,
        resolve_plan,
        Path(app["root"]),
        str(body.get("name") or "").strip(),
        str(body.get("task") or "").strip(),
        body,
        config,
        app["session"].running(),
    )


async def handle_session_plan(request: web.Request) -> web.Response:
    """Resolve a start request without running it: the flags, and any refusal."""
    body = await request.json()
    return web.json_response(await _plan_for(request, body))


async def handle_session(request: web.Request) -> web.Response:
    app = request.app
    state = app["session"].state()
    state["monitor"] = await monitor_get(app, "/status") if state["running"] else None
    state["preview"] = app["preview"].stream_names()
    # With no session, the console's own reading of the arms fills the same
    # table (measured only: nothing is commanding them).
    state["preview_joints"] = (
        None if state["running"] else await in_executor(app, app["arms"].snapshot)
    )
    if state["running"]:
        state["controls"] = control_steps(str(state.get("input") or "quest"))
    return web.json_response(state)


async def handle_session_start(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    name = str(body.get("name") or "").strip()
    task = str(body.get("task") or "").strip()
    plan = await _plan_for(request, body)
    if plan["refusals"]:
        raise web.HTTPBadRequest(text="; ".join(plan["refusals"]))
    # One process owns a device: both previews must let go before the recorder
    # tries to open the same cameras and the same buses.
    await in_executor(app, app["preview"].stop)
    await in_executor(app, app["arms"].stop)
    try:
        state = await in_executor(app, app["session"].start, name, task, plan, body)
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    except OSError as exc:
        raise web.HTTPInternalServerError(text=str(exc))
    state["plan"] = plan
    return web.json_response(state)


async def _press(app: web.Application, key: str) -> "dict[str, Any]":
    if not app["session"].running():
        raise web.HTTPConflict(text="no session is running")
    try:
        timeout = aiohttp.ClientTimeout(total=MONITOR_TIMEOUT_S)
        async with app["http"].post(
            monitor_url(app, "/key"), json={"key": key}, timeout=timeout
        ) as resp:
            text = await resp.text()
            if 400 <= resp.status < 500:
                # The session refused the key (it is not one this mode allows).
                # That is the operator's answer, not a broken link, so it keeps
                # its own status and its own words.
                raise web.HTTPForbidden(
                    text=text or f"the session refused {key!r}"
                ) from None
            if resp.status != 200:
                raise web.HTTPBadGateway(text=text or f"monitor said {resp.status}")
            return {"pressed": key}
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise web.HTTPBadGateway(
            text=f"the session's live monitor did not answer ({exc})"
        )


async def handle_session_episode(request: web.Request) -> web.Response:
    """Start or stop-and-save the current episode (the session's own A button)."""
    return web.json_response(await _press(request.app, "a"))


async def handle_session_key(request: web.Request) -> web.Response:
    """Press one of the session's control keys, if the session allows it.

    The session -- not the console -- decides which keys a watcher may press,
    because only it knows how it is being driven: with a headset on, every
    button that moves an arm is already to hand and stays there; with leader
    arms there is no second surface, so enabling is allowed from here. A key it
    refuses comes back with the session's own reason rather than a second copy
    of the rule kept here.
    """
    body = await request.json()
    return web.json_response(await _press(request.app, str(body.get("key", ""))))


async def handle_session_stop(request: web.Request) -> web.Response:
    """Ask the session to end, then escalate on its own if it does not.

    Quitting is a request the recorder honours at the top of its loop, which is
    what parks the arms, finishes an in-flight episode and closes the dataset.
    Only if that is ignored does this interrupt and finally terminate; it never
    kills, because a killed session abandons an open episode.
    """
    app = request.app
    session = app["session"]
    result = await _press(app, "q")
    session.signal_stop()

    def ladder() -> None:
        while session.running():
            session.escalate()
            time.sleep(1.0)

    threading.Thread(target=ladder, name="session-stop", daemon=True).start()
    result["stopping"] = True
    return web.json_response(result)


# ── The live view ────────────────────────────────────────────────────────────


async def handle_live_streams(request: web.Request) -> web.Response:
    """The tiles to show: the session's streams, or the console's preview."""
    app = request.app
    if app["session"].running():
        body = await monitor_get(app, "/streams")
        return web.json_response(
            {"source": "session", "streams": (body or {}).get("streams", [])}
        )
    return web.json_response(
        {"source": "preview", "streams": app["preview"].stream_names()}
    )


async def handle_live_stream(request: web.Request) -> web.StreamResponse:
    name = request.match_info["name"]
    app = request.app
    if app["session"].running():
        return await _proxy_stream(request, name)
    if not app["preview"].running():
        raise web.HTTPConflict(text="no session and no preview is running")
    return await _preview_stream(request, name)


async def _proxy_stream(request: web.Request, name: str) -> web.StreamResponse:
    """Forward the session monitor's multipart stream to the browser."""
    app = request.app
    url = monitor_url(app, f"/stream/{name}.mjpg")
    try:
        async with app["http"].get(
            url, timeout=aiohttp.ClientTimeout(total=None)
        ) as up:
            if up.status != 200:
                raise web.HTTPNotFound(text=f"no stream {name!r} in this session")
            response = web.StreamResponse(
                headers={
                    "Content-Type": up.headers.get(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    ),
                    "Cache-Control": "no-store",
                }
            )
            await response.prepare(request)
            async for chunk in up.content.iter_chunked(16384):
                await response.write(chunk)
            return response
    except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError):
        # The viewer or the session went away mid-stream; neither is an error
        # worth a traceback, and the pane reconnects on its own.
        return web.Response(status=204)


async def _preview_stream(request: web.Request, name: str) -> web.StreamResponse:
    app = request.app
    if name not in app["preview"].stream_names():
        raise web.HTTPNotFound(text=f"no preview camera {name!r}")
    response = web.StreamResponse(
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store",
        }
    )
    await response.prepare(request)
    loop = asyncio.get_running_loop()
    try:
        while app["preview"].running():
            frame = app["preview"].frame(name)
            jpeg = await loop.run_in_executor(None, encode_jpeg, frame)
            if jpeg is not None:
                await response.write(mjpeg_part(jpeg))
            await asyncio.sleep(0.1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def handle_preview_start(request: web.Request) -> web.Response:
    app = request.app
    if app["session"].running():
        raise web.HTTPConflict(
            text="a session is running — its own cameras are already on view"
        )
    try:
        names = await in_executor(app, app["preview"].start)
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"streams": names})


async def handle_preview_stop(request: web.Request) -> web.Response:
    app = request.app
    await in_executor(app, app["preview"].stop)
    return web.json_response({"streams": []})


async def handle_arms_start(request: web.Request) -> web.Response:
    """Read the follower arms while nothing is recording (torque stays off)."""
    app = request.app
    if app["session"].running():
        raise web.HTTPConflict(
            text="a session is running — its own arms are already reported"
        )
    try:
        sides = await in_executor(app, app["arms"].start, app["sensor_map_path"])
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"arms": sides})


async def handle_arms_stop(request: web.Request) -> web.Response:
    app = request.app
    await in_executor(app, app["arms"].stop)
    return web.json_response({"arms": []})


def add_session_routes(app: web.Application) -> None:
    """Register the Collect tab's routes. ``session``/``preview``/``http`` must exist."""
    app.add_routes(
        [
            web.get("/api/collect/config", handle_collect_config),
            web.get("/api/preflight", handle_preflight),
            web.get("/api/session", handle_session),
            web.post("/api/session/plan", handle_session_plan),
            web.post("/api/session/start", handle_session_start),
            web.post("/api/session/episode", handle_session_episode),
            web.post("/api/session/key", handle_session_key),
            web.post("/api/session/stop", handle_session_stop),
            web.get("/api/live/streams", handle_live_streams),
            web.get("/api/live/{name}.mjpg", handle_live_stream),
            web.post("/api/preview/start", handle_preview_start),
            web.post("/api/preview/stop", handle_preview_stop),
            web.post("/api/preview/arms/start", handle_arms_start),
            web.post("/api/preview/arms/stop", handle_arms_stop),
        ]
    )
