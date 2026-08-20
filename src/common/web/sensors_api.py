"""Routes for the Signals tab: which device is which, and what it is called.

The tab is called Signals in the browser; this module, its routes and the file
it writes keep the older ``sensor`` name, which is what the recorder, the
preflight tool and ``src/conf/sensor_map.yaml`` have always used.

Every route here touches hardware -- it opens a capture node to show what a
camera sees, or a serial port to watch an arm's raw ticks -- so all of them are
refused while a collection session is running, which owns those devices. The
rules and the map operations live in ``common.web.sensors``.

Assignments are written through the same loader and saver the desktop
assignment tool uses, and the whole map is rewritten each time (that saver
overwrites the file wholesale), so the two ways of doing this job cannot drift.
"""

from __future__ import annotations

from typing import Any

from aiohttp import web  # type: ignore[import]

from common.web.sensors import (
    ASSIGNABLE_CAMERA_NAMES,
    SENSOR_MAP_PATH,
    ArmProbe,
    assign_arm,
    assign_camera,
    assign_realsense,
    clear_assignment,
    discover,
    map_overview,
    read_map,
    stable_device_path,
    write_map,
)
from common.web.util import in_executor


def _map_path(app: web.Application):
    """Where this console reads and writes assignments (a test points it away)."""
    return app["sensor_map_path"]


def _require_idle(app: web.Application) -> None:
    if app["session"].running():
        raise web.HTTPConflict(
            text="a collection session is running — it owns the cameras and the "
            "arm buses. Stop it before reassigning devices"
        )


def _body_str(body: "dict[str, Any]", key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise web.HTTPBadRequest(text=f"missing {key}")
    return value.strip()


def _overview(app: web.Application, rescan: bool) -> "dict[str, Any]":
    """Blocking: the saved map beside the devices that are actually here."""
    if rescan or app.get("sensor_candidates") is None:
        app["sensor_candidates"] = discover()
    candidates = app["sensor_candidates"]
    sensor_map = read_map(_map_path(app))
    return {
        "map": sensor_map,
        "candidates": candidates,
        "names": list(ASSIGNABLE_CAMERA_NAMES),
        "overview": map_overview(
            sensor_map,
            candidates["cameras"],
            candidates["serial"],
            [(d["serial"], d["name"]) for d in candidates["realsense"]],
        ),
    }


async def handle_sensors(request: web.Request) -> web.Response:
    """The map and the devices. Scanning opens each capture node, so it is
    done once and repeated only when asked."""
    app = request.app
    rescan = request.query.get("scan") == "1"
    if rescan:
        _require_idle(app)
    body = await in_executor(app, _overview, app, rescan)
    body["busy"] = app["session"].running()
    body["probe_port"] = app["sensors_probe"].port
    body["preview"] = app["preview"].stream_names()
    return web.json_response(body)


# ── Cameras ──────────────────────────────────────────────────────────────────


async def handle_camera_preview(request: web.Request) -> web.Response:
    """Show one candidate camera, so a gel press says which one it is."""
    app = request.app
    _require_idle(app)
    body = await request.json()
    device = _body_str(body, "device")
    label = str(body.get("label") or "").strip() or device.rsplit("/", 1)[-1]
    try:
        names = await in_executor(app, app["preview"].start, [(label, device)])
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"streams": names})


async def handle_camera_assign(request: web.Request) -> web.Response:
    app = request.app
    _require_idle(app)
    body = await request.json()
    device = _body_str(body, "device")
    name = _body_str(body, "name")
    try:
        node = await in_executor(app, stable_device_path, device)
        sensor_map = await in_executor(app, read_map, _map_path(app))
        sensor_map = assign_camera(sensor_map, name, node)
        await in_executor(app, write_map, sensor_map, _map_path(app))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"name": name, "device": node})


# ── Arms ─────────────────────────────────────────────────────────────────────


async def handle_arm_probe(request: web.Request) -> web.Response:
    """Open one serial port and start reading raw ticks (torque stays off)."""
    app = request.app
    _require_idle(app)
    body = await request.json()
    port = _body_str(body, "port")
    try:
        state = await in_executor(app, app["sensors_probe"].open, port)
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response(state)


async def handle_arm_ticks(request: web.Request) -> web.Response:
    app = request.app
    if request.query.get("rebase") == "1":
        app["sensors_probe"].rebase()
    return web.json_response(await in_executor(app, app["sensors_probe"].read))


async def handle_arm_release(request: web.Request) -> web.Response:
    app = request.app
    await in_executor(app, app["sensors_probe"].close)
    return web.json_response({"port": None})


async def handle_arm_assign(request: web.Request) -> web.Response:
    app = request.app
    _require_idle(app)
    body = await request.json()
    port = _body_str(body, "port")
    role = _body_str(body, "role")
    side = _body_str(body, "side")
    try:
        node = await in_executor(app, stable_device_path, port)
        sensor_map = await in_executor(app, read_map, _map_path(app))
        sensor_map = assign_arm(sensor_map, role, side, node)
        await in_executor(app, write_map, sensor_map, _map_path(app))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"role": role, "side": side, "device": node})


# ── Depth camera and unassignment ────────────────────────────────────────────


async def handle_realsense_assign(request: web.Request) -> web.Response:
    app = request.app
    _require_idle(app)
    body = await request.json()
    serial = _body_str(body, "serial")
    sensor_map = await in_executor(app, read_map, _map_path(app))
    sensor_map = assign_realsense(sensor_map, serial)
    await in_executor(app, write_map, sensor_map, _map_path(app))
    return web.json_response({"serial": serial})


async def handle_clear(request: web.Request) -> web.Response:
    app = request.app
    _require_idle(app)
    body = await request.json()
    kind = _body_str(body, "kind")
    key = str(body.get("key") or "")
    sensor_map = await in_executor(app, read_map, _map_path(app))
    try:
        sensor_map = clear_assignment(sensor_map, kind, key)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    await in_executor(app, write_map, sensor_map, _map_path(app))
    return web.json_response({"cleared": [kind, key]})


def add_sensor_routes(app: web.Application) -> None:
    """Register the Signals tab's routes and the state they need."""
    app["sensors_probe"] = ArmProbe()
    app["sensor_candidates"] = None
    app.setdefault("sensor_map_path", SENSOR_MAP_PATH)
    app.add_routes(
        [
            web.get("/api/sensors", handle_sensors),
            web.post("/api/sensors/camera/preview", handle_camera_preview),
            web.post("/api/sensors/camera/assign", handle_camera_assign),
            web.post("/api/sensors/arm/probe", handle_arm_probe),
            web.get("/api/sensors/arm/ticks", handle_arm_ticks),
            web.post("/api/sensors/arm/release", handle_arm_release),
            web.post("/api/sensors/arm/assign", handle_arm_assign),
            web.post("/api/sensors/realsense/assign", handle_realsense_assign),
            web.post("/api/sensors/clear", handle_clear),
        ]
    )
