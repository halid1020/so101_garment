"""A small loopback HTTP monitor served from inside a collection session.

One process owns the rig: the V4L2 nodes do not open twice and the arm buses
are lock-guarded, so nothing outside the recorder can read a camera while a
session runs. That is the reason this server exists inside it. It publishes
what the session already has in memory -- the latest frame of every stream, the
recorder's state, each stream's age and reconnect count -- over the loopback
interface, so the rig console can show a live view and the operator can drive an
episode from a browser without a display attached to the rig.

It is deliberately thin:

* frames come from the data manager's latest-frame store, which hands out a
  copy under its own lock, so serving a viewer cannot slow the record loop or
  interleave with it, and the joint snapshot beside them is read the same way --
  nothing here opens a device or talks to a bus;
* the control surface is an allow-list of keys, defaulting to the episode
  toggle and quit. Enabling, parking and homing drive both arms and stay on the
  headset and the keyboard, where the operator is looking at the rig;
* it runs its own event loop on a daemon thread, so it neither needs nor gets a
  say in the session's shutdown path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Iterable

from aiohttp import web  # type: ignore[import]

from common.recording.features import ACTION_FRESH_S, BODY_DOF, BODY_JOINTS, SIDES

# The live view is a monitor, not a recording: a low rate and a small frame keep
# it far below the cost of the capture threads it watches, and JPEG at this
# quality is indistinguishable at tile size.
DEFAULT_VIEW_FPS = 10.0
DEFAULT_QUALITY = 70
DEFAULT_MAX_WIDTH = 480

BOUNDARY = "frame"

# Only these reach the session's button callbacks. Both are decisions the
# operator can safely make from another room: start/stop this episode, and end
# the session. Everything that moves an arm is deliberately absent.
DEFAULT_ALLOWED_KEYS = frozenset({"a", "q"})


def mjpeg_part(jpeg: bytes, boundary: str = BOUNDARY) -> bytes:
    """One ``multipart/x-mixed-replace`` part. Pure — unit-tested."""
    return (
        (
            f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        + jpeg
        + b"\r\n"
    )


def key_refusal(key: str, allowed: "Iterable[str]", known: "Iterable[str]") -> str:
    """Why this key may not be pressed remotely, or "". Pure — unit-tested."""
    allowed = set(allowed)
    known = set(known)
    if not key:
        return "no key given"
    if key not in known:
        return f"no such control key {key!r}"
    if key not in allowed:
        return (
            f"{key!r} is not remotely controllable: it moves the arms, so it "
            "stays on the headset and the session keyboard"
        )
    return ""


def encode_jpeg(
    rgb, quality: int = DEFAULT_QUALITY, max_width: int = DEFAULT_MAX_WIDTH
):
    """RGB array → JPEG bytes, downscaled to ``max_width``. ``None`` if it cannot."""
    import cv2  # type: ignore[import]

    if rgb is None:
        return None
    frame = rgb
    if max_width and frame.shape[1] > max_width:
        scale = max_width / float(frame.shape[1])
        frame = cv2.resize(
            frame, (max_width, max(1, int(round(frame.shape[0] * scale))))
        )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def _row(values, gripper: "float | None") -> "dict[str, float | None]":
    """One side's ``{joint: value}``: five body joints and the gripper. Pure."""
    out: dict[str, float | None] = {}
    for i, name in enumerate(BODY_JOINTS):
        out[name] = None if values is None else round(float(values[i]), 3)
    out["gripper"] = None if gripper is None else round(float(gripper), 3)
    return out


def joint_snapshot(
    measured: Any,
    grippers: "dict[str, float | None]",
    commands: "dict[str, tuple[Any, float | None, float | None]]",
    now: float,
    fresh_s: float = ACTION_FRESH_S,
) -> "dict[str, Any]":
    """Both arms' measured joints beside their last sent command. Pure.

    The pair is the one the recorder STORES -- ``observation.state`` beside
    ``action`` -- so what an operator watches during collection is what a policy
    will later be trained on.

    ``measured`` is the 10-DOF URDF-degree vector (left five, right five) or
    ``None`` before the first read; ``commands[side]`` is what
    ``DualDataManager.get_last_sent_command`` returns. ``fresh`` says whether
    that command is recent enough to be the frame's action, which is the
    question an operator watching this table is really asking.
    """
    out: dict[str, Any] = {}
    for s, side in enumerate(SIDES):
        segment = (
            None if measured is None else measured[s * BODY_DOF : (s + 1) * BODY_DOF]
        )
        urdf_deg, gripper_open, t_mono = commands[side]
        age = None if t_mono is None else max(0.0, now - float(t_mono))
        out[side] = {
            "state": _row(segment, grippers.get(side)),
            "command": _row(urdf_deg, gripper_open),
            "command_age_s": None if age is None else round(age, 3),
            "fresh": age is not None and age < fresh_s,
        }
    return out


class MonitorServer:
    """Serves the live view and the allowed controls of a running session."""

    def __init__(
        self,
        data_manager: Any,
        captures: "Iterable[Any]" = (),
        status_provider: "Callable[[], Any] | None" = None,
        key_callbacks: "dict[str, Callable[[], None]] | None" = None,
        allowed_keys: "Iterable[str]" = DEFAULT_ALLOWED_KEYS,
        host: str = "127.0.0.1",
        port: int = 8766,
        view_fps: float = DEFAULT_VIEW_FPS,
        quality: int = DEFAULT_QUALITY,
        max_width: int = DEFAULT_MAX_WIDTH,
    ) -> None:
        self.data_manager = data_manager
        self.captures = list(captures)
        self.status_provider = status_provider
        self.key_callbacks = dict(key_callbacks or {})
        self.allowed_keys = frozenset(allowed_keys)
        self.host = host
        self.port = port
        self.view_fps = float(view_fps)
        self.quality = int(quality)
        self.max_width = int(max_width)

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._started = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.get("/streams", self.handle_streams),
                web.get("/status", self.handle_status),
                web.get("/stream/{name}.mjpg", self.handle_mjpeg),
                web.post("/key", self.handle_key),
            ]
        )
        return app

    def start(self) -> None:
        """Serve on a daemon thread with its own event loop."""
        self._thread = threading.Thread(
            target=self._serve, name="monitor-server", daemon=True
        )
        self._thread.start()
        self._started.wait(timeout=5.0)

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
            self._started.set()
            loop.run_forever()
        except Exception as exc:  # a dead monitor must never end the session
            print(f"⚠️  live monitor stopped: {exc}")
            self._started.set()
        finally:
            loop.close()

    async def _run(self) -> None:
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(f"👁  live monitor on http://{self.host}:{self.port}/")

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── State ────────────────────────────────────────────────────────────────

    def stream_names(self) -> "list[str]":
        """Every stream with a published frame, cameras named in config order."""
        published = set(self.data_manager.get_rgb_camera_names())
        ordered = [c.name for c in self.captures if c.name in published]
        return ordered + sorted(published - set(ordered))

    def status(self) -> "dict[str, Any]":
        """One snapshot of the session, as the console's Collect pane shows it."""
        now = time.monotonic()
        streams = []
        by_name = {c.name: c for c in self.captures}
        for name in self.stream_names():
            capture = by_name.get(name)
            age = self.data_manager.get_rgb_image_age(name, now)
            streams.append(
                {
                    "name": name,
                    "age_s": None if age is None else round(age, 3),
                    "configured_fps": getattr(capture, "fps", None),
                    "disconnects": getattr(capture, "disconnects", 0),
                }
            )
        measured = self.data_manager.get_current_joint_angles()
        interpolated = self.data_manager.get_current_joint_angles_at(now)
        out: dict[str, Any] = {
            "streams": streams,
            "joints": joint_snapshot(
                measured,
                {
                    side: self.data_manager.get_current_gripper_open_value(side)
                    for side in SIDES
                },
                {side: self.data_manager.get_last_sent_command(side) for side in SIDES},
                now,
            ),
            # How stale the measured joints are at this instant: the same drift
            # the desktop view prints, and the first thing to look at when the
            # table stops moving.
            "joint_drift_s": (
                None if interpolated is None else round(abs(interpolated[1]), 4)
            ),
            "arms": self.data_manager.get_robot_activity_state().value,
            "teleop_active": bool(self.data_manager.get_teleop_active()),
            "shutdown_requested": bool(self.data_manager.is_shutdown_requested()),
            "keys": sorted(self.key_callbacks),
            "allowed_keys": sorted(self.allowed_keys & set(self.key_callbacks)),
        }
        if self.status_provider is not None:
            collection = self.status_provider()
            out["recorder"] = {
                "state": collection.state_label,
                "episodes_done": collection.episodes_done,
                "episodes_goal": collection.episodes_goal,
                "current_frames": collection.current_frames,
                "recording": bool(collection.recording),
            }
        return out

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def handle_streams(self, request: web.Request) -> web.Response:
        return web.json_response({"streams": self.stream_names()})

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(self.status())

    async def handle_key(self, request: web.Request) -> web.Response:
        body = await request.json()
        key = str(body.get("key") or "").lower()
        refusal = key_refusal(key, self.allowed_keys, self.key_callbacks)
        if refusal:
            raise web.HTTPForbidden(text=refusal)
        # The callbacks are the session's own button handlers, already written
        # to be called from a background thread (the headset reader does the
        # same) and already wrapped so a failing one cannot kill the caller.
        threading.Thread(
            target=self.key_callbacks[key], name=f"monitor-key-{key}", daemon=True
        ).start()
        return web.json_response({"pressed": key})

    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in self.stream_names():
            raise web.HTTPNotFound(text=f"no stream {name!r}")
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        period = 1.0 / self.view_fps if self.view_fps > 0 else 0.1
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame = self.data_manager.get_rgb_image(name)
                jpeg = await loop.run_in_executor(
                    None, encode_jpeg, frame, self.quality, self.max_width
                )
                if jpeg is not None:
                    await response.write(mjpeg_part(jpeg))
                await asyncio.sleep(period)
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # the viewer went away; nothing to clean up
        return response
