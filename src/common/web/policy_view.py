"""A live view of one autonomous rollout, served by tool/run_policy.py.

A rollout that misbehaves gives an operator almost nothing to go on: the arms
move, the grasp misses, and it is over. The four things that would explain it
are all invisible from the terminal -- what the policy was SHOWN, what it
PLANNED from that, what the arms DID with the plan, and whether the plan arrived
in TIME -- so this serves them, on loopback, while the run happens.

It is a window onto a run, not a controller of the rig. It never touches a
camera or a bus: frames come from the data manager the control loop already
publishes into, numbers come from the snapshot that loop posts every tick, and
the only thing it can change is the run's mode (``common.policy_run``). That
mode can hold the arms, advance them one chunk, resume, or end the run -- every
one of which asks for LESS motion than the terminal already authorised. Torque
is enabled once, at the confirmation prompt, and nothing here can enable it.

Served on 127.0.0.1: it is unauthenticated, and it steers a robot.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import web  # type: ignore[import]

from common.chunking import STRATEGIES
from common.recording.monitor_server import (
    BOUNDARY,
    DEFAULT_MAX_WIDTH,
    DEFAULT_QUALITY,
    DEFAULT_VIEW_FPS,
    encode_jpeg,
    mjpeg_part,
)
from common.web.util import revalidate_assets

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _dumps(payload) -> str:
    """JSON for the status poll, tolerant of numpy's scalar types.

    Belt and braces. Everything published SHOULD already be plain Python -- the
    control loop converts -- but a single numpy float leaking in makes the
    encoder raise, the poll 500, and the whole page go blank in the middle of a
    rollout. Losing the page is a far worse outcome than printing a number that
    came in the wrong wrapper.
    """
    return json.dumps(payload, default=_plain)


def _plain(value):
    item = getattr(value, "item", None)
    if item is not None:
        return item()
    raise TypeError(f"cannot serialise {type(value).__name__} for the status poll")


def chunk_payload(source) -> "dict[str, Any] | None":
    """The plan a source last received, as the page draws it. Pure."""
    chunk = getattr(source, "last_chunk", None)
    if chunk is None:
        return None
    actions = np.asarray(chunk, dtype=float)
    return {
        "kind": "plan",
        "seq": int(getattr(source, "last_chunk_seq", 0) or 0),
        "at": float(getattr(source, "last_chunk_at", 0.0) or 0.0),
        "n": int(actions.shape[0]),
        "actions": actions.round(3).tolist(),
    }


def pending_payload(source) -> "dict[str, Any] | None":
    """What the arms will actually execute next, in order. Pure.

    Distinct from :func:`chunk_payload`, which is what the policy SAID. The two
    differ under every splice but ``append``: the queue has had the stale rows
    removed and, under ``blend``, its first few actions are a cross-fade that
    appears in no chunk -- so this is what the arms will really do, and the page
    shows it beside the plan that was asked for.
    """
    getter = getattr(source, "pending", None)
    if getter is None:
        return None
    queued = getter()
    if queued is None or len(queued) == 0:
        return None
    actions = np.asarray(queued, dtype=float)
    return {
        "kind": "queued",
        "seq": int(getattr(source, "last_chunk_seq", 0) or 0),
        "n": int(actions.shape[0]),
        "actions": actions.round(3).tolist(),
    }


class PolicyView:
    """One rollout's view: the frames, the numbers, the twin and the throttle."""

    def __init__(
        self,
        control,
        source,
        data_manager,
        image_names: "list[str]",
        host: str = "127.0.0.1",
        port: int = 8767,
        view_fps: float = DEFAULT_VIEW_FPS,
        quality: int = DEFAULT_QUALITY,
        max_width: int = DEFAULT_MAX_WIDTH,
        arm_from_view: bool = False,
        twin_port: "int | None" = None,
    ) -> None:
        #: Whether this run delegated its 'the arms will move' consent to the
        #: page. False means the terminal already took it and the page must
        #: not offer a second, unguarded door to the same torque.
        self.arm_from_view = bool(arm_from_view)
        #: Why the view is not serving, if it is not. See :meth:`start`.
        self.error: "str | None" = None
        #: Why the 3D twin is not there, if it is not. Unlike ``error`` this is
        #: survivable: the rest of the page is still true without it.
        self.twin_error: "str | None" = None
        #: Called with the new settings whenever the splice changes, so a run
        #: can close one measurement segment and open the next. Set by the
        #: caller; a no-op if nobody cares.
        self.on_splice_change = lambda _settings: None
        self.control = control
        self.source = source
        self.data_manager = data_manager
        self.image_names = list(image_names)
        self.host = host
        self.port = port
        self.view_fps = float(view_fps)
        self.quality = int(quality)
        self.max_width = int(max_width)
        #: The 3D twin is a SECOND server, on its own port, because viser is a
        #: server: the page embeds it rather than proxying it. Zero means do
        #: not serve one at all -- which is what a test wants, and what a box
        #: with no GPU to render on may want too.
        self.twin_port = int(port + 1 if twin_port is None else twin_port)

        self._twin: "Any | None" = None
        self._twin_lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._runner: "web.AppRunner | None" = None
        self._started = threading.Event()

    # -- lifecycle -----------------------------------------------------
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=4096, middlewares=[revalidate_assets])
        app.add_routes(
            [
                web.get("/", self.handle_index),
                web.get("/api/status", self.handle_status),
                web.post("/api/mode", self.handle_mode),
                web.post("/api/strategy", self.handle_strategy),
                web.post("/api/task", self.handle_task),
                web.get("/stream/{name}.mjpg", self.handle_mjpeg),
                web.get("/shown/{name}.jpg", self.handle_shown),
                web.get("/twin/at", self.handle_twin),
                web.static("/static", STATIC_DIR),
            ]
        )
        return app

    def start(self) -> bool:
        """Serve the view. False if it could not bind, with ``error`` set.

        The caller must not shrug this off. A rollout whose view failed to bind
        leaves the operator looking at SOME OTHER run's page -- the previous
        one, still holding the port -- showing that run's cameras and that run's
        plan, with a throttle wired to a rollout that has nothing to do with the
        arms now moving. Everything on screen is plausible and none of it is
        true.
        """
        # Eagerly, before the page exists: the twin is an <iframe>, and an
        # iframe that loads before its server is listening shows a browser
        # error page and never retries.
        if self.twin_port > 0:
            try:
                from common.web.policy_ghost import GhostPair

                self._twin = GhostPair(host=self.host, port=self.twin_port)
            except Exception as exc:  # noqa: BLE001 - a missing twin is not fatal
                # The numbers, the cameras and the throttle are all still worth
                # having, so this degrades rather than refusing to serve.
                self.twin_error = f"{type(exc).__name__}: {exc}"
                self._twin = None

        self._thread = threading.Thread(
            target=self._serve, name="policy-view", daemon=True
        )
        self._thread.start()
        self._started.wait(timeout=5.0)
        return self.error is None

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
            self._started.set()
            loop.run_forever()
        except Exception as exc:  # noqa: BLE001 - reported, not raised, here
            self.error = f"{type(exc).__name__}: {exc}"
            self._started.set()
        finally:
            loop.close()

    async def _run(self) -> None:
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    def stop(self) -> None:
        with self._twin_lock:
            if self._twin is not None:
                self._twin.stop()
                self._twin = None
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- what the page reads -------------------------------------------
    def status(self) -> "dict[str, Any]":
        source = self.source
        sent = getattr(source, "last_sent", lambda: None)()
        return {
            **self.control.snapshot(),
            "cameras": self.image_names,
            "chunk": chunk_payload(source),
            "sent_state": None
            if sent is None
            else np.asarray(sent[0]).round(3).tolist(),
            "threshold": int(getattr(source, "threshold", 0) or 0),
            "actions_per_chunk": int(getattr(source, "actions", 0) or 0),
            "pending": pending_payload(source),
            "strategies": list(STRATEGIES),
            "splice": (source.settings() if hasattr(source, "settings") else None),
            "arm_from_view": bool(self.arm_from_view),
            "twin_url": (None if self._twin is None else self._twin.url),
            "twin_error": self.twin_error,
        }

    async def handle_strategy(self, request: web.Request) -> web.Response:
        """Change the splice while the run continues. See policy_client."""
        setter = getattr(self.source, "set_strategy", None)
        if setter is None:
            raise web.HTTPBadRequest(
                text="this run infers locally, one action at a time: there is "
                "no chunk to splice"
            )
        body = await request.json()
        params = {
            key: body[key]
            for key in ("execute_ratio", "blend_window", "new_weight", "ramp_kind")
            if body.get(key) is not None
        }
        try:
            settings = setter(body.get("strategy"), **params)
        except ValueError as exc:  # ChunkingError is one
            raise web.HTTPBadRequest(text=str(exc))
        self.on_splice_change(settings)
        return web.json_response(settings)

    async def handle_task(self, request: web.Request) -> web.Response:
        """Set the language task the policy is given. See run_policy.

        A run may be started without one -- that is the point of a console you
        drive entirely from the browser -- and then it waits here before it
        infers anything. Changing it later is allowed too: the task travels with
        every request, so the next chunk simply answers a different question.
        """
        body = await request.json()
        task = str(body.get("task", "")).strip()
        if not task:
            raise web.HTTPBadRequest(text="a task cannot be empty")
        setter = getattr(self.source, "set_task", None)
        if setter is None:
            raise web.HTTPBadRequest(text="this source has no task to set")
        setter(task)
        self.control.publish(task=task)
        return web.json_response({"task": task})

    async def handle_index(self, _request: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "policy.html")

    async def handle_status(self, _request: web.Request) -> web.Response:
        return web.json_response(self.status(), dumps=_dumps)

    async def handle_mode(self, request: web.Request) -> web.Response:
        body = await request.json()
        mode = str(body.get("mode", ""))
        if mode == "arm" and not self.arm_from_view:
            # This run took its consent at the terminal. Accepting it here too
            # would mean two doors to the same torque, one of them unguarded.
            raise web.HTTPBadRequest(
                text="this run was armed at the terminal; the page cannot "
                "enable the arms (start it with --arm-from-view)"
            )
        if mode in ("step", "preview") and not getattr(self.source, "chunked", False):
            raise web.HTTPBadRequest(
                text="this run infers locally, one action at a time: it has no "
                "chunk to preview or step through"
            )
        try:
            now = self.control.request(mode)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
        return web.json_response({"mode": now})

    # -- pictures ------------------------------------------------------
    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        """The cameras, live: what the rig sees right now."""
        name = request.match_info["name"]
        if name not in self.image_names:
            raise web.HTTPNotFound(text=f"no stream {name!r}")
        return await self._stream(
            request, lambda: self.data_manager.get_rgb_image(name), self.view_fps
        )

    async def handle_shown(self, request: web.Request) -> web.Response:
        """One frame of the window the policy was last SENT -- not the live one.

        The difference matters: by the time a chunk is executing, the frames it
        was planned from are half a second old, and those are the ones that
        explain the plan.
        """
        name = request.match_info["name"]
        sent = getattr(self.source, "last_sent", lambda: None)()
        frame = None if sent is None else (sent[1] or {}).get(name)
        loop = asyncio.get_running_loop()
        jpeg = await loop.run_in_executor(
            None, encode_jpeg, frame, self.quality, self.max_width
        )
        if jpeg is None:
            raise web.HTTPNotFound(text=f"nothing sent for {name!r} yet")
        return web.Response(body=jpeg, content_type="image/jpeg")

    async def handle_twin(self, request: web.Request) -> web.Response:
        """Point the twin at one action: the arms now, and where it sends them.

        Blue is the measured pose, orange is ``last_chunk[i]``, shown together in
        a 3D scene the viewer can turn, so the gap between the ghosts is the
        motion still to come and can be looked at from the side.

        Nothing is returned but ``204``: the picture is drawn in the browser, by
        viser, over its own connection. This route only moves the two poses.

        The CALLER names the frame -- ``?seq=<n>&i=<k>`` -- and gets a 409 if
        that plan is no longer the current one. That is the whole point of this
        route's shape: the page, not the server, decides which action is on
        screen, so the subject can never be swapped underneath the animation.
        """
        try:
            want_seq = int(request.query.get("seq", "0"))
            index = int(request.query.get("i", "0"))
        except ValueError:
            raise web.HTTPBadRequest(text="seq and i must be integers")

        chunk = getattr(self.source, "last_chunk", None)
        if chunk is None:
            raise web.HTTPNotFound(text="no plan has arrived yet")
        have_seq = int(getattr(self.source, "last_chunk_seq", 0) or 0)
        if have_seq != want_seq:
            # Not an error: the page is one poll behind. It re-reads the status
            # and asks again, and meanwhile keeps the pose already on screen.
            raise web.HTTPConflict(text=f"plan #{want_seq} is gone; now #{have_seq}")
        actions = np.asarray(chunk, dtype=float)
        plan = actions[index % len(actions)] if len(actions) else None
        state = (self.control.snapshot().get("state")) or None

        twin = self._twin
        if twin is None:
            raise web.HTTPNotFound(text="the twin is not running")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, twin.show, state, plan)
        return web.Response(status=204)

    async def _stream(self, request, frame_source, fps: float) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        period = 1.0 / fps if fps > 0 else 0.1
        loop = asyncio.get_running_loop()
        try:
            while True:
                rgb = await loop.run_in_executor(None, frame_source)
                jpeg = await loop.run_in_executor(
                    None, encode_jpeg, rgb, self.quality, self.max_width
                )
                if jpeg is not None:
                    await response.write(mjpeg_part(jpeg))
                await asyncio.sleep(period)
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # the viewer went away; nothing to clean up
        return response
