"""End-to-end check of the console driving a collection session.

The risky part of the Collect tab is the seam between two processes: the
console supervises a session, and everything live comes back through the
session's own monitor. This test exercises that whole chain -- start, proxied
status, proxied live frames, the episode key, the stop key -- against a
stand-in for the recorder that serves a real ``MonitorServer`` over stub
frames. No rig, no cameras, no MuJoCo; it spawns one subprocess.

Run:  PYTHONPATH=.:src python -m unittest test.integration.test_console_session
"""

import argparse
import json
import os
import socket
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from tool.rig_web import build_app

FAKE_RECORDER = '''
"""Stand-in for the teleoperation recorder: a monitor over stub frames."""
import sys, time
import numpy as np
sys.path[:0] = [{repo!r}, {src!r}]
from common.recording.monitor_server import MonitorServer

class Stub:
    def __init__(self):
        self.frames = {{"central": np.full((48, 64, 3), 90, dtype=np.uint8)}}
        self.state = "DISABLED"
        self.pressed = []
    def get_rgb_camera_names(self): return sorted(self.frames)
    def get_rgb_image(self, n): return self.frames[n].copy()
    def get_rgb_image_age(self, n, now=None): return 0.01
    def get_current_joint_angles(self): return np.arange(10, dtype=float)
    def get_current_joint_angles_at(self, t): return (np.arange(10, dtype=float), 0.003)
    def get_current_gripper_open_value(self, side): return 0.5
    def get_last_sent_command(self, side):
        return (np.arange(5, dtype=float), 0.5, time.monotonic())
    def get_robot_activity_state(self):
        return type("S", (), {{"value": self.state}})()
    def get_teleop_active(self): return False
    def is_shutdown_requested(self): return False

stub = Stub()
done = []

class Status:
    state_label = "IDLE"
    episodes_done = 0
    episodes_goal = 0
    current_frames = 0
    recording = False

def on_a():
    stub.pressed.append("a")
    Status.state_label = "RECORDING"
    Status.recording = True

monitor = MonitorServer(
    stub,
    status_provider=Status,
    key_callbacks={{"a": on_a, "q": lambda: done.append(True)}},
    port={port},
)
monitor.start()
print("recorder ready", flush=True)
while not done:
    time.sleep(0.05)
print("recorder quitting", flush=True)
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestConsoleSession(AioHTTPTestCase):
    async def get_application(self):
        warnings.filterwarnings("ignore", category=web.NotAppKeyWarning)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.port = free_port()
        repo = Path(__file__).resolve().parents[2]
        script = self.root / "fake_recorder.py"
        script.write_text(
            FAKE_RECORDER.format(repo=str(repo), src=str(repo / "src"), port=self.port)
        )
        # The console's own state stays inside the temporary directory.
        self._outputs = os.environ.get("SO101_OUTPUT_DIR")
        os.environ["SO101_OUTPUT_DIR"] = str(self.root / "outputs")
        app = build_app(
            argparse.Namespace(
                dir=str(self.root),
                fps=None,
                prerender=False,
                monitor_port=self.port,
                mount_dir=str(self.root / "mounts"),
            )
        )
        app["session"].teleop = script
        app["session"].python = sys.executable
        # Never the machine's real assignment file, even though the routes
        # under test are refused before they would reach it.
        app["sensor_map_path"] = self.root / "sensor_map.yaml"
        return app

    async def tearDownAsync(self):
        session = self.app["session"]
        proc = session._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        await super().tearDownAsync()
        if self._outputs is None:
            os.environ.pop("SO101_OUTPUT_DIR", None)
        else:
            os.environ["SO101_OUTPUT_DIR"] = self._outputs
        self.tmp.cleanup()

    async def _wait_for_monitor(self, tries: int = 200) -> "dict":
        import asyncio

        for _ in range(tries):
            body = await (await self.client.get("/api/session")).json()
            if body.get("monitor"):
                return body
            await asyncio.sleep(0.05)
        self.fail(f"the session's monitor never answered; tail={body.get('tail')}")
        return {}  # unreachable: self.fail raises

    async def test_the_console_drives_a_session_end_to_end(self):
        # 1. Start: the plan is resolved and the recorder is launched with it.
        start = await self.client.post(
            "/api/session/start",
            json={"name": "towel-fold", "task": "fold the towel", "streams": []},
        )
        self.assertEqual(start.status, 200, await start.text())
        started = await start.json()
        self.assertTrue(started["running"])
        self.assertIn("--monitor-port", started["argv"])

        # 2. Status: the session's own monitor is proxied to one origin.
        body = await self._wait_for_monitor()
        self.assertEqual(body["monitor"]["arms"], "DISABLED")
        self.assertEqual(body["monitor"]["recorder"]["state"], "IDLE")
        self.assertEqual(body["monitor"]["allowed_keys"], ["a", "q"])
        # The proprioception the operator watches instead of standing at the
        # rig: both arms' measured joints beside the command last sent.
        joints = body["monitor"]["joints"]
        self.assertEqual(joints["left"]["state"]["shoulder_pan"], 0.0)
        self.assertEqual(joints["right"]["state"]["shoulder_pan"], 5.0)
        self.assertTrue(joints["right"]["fresh"])
        self.assertEqual(body["monitor"]["joint_drift_s"], 0.003)
        # And how it is driven, for the pane beside it.
        self.assertEqual(body["controls"][0]["key"], "Y")

        # 3. Live view: the tiles follow the session, and frames arrive.
        streams = await (await self.client.get("/api/live/streams")).json()
        self.assertEqual(streams["source"], "session")
        self.assertEqual(streams["streams"], ["central"])
        resp = await self.client.get("/api/live/central.mjpg")
        self.assertEqual(resp.status, 200)
        chunk = await resp.content.read(300)
        self.assertIn(b"--frame", chunk)
        resp.close()

        # 4. The episode key reaches the recorder's own button handler.
        press = await self.client.post("/api/session/episode", json={})
        self.assertEqual(press.status, 200)
        for _ in range(200):
            body = await (await self.client.get("/api/session")).json()
            if body["monitor"]["recorder"]["recording"]:
                break
            await __import__("asyncio").sleep(0.05)
        self.assertTrue(body["monitor"]["recorder"]["recording"])

        # 5. Stopping asks the session to quit, and it goes away on its own.
        stop = await self.client.post("/api/session/stop", json={})
        self.assertEqual(stop.status, 200)
        for _ in range(200):
            body = await (await self.client.get("/api/session")).json()
            if not body["running"]:
                break
            await __import__("asyncio").sleep(0.05)
        self.assertFalse(body["running"])
        self.assertIn("recorder quitting", "\n".join(body["tail"]))

    async def test_reassigning_devices_is_refused_while_a_session_runs(self):
        start = await self.client.post(
            "/api/session/start", json={"name": "towel-fold", "task": "fold it"}
        )
        self.assertEqual(start.status, 200, await start.text())
        await self._wait_for_monitor()
        # The session holds every camera and both arm buses; probing one now
        # would either fail or disturb a recording.
        for url, payload in (
            ("/api/sensors/camera/preview", {"device": "/dev/video0"}),
            (
                "/api/sensors/camera/assign",
                {"device": "/dev/video0", "name": "central"},
            ),
            ("/api/sensors/arm/probe", {"port": "/dev/ttyACM0"}),
            (
                "/api/sensors/arm/assign",
                {"port": "/dev/ttyACM0", "role": "follower", "side": "right"},
            ),
        ):
            resp = await self.client.post(url, json=payload)
            self.assertEqual(resp.status, 409, url)
        resp = await self.client.get("/api/sensors?scan=1")
        self.assertEqual(resp.status, 409)
        self.assertFalse((self.root / "sensor_map.yaml").exists())

    async def test_the_collection_directory_cannot_change_under_a_session(self):
        start = await self.client.post(
            "/api/session/start", json={"name": "towel-fold", "task": "fold it"}
        )
        self.assertEqual(start.status, 200, await start.text())
        await self._wait_for_monitor()
        # The running recorder has a dataset open under this directory; moving
        # the console elsewhere would leave it recording somewhere invisible.
        elsewhere = self.root / "second"
        elsewhere.mkdir()
        resp = await self.client.post("/api/roots/use", json={"path": str(elsewhere)})
        self.assertEqual(resp.status, 409)
        self.assertIn("session is running", await resp.text())
        body = await (await self.client.get("/api/roots")).json()
        self.assertEqual(body["root"], str(self.root))

    async def test_a_dataset_that_exists_but_holds_nothing_is_refused(self):
        (self.root / "stillborn" / "meta").mkdir(parents=True)
        (self.root / "stillborn" / "meta" / "info.json").write_text(
            json.dumps({"fps": 30, "total_episodes": 0, "features": {}})
        )
        resp = await self.client.post(
            "/api/session/start", json={"name": "stillborn", "task": "x"}
        )
        self.assertEqual(resp.status, 400)
        self.assertIn("no saved episodes", await resp.text())


if __name__ == "__main__":
    unittest.main()
