"""Unit tests for the live monitor served from inside a collection session.

No rig and no cameras: a stub data manager stands in for the session's
latest-frame store, which is the only thing the monitor reads. What is checked
is the contract the console depends on -- the stream list, the status snapshot,
the multipart framing, and above all the control allow-list, because a key that
moves both arms must not be pressable from a browser.
"""

import unittest
from dataclasses import dataclass

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

from common.recording.monitor_server import (
    BOUNDARY,
    MonitorServer,
    encode_jpeg,
    key_refusal,
    mjpeg_part,
)


@dataclass
class _Status:
    state_label: str = "IDLE"
    episodes_done: int = 2
    episodes_goal: int = 10
    current_frames: int = 0
    recording: bool = False


class _State:
    def __init__(self, value):
        self.value = value


class StubDataManager:
    """The three reads the monitor makes, and nothing else."""

    def __init__(self, names=("central", "wrist_camera_left")):
        self.frames = {n: np.full((48, 64, 3), 128, dtype=np.uint8) for n in names}
        self.ages = {n: 0.02 for n in names}
        self.shutdown = False

    def get_rgb_camera_names(self):
        return sorted(self.frames)

    def get_rgb_image(self, name):
        frame = self.frames.get(name)
        return None if frame is None else frame.copy()

    def get_rgb_image_age(self, name, now=None):
        return self.ages.get(name)

    def get_robot_activity_state(self):
        return _State("ENABLED")

    def get_teleop_active(self):
        return True

    def is_shutdown_requested(self):
        return self.shutdown


class _Capture:
    def __init__(self, name, fps=30, disconnects=0):
        self.name = name
        self.fps = fps
        self.disconnects = disconnects


class TestPureHelpers(unittest.TestCase):
    def test_a_part_carries_its_own_length_and_boundary(self):
        part = mjpeg_part(b"\xff\xd8jpeg", BOUNDARY)
        self.assertTrue(part.startswith(b"--frame\r\n"))
        self.assertIn(b"Content-Type: image/jpeg", part)
        self.assertIn(b"Content-Length: 6", part)
        self.assertTrue(part.endswith(b"\r\n"))

    def test_parts_concatenate_into_one_stream(self):
        stream = mjpeg_part(b"a") + mjpeg_part(b"bb")
        self.assertEqual(stream.count(b"--frame\r\n"), 2)

    def test_the_two_remote_keys_are_allowed(self):
        known = {"y": 1, "x": 1, "a": 1, "b": 1, "q": 1}
        self.assertEqual(key_refusal("a", {"a", "q"}, known), "")
        self.assertEqual(key_refusal("q", {"a", "q"}, known), "")

    def test_the_keys_that_move_the_arms_are_refused(self):
        known = {"y": 1, "x": 1, "a": 1, "b": 1, "q": 1}
        for key in ("y", "x", "b"):
            self.assertIn("stays on the headset", key_refusal(key, {"a", "q"}, known))

    def test_an_unknown_or_empty_key_is_refused(self):
        self.assertIn("no such", key_refusal("z", {"a", "q"}, {"a": 1}))
        self.assertIn("no key", key_refusal("", {"a", "q"}, {"a": 1}))

    def test_jpeg_encoding_downscales_wide_frames(self):
        import cv2

        jpeg = encode_jpeg(np.zeros((480, 640, 3), dtype=np.uint8), max_width=320)
        self.assertIsNotNone(jpeg)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[1], 320)

    def test_a_missing_frame_encodes_to_nothing(self):
        self.assertIsNone(encode_jpeg(None))


class TestStatusSnapshot(unittest.TestCase):
    def setUp(self):
        self.dm = StubDataManager()
        self.monitor = MonitorServer(
            self.dm,
            captures=[
                _Capture("central"),
                _Capture("wrist_camera_left", disconnects=2),
            ],
            status_provider=_Status,
            key_callbacks={k: (lambda: None) for k in "yxabq"},
        )

    def test_streams_follow_the_capture_order_then_the_rest(self):
        self.dm.frames["extra_cam"] = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertEqual(
            self.monitor.stream_names(),
            ["central", "wrist_camera_left", "extra_cam"],
        )

    def test_status_reports_ages_and_reconnects(self):
        status = self.monitor.status()
        by_name = {s["name"]: s for s in status["streams"]}
        self.assertEqual(by_name["wrist_camera_left"]["disconnects"], 2)
        self.assertEqual(by_name["central"]["age_s"], 0.02)
        self.assertEqual(by_name["central"]["configured_fps"], 30)

    def test_status_reports_the_recorder_and_the_arms(self):
        status = self.monitor.status()
        self.assertEqual(status["arms"], "ENABLED")
        self.assertEqual(status["recorder"]["episodes_done"], 2)
        self.assertEqual(status["recorder"]["episodes_goal"], 10)

    def test_status_says_which_keys_may_be_pressed_remotely(self):
        self.assertEqual(self.monitor.status()["allowed_keys"], ["a", "q"])

    def test_a_session_without_a_recorder_has_no_recorder_block(self):
        monitor = MonitorServer(self.dm)
        self.assertNotIn("recorder", monitor.status())


class TestMonitorRoutes(AioHTTPTestCase):
    async def get_application(self):
        self.dm = StubDataManager()
        self.pressed = []
        self.monitor = MonitorServer(
            self.dm,
            captures=[_Capture("central")],
            status_provider=_Status,
            key_callbacks={k: (lambda k=k: self.pressed.append(k)) for k in "yxabq"},
        )
        return self.monitor.build_app()

    async def test_health_and_streams(self):
        self.assertEqual((await self.client.get("/health")).status, 200)
        body = await (await self.client.get("/streams")).json()
        self.assertIn("central", body["streams"])

    async def test_pressing_the_episode_key_calls_the_session_callback(self):
        resp = await self.client.post("/key", json={"key": "a"})
        self.assertEqual(resp.status, 200)
        for _ in range(100):
            if self.pressed:
                break
            await __import__("asyncio").sleep(0.01)
        self.assertEqual(self.pressed, ["a"])

    async def test_pressing_a_key_that_moves_the_arms_is_forbidden(self):
        resp = await self.client.post("/key", json={"key": "y"})
        self.assertEqual(resp.status, 403)
        self.assertEqual(self.pressed, [])

    async def test_an_unknown_stream_is_a_404(self):
        resp = await self.client.get("/stream/nope.mjpg")
        self.assertEqual(resp.status, 404)

    async def test_the_live_view_delivers_multipart_jpeg(self):
        resp = await self.client.get("/stream/central.mjpg")
        self.assertEqual(resp.status, 200)
        self.assertIn("multipart/x-mixed-replace", resp.headers["Content-Type"])
        chunk = await resp.content.read(200)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"image/jpeg", chunk)
        resp.close()


if __name__ == "__main__":
    unittest.main()
