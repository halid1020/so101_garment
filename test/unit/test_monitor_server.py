"""Unit tests for the live monitor served from inside a collection session.

No rig and no cameras: a stub data manager stands in for the session's
latest-frame store, which is the only thing the monitor reads. What is checked
is the contract the console depends on -- the stream list, the status snapshot,
the multipart framing, and above all the control allow-list, because a key that
moves both arms must not be pressable from a browser.
"""

import time
import unittest
from dataclasses import dataclass

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

from common.recording.controls import control_steps
from common.recording.monitor_server import (
    BOUNDARY,
    MonitorServer,
    allowed_keys_for,
    encode_jpeg,
    joint_snapshot,
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
    """Exactly the reads the monitor makes, and nothing else."""

    def __init__(self, names=("central", "wrist_camera_left")):
        self.frames = {n: np.full((48, 64, 3), 128, dtype=np.uint8) for n in names}
        self.ages = {n: 0.02 for n in names}
        self.shutdown = False
        self.joints = np.arange(10, dtype=np.float64)
        self.grippers = {"left": 0.25, "right": 0.75}
        self.commands = {
            "left": (np.arange(5, dtype=np.float64) + 0.5, 0.3, time.monotonic()),
            "right": (None, None, None),
        }

    def get_rgb_camera_names(self):
        return sorted(self.frames)

    def get_rgb_image(self, name):
        frame = self.frames.get(name)
        return None if frame is None else frame.copy()

    def get_rgb_image_age(self, name, now=None):
        return self.ages.get(name)

    def get_current_joint_angles(self):
        return None if self.joints is None else self.joints.copy()

    def get_current_joint_angles_at(self, t_ref):
        return None if self.joints is None else (self.joints.copy(), 0.004)

    def get_current_gripper_open_value(self, side):
        return self.grippers[side]

    def get_last_sent_command(self, side):
        return self.commands[side]

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

    def test_the_allow_list_follows_how_the_session_is_driven(self):
        # Quest: the headset has every arm-moving button already. Leader: the
        # control surface is a keyboard, and a console-started session has none.
        self.assertEqual(set(allowed_keys_for("quest")), {"a", "q"})
        self.assertEqual(set(allowed_keys_for("leader")), {"y", "a", "q"})

    def test_an_unrecognised_mode_gets_the_careful_answer(self):
        self.assertEqual(set(allowed_keys_for("")), {"a", "q"})

    def test_enabling_is_allowed_for_a_leader_session_and_not_a_quest_one(self):
        known = {"y": 1, "x": 1, "a": 1, "b": 1, "q": 1}
        self.assertEqual(key_refusal("y", allowed_keys_for("leader"), known), "")
        self.assertIn(
            "stays on the headset",
            key_refusal("y", allowed_keys_for("quest"), known),
        )

    def test_park_and_home_are_refused_however_the_session_is_driven(self):
        known = {"y": 1, "x": 1, "a": 1, "b": 1, "q": 1}
        for mode in ("quest", "leader"):
            for key in ("x", "b"):
                self.assertNotEqual(key_refusal(key, allowed_keys_for(mode), known), "")

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


class TestJointSnapshot(unittest.TestCase):
    """What the console draws: the measured joints beside the sent command."""

    def _commands(self, age=0.001):
        now = 100.0
        return now, {
            "left": (np.arange(5, dtype=np.float64), 0.4, now - age),
            "right": (None, None, None),
        }

    def test_each_side_takes_its_half_of_the_joint_vector(self):
        now, commands = self._commands()
        snap = joint_snapshot(
            np.arange(10, dtype=np.float64), {"left": 0.1, "right": 0.9}, commands, now
        )
        self.assertEqual(snap["left"]["state"]["shoulder_pan"], 0.0)
        self.assertEqual(snap["left"]["state"]["wrist_roll"], 4.0)
        self.assertEqual(snap["right"]["state"]["shoulder_pan"], 5.0)
        self.assertEqual(snap["right"]["state"]["wrist_roll"], 9.0)
        self.assertEqual(snap["right"]["state"]["gripper"], 0.9)

    def test_a_fresh_command_is_marked_fresh_with_its_age(self):
        now, commands = self._commands(age=0.001)
        snap = joint_snapshot(None, {}, commands, now)
        self.assertTrue(snap["left"]["fresh"])
        self.assertEqual(snap["left"]["command_age_s"], 0.001)
        self.assertEqual(snap["left"]["command"]["gripper"], 0.4)

    def test_a_stale_command_is_marked_stale(self):
        # This is what makes the recorded action fall back to the measured
        # state, so it has to be visible rather than merely old.
        now, commands = self._commands(age=1.5)
        snap = joint_snapshot(None, {}, commands, now)
        self.assertFalse(snap["left"]["fresh"])
        self.assertEqual(snap["left"]["command_age_s"], 1.5)

    def test_before_any_reading_every_cell_is_empty(self):
        now, commands = self._commands()
        snap = joint_snapshot(None, {}, commands, now)
        self.assertIsNone(snap["left"]["state"]["shoulder_pan"])
        self.assertIsNone(snap["right"]["command"]["gripper"])
        self.assertIsNone(snap["right"]["command_age_s"])
        self.assertFalse(snap["right"]["fresh"])


class TestControlSteps(unittest.TestCase):
    """One list drives the terminal print and the console's instructions."""

    def test_quest_mode_starts_by_enabling_and_ends_by_quitting(self):
        steps = control_steps("quest")
        self.assertEqual(steps[0]["key"], "Y")
        self.assertEqual(steps[-1]["key"], "Q")

    def test_only_the_two_allowed_keys_say_they_can_be_pressed_here(self):
        here = {s["key"] for s in control_steps("quest") if "this page" in s["where"]}
        self.assertEqual(here, {"A", "Q"})

    def test_the_arm_moving_keys_stay_on_the_headset(self):
        where = {s["key"]: s["where"] for s in control_steps("quest")}
        for key in ("Y", "B", "X"):
            self.assertEqual(where[key], "headset")

    def test_leader_mode_names_the_keyboard_and_the_leader_arms(self):
        steps = control_steps("leader")
        where = {s["key"]: s["where"] for s in steps}
        self.assertIn("session keyboard", where["Y"])
        self.assertIn("this page", where["A"])
        self.assertTrue(any("leader arms" in s["what"] for s in steps))
        self.assertFalse(any(s["where"] == "controllers" for s in steps))

    def test_enabling_is_offered_to_the_page_only_where_there_is_no_headset(self):
        # A leader session started from the console has no headset and no
        # keyboard of its own, so the page is the only surface left for Y.
        # A Quest session has the button under the operator's thumb.
        self.assertIn(
            "this page", {s["key"]: s["where"] for s in control_steps("leader")}["Y"]
        )
        self.assertNotIn(
            "this page", {s["key"]: s["where"] for s in control_steps("quest")}["Y"]
        )

    def test_home_and_park_stay_physical_in_both_modes(self):
        for mode in ("quest", "leader"):
            where = {s["key"]: s["where"] for s in control_steps(mode)}
            for key in ("B", "X"):
                self.assertNotIn("this page", where[key], f"{key} in {mode}")

    def test_quest_mode_explains_the_grips_and_the_triggers(self):
        what = " ".join(s["what"] for s in control_steps("quest"))
        self.assertIn("grips", what)
        self.assertIn("trigger", what)

    def test_the_wrist_trim_appears_only_for_the_method_that_has_it(self):
        plain = " ".join(s["what"] for s in control_steps("quest"))
        trimmed = " ".join(s["what"] for s in control_steps("quest", method="mymethod"))
        self.assertNotIn("thumbstick", plain)
        self.assertIn("thumbstick", trimmed)

    def test_a_session_that_does_not_record_says_so_on_the_episode_key(self):
        episode = next(
            s for s in control_steps("quest", record=False) if s["key"] == "A"
        )
        self.assertIn("without recording", episode["what"])


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

    def test_status_carries_both_arms_joints_and_their_drift(self):
        status = self.monitor.status()
        self.assertEqual(status["joints"]["right"]["state"]["shoulder_pan"], 5.0)
        self.assertTrue(status["joints"]["left"]["fresh"])
        self.assertFalse(status["joints"]["right"]["fresh"])
        self.assertEqual(status["joint_drift_s"], 0.004)

    def test_status_before_the_arms_are_read_has_empty_joint_cells(self):
        self.dm.joints = None
        status = self.monitor.status()
        self.assertIsNone(status["joints"]["left"]["state"]["elbow_flex"])
        self.assertIsNone(status["joint_drift_s"])

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
