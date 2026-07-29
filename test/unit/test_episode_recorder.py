#!/usr/bin/env python3
"""Unit tests for the EpisodeRecorder state machine.

No MuJoCo, no hardware, no LeRobot dataset on disk: a fake dataset object
records every writer call, a REAL DualDataManager carries the state (pure
numpy), and tiny synthetic 8x6x3 images are pushed via ``set_rgb_image`` from
a background pusher thread. The recorder loop runs at a high fps so each test
finishes in well under a second of recording.

Covers: frame keys + task, save exactly once, DISABLED discard without park,
shutdown discard with park, HOMING->ENABLED no-discard, action fallback for a
stale command, camera-staleness discard without park, and start rejected while
SAVING.

Run via: python -m unittest test.unit.test_episode_recorder
(requires PYTHONPATH=.:src, as set by `source setup.sh`).
"""

import threading
import time
import unittest

import numpy as np

from common.data_manager_dual import DualDataManager, RobotActivityState
from common.recording.episode_recorder import EpisodeRecorder, RecorderState
from common.recording.features import ACTION_FRESH_S, STATE_NAMES, build_action

_CAMERAS = ["cam_a", "cam_b"]
_FPS = 100  # fast ticks so tests stay quick
_IMG = np.zeros((6, 8, 3), dtype=np.uint8)


class FakeDataset:
    """Records writer calls; save_episode can be made to block on an event."""

    num_episodes = 0

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.save_calls = 0
        self.clear_calls = 0
        self.finalized = False
        self.save_gate: threading.Event | None = None

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self, *args, **kwargs) -> None:
        if self.save_gate is not None:
            self.save_gate.wait(timeout=5.0)
        self.save_calls += 1

    def clear_episode_buffer(self, *args, **kwargs) -> None:
        self.clear_calls = self.clear_calls + 1
        self.frames = []

    def finalize(self) -> None:
        self.finalized = True


class FramePusher:
    """Background thread feeding synthetic camera frames into the manager."""

    def __init__(self, dm: DualDataManager, names: list[str]) -> None:
        self.dm = dm
        self.names = names
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            for name in self.names:
                self.dm.set_rgb_image(_IMG, name)
            time.sleep(0.002)


def _wait_for(predicate, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class RecorderTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.dm = DualDataManager()
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED)
        self.dm.set_current_joint_angles(np.arange(10, dtype=np.float64))
        self.dm.set_current_gripper_open_value("left", 0.4)
        self.dm.set_current_gripper_open_value("right", 0.6)
        self.dataset = FakeDataset()
        self.park_calls = 0

        def park() -> None:
            self.park_calls += 1

        self.recorder = EpisodeRecorder(
            dataset=self.dataset,
            data_manager=self.dm,
            task="fold the towel",
            fps=_FPS,
            camera_names=list(_CAMERAS),
            sidecar=None,
            park_arms=park,
            camera_stale_s=0.5,
        )
        self.pusher = FramePusher(self.dm, _CAMERAS)
        self.pusher.start()
        # Let the first frames land so the freshness gate opens.
        self.assertTrue(
            _wait_for(
                lambda: all(self.dm.get_rgb_image_age(n) is not None for n in _CAMERAS)
            )
        )
        self.recorder.start()

    def tearDown(self) -> None:
        self.recorder.shutdown()
        self.pusher.stop()

    def _start_and_wait_recording(self) -> None:
        self.assertTrue(self.recorder.request_start_episode())
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.RECORDING)
        )

    def _record_some_frames(self, n: int = 3) -> None:
        self._start_and_wait_recording()
        self.assertTrue(_wait_for(lambda: len(self.dataset.frames) >= n))


class TestEpisodeRecorder(RecorderTestBase):
    def test_frames_have_all_feature_keys_and_task(self) -> None:
        self._record_some_frames()
        frame = self.dataset.frames[0]
        expected = {
            "observation.state",
            "action",
            "task",
            *(f"observation.images.{n}" for n in _CAMERAS),
        }
        self.assertEqual(set(frame), expected)
        self.assertEqual(frame["task"], "fold the towel")
        self.assertEqual(frame["observation.state"].shape, (len(STATE_NAMES),))
        self.assertEqual(frame["observation.state"].dtype, np.float32)
        self.assertEqual(frame["action"].shape, (len(STATE_NAMES),))
        self.assertEqual(frame["observation.images.cam_a"].shape, (6, 8, 3))
        # timestamp/frame_index must NEVER be present (LeRobot derives them).
        self.assertNotIn("timestamp", frame)
        self.assertNotIn("frame_index", frame)

    def test_stop_saves_exactly_once(self) -> None:
        self._record_some_frames()
        self.assertTrue(self.recorder.request_stop_save())
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )
        self.assertEqual(self.dataset.save_calls, 1)
        self.assertEqual(self.dataset.clear_calls, 0)
        # A second stop is rejected (recorder is IDLE now).
        self.assertFalse(self.recorder.request_stop_save())
        self.assertEqual(self.dataset.save_calls, 1)

    def test_disabled_discards_without_park(self) -> None:
        self._record_some_frames()
        self.dm.set_robot_activity_state(RobotActivityState.DISABLED)
        self.assertTrue(_wait_for(lambda: self.dataset.clear_calls == 1))
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )
        self.assertEqual(self.dataset.save_calls, 0)
        self.assertEqual(self.park_calls, 0, "park must NOT run for DISABLED")

    def test_shutdown_discards_with_park(self) -> None:
        self._record_some_frames()
        self.dm.request_shutdown()
        self.assertTrue(_wait_for(lambda: self.dataset.clear_calls == 1))
        self.assertTrue(_wait_for(lambda: self.park_calls == 1))
        self.assertEqual(self.dataset.save_calls, 0)

    def test_homing_to_enabled_does_not_discard(self) -> None:
        self._record_some_frames()
        # B-button style: HOMING then back to ENABLED, recorder untouched.
        self.dm.set_robot_activity_state(RobotActivityState.HOMING)
        time.sleep(0.1)
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED)
        time.sleep(0.1)
        self.assertEqual(self.recorder.get_state(), RecorderState.RECORDING)
        self.assertEqual(self.dataset.clear_calls, 0)
        self.assertEqual(self.dataset.save_calls, 0)

    def test_action_fallback_when_command_stale(self) -> None:
        # A command older than the freshness window must be ignored: the
        # action falls back to the measured state.
        stale_t = time.monotonic() - 10 * ACTION_FRESH_S
        self.dm.set_last_sent_command("left", np.full(5, 99.0), 0.9, stale_t)
        self._record_some_frames()
        frame = self.dataset.frames[-1]
        np.testing.assert_allclose(frame["action"], frame["observation.state"])

    def test_action_uses_fresh_command(self) -> None:
        self._start_and_wait_recording()
        # Keep the command fresh while a few frames are recorded.
        deadline = time.monotonic() + 1.0
        while len(self.dataset.frames) < 5 and time.monotonic() < deadline:
            self.dm.set_last_sent_command(
                "left", np.full(5, 99.0), 0.9, time.monotonic()
            )
            time.sleep(0.002)
        self.assertGreaterEqual(len(self.dataset.frames), 5)
        frame = self.dataset.frames[-1]
        np.testing.assert_allclose(frame["action"][:5], np.full(5, 99.0))
        self.assertAlmostEqual(float(frame["action"][5]), 0.9, places=5)
        # Right side had no command: falls back to measured state.
        np.testing.assert_allclose(frame["action"][6:], frame["observation.state"][6:])

    def test_build_action_unit(self) -> None:
        # Direct check of the pure builder (no threads involved).
        state = np.arange(12, dtype=np.float32)
        now = 100.0
        cmds = {
            "left": (np.full(5, 7.0), 0.5, now - 0.001),  # fresh
            "right": (np.full(5, 3.0), 0.2, now - 1.0),  # stale
        }
        action = build_action(state, cmds, now)
        np.testing.assert_allclose(action[:5], 7.0)
        self.assertAlmostEqual(float(action[5]), 0.5)
        np.testing.assert_allclose(action[6:], state[6:])

    def test_start_rejected_while_saving(self) -> None:
        self.dataset.save_gate = threading.Event()  # blocks save_episode
        self._record_some_frames()
        self.assertTrue(self.recorder.request_stop_save())
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.SAVING)
        )
        self.assertFalse(self.recorder.request_start_episode())
        self.dataset.save_gate.set()  # unblock so tearDown can shut down
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )


class TestCameraStalenessDiscard(RecorderTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Tighten the staleness tolerance so the test does not wait 0.5 s.
        self.recorder.camera_stale_s = 0.06

    def test_stale_camera_discards_without_park(self) -> None:
        self._record_some_frames()
        self.pusher.stop()  # camera "unplugged": frames stop arriving
        self.assertTrue(_wait_for(lambda: self.dataset.clear_calls == 1))
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )
        self.assertEqual(self.dataset.save_calls, 0)
        self.assertEqual(self.park_calls, 0, "park must NOT run for camera staleness")
        # New episodes are refused until the streams are fresh again.
        self.recorder.request_start_episode()
        time.sleep(0.1)
        self.assertEqual(self.recorder.get_state(), RecorderState.IDLE)


if __name__ == "__main__":
    unittest.main()
