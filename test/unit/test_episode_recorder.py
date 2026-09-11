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
from common.recording.features import (
    ACTION_FRESH_S,
    STATE_NAMES,
    TELEOP_ACTIVE_KEY,
    build_action,
)

_CAMERAS = ["cam_a", "cam_b"]
_FPS = 100  # fast ticks so tests stay quick
_IMG = np.zeros((6, 8, 3), dtype=np.uint8)


class FakeDataset:
    """Records writer calls; save_episode can be made to block, or to fail.

    ``num_episodes`` advances only on a successful save, as a real dataset's
    does: it is the dataset, not the recorder, that numbers episodes.
    """

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.save_calls = 0
        self.clear_calls = 0
        self.finalized = False
        self.num_episodes = 0
        self.save_gate: threading.Event | None = None
        self.save_error: "Exception | None" = None

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self, *args, **kwargs) -> None:
        if self.save_gate is not None:
            self.save_gate.wait(timeout=5.0)
        self.save_calls += 1
        if self.save_error is not None:
            raise self.save_error
        self.num_episodes += 1

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
            TELEOP_ACTIVE_KEY,
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

    def test_teleop_active_flag_recorded(self) -> None:
        # Flag mirrors DualDataManager.get_teleop_active() at record time.
        self.dm.set_teleop_state(True)
        self._record_some_frames()
        frame = self.dataset.frames[-1]
        self.assertEqual(frame[TELEOP_ACTIVE_KEY].shape, (1,))
        self.assertEqual(float(frame[TELEOP_ACTIVE_KEY][0]), 1.0)

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


class TestSaveOnQuit(unittest.TestCase):
    """Pressing A (save) then Q (quit) must keep the episode, not lose it."""

    def _make(self, gate: bool = False):
        dm = DualDataManager()
        dm.set_robot_activity_state(RobotActivityState.ENABLED)
        dm.set_current_joint_angles(np.arange(10, dtype=np.float64))
        dm.set_current_gripper_open_value("left", 0.4)
        dm.set_current_gripper_open_value("right", 0.6)
        dataset = FakeDataset()
        if gate:
            dataset.save_gate = threading.Event()
        recorder = EpisodeRecorder(
            dataset=dataset,
            data_manager=dm,
            task="t",
            fps=_FPS,
            camera_names=list(_CAMERAS),
            sidecar=None,
        )
        return dm, dataset, recorder

    def test_pending_save_wins_over_shutdown_discard(self) -> None:
        # A then Q, with the quit arriving before the loop processes the save:
        # the requested save must win over the shutdown-discard trigger. Driven
        # single-threaded (no start()) so the ordering is deterministic.
        dm, dataset, recorder = self._make()
        with recorder._lock:
            recorder._state = RecorderState.RECORDING
        self.assertTrue(recorder.request_stop_save())  # A: sets _pending_stop
        dm.request_shutdown()  # Q: arrives after A
        recorder._step_recording()  # one loop tick
        self.assertEqual(dataset.save_calls, 1)
        self.assertEqual(dataset.clear_calls, 0, "episode must be saved, not discarded")
        self.assertEqual(recorder.get_state(), RecorderState.IDLE)

    def test_shutdown_waits_for_in_flight_save(self) -> None:
        # The cube-pnp failure: a long save is still encoding when shutdown runs.
        # shutdown() must not finalize until the save finishes, or the episode is
        # lost (videos unencoded, info.json never updated).
        dm, dataset, recorder = self._make(gate=True)
        pusher = FramePusher(dm, _CAMERAS)
        pusher.start()
        self.assertTrue(
            _wait_for(
                lambda: all(dm.get_rgb_image_age(n) is not None for n in _CAMERAS)
            )
        )
        recorder.start()
        try:
            self.assertTrue(recorder.request_start_episode())
            self.assertTrue(_wait_for(lambda: len(dataset.frames) >= 2))
            self.assertTrue(recorder.request_stop_save())
            self.assertTrue(
                _wait_for(lambda: recorder.get_state() == RecorderState.SAVING)
            )
            done = threading.Event()

            def _run_shutdown() -> None:
                recorder.shutdown()
                done.set()

            threading.Thread(target=_run_shutdown, daemon=True).start()
            time.sleep(0.2)  # shutdown must be blocking on the save
            self.assertFalse(dataset.finalized, "finalize ran before the save finished")
            self.assertFalse(
                done.is_set(), "shutdown returned before the save finished"
            )
            assert dataset.save_gate is not None
            dataset.save_gate.set()  # let the save complete
            self.assertTrue(done.wait(timeout=5.0))
            self.assertEqual(dataset.save_calls, 1)
            self.assertEqual(dataset.clear_calls, 0, "saved, not discarded")
            self.assertTrue(dataset.finalized)
        finally:
            if dataset.save_gate is not None:
                dataset.save_gate.set()
            pusher.stop()


class FakeDepthWriter:
    """Records DepthWriter lifecycle calls (no disk, no hardware)."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.begun: list[int] = []
        self.added: list[int] = []
        self.ended = 0
        self.aborted = 0
        self.dropped = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def begin_episode(self, ep_idx: int) -> None:
        self.begun.append(ep_idx)

    def add(self, frame_index: int, depth_by_stream: dict) -> None:
        self.added.append(frame_index)

    def end_episode(self) -> None:
        self.ended += 1

    def abort_episode(self) -> None:
        self.aborted += 1


class SpySidecar:
    """Records the sidecar lifecycle an episode goes through."""

    def __init__(self) -> None:
        self.ended: list[int] = []
        self.aborted = 0

    def begin_episode(self, *a, **k) -> None:
        pass

    def end_episode(self, index: int) -> None:
        self.ended.append(index)

    def abort_episode(self) -> None:
        self.aborted += 1

    def stop(self) -> None:
        pass


class TestSaveThatFails(RecorderTestBase):
    """A save that raises has not produced an episode, and must not pretend.

    This is the fault that leaves a dataset counting an episode nobody wrote,
    which LeRobot cannot open at all: it judges the whole local copy incomplete
    and goes to the Hub for a version tag (see actoris_harena.recording.dataset_check).
    The recorder must therefore not advance its numbering past an episode that
    was not written, nor leave side files named after it.
    """

    def setUp(self) -> None:
        super().setUp()
        self.sidecar = SpySidecar()
        self.recorder.sidecar = self.sidecar
        self.depth = FakeDepthWriter()
        self.recorder.depth_writer = self.depth

    def _record_one(self) -> None:
        self._record_some_frames()
        self.assertTrue(self.recorder.request_stop_save())
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )

    def test_a_written_episode_advances_the_count(self):
        self._record_one()

        self.assertEqual(self.dataset.num_episodes, 1)
        self.assertEqual(self.recorder.get_episode_count(), 1)
        self.assertEqual(self.sidecar.ended, [0])

    def test_a_failed_save_does_not_advance_the_count(self):
        self.dataset.save_error = OSError("no space left on device")

        self._record_one()

        self.assertEqual(self.dataset.num_episodes, 0)
        self.assertEqual(self.recorder.get_episode_count(), 0)

    def test_a_failed_save_leaves_no_side_files_behind(self):
        self.dataset.save_error = OSError("no space left on device")

        self._record_one()

        self.assertEqual(self.sidecar.ended, [])
        self.assertEqual(self.sidecar.aborted, 1)
        self.assertEqual(self.depth.ended, 0)
        self.assertEqual(self.depth.aborted, 1)

    def test_the_next_episode_takes_the_index_the_failed_one_did_not(self):
        self.dataset.save_error = OSError("no space left on device")
        self._record_one()
        self.dataset.save_error = None

        self._record_one()

        self.assertEqual(self.sidecar.ended, [0])
        self.assertEqual(self.recorder.get_episode_count(), 1)


class TestDepthWiring(unittest.TestCase):
    """Depth streams are sampled per frame and handed to the writer 1:1."""

    def setUp(self) -> None:
        self.dm = DualDataManager()
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED)
        self.dm.set_current_joint_angles(np.arange(10, dtype=np.float64))
        self.dm.set_current_gripper_open_value("left", 0.4)
        self.dm.set_current_gripper_open_value("right", 0.6)
        self.dataset = FakeDataset()
        self.depth_writer = FakeDepthWriter()
        self.recorder = EpisodeRecorder(
            dataset=self.dataset,
            data_manager=self.dm,
            task="t",
            fps=_FPS,
            camera_names=list(_CAMERAS),
            sidecar=None,
            depth_streams=["central_depth"],
            depth_writer=self.depth_writer,
        )
        self.pusher = FramePusher(self.dm, _CAMERAS)
        self.pusher.start()
        self._stop_depth = threading.Event()

        def push_depth() -> None:
            while not self._stop_depth.is_set():
                self.dm.set_depth_image(
                    np.zeros((6, 8), dtype=np.uint16), "central_depth"
                )
                time.sleep(0.002)

        self._depth_thread = threading.Thread(target=push_depth, daemon=True)
        self._depth_thread.start()
        self.assertTrue(
            _wait_for(
                lambda: all(self.dm.get_rgb_image_age(n) is not None for n in _CAMERAS)
                and self.dm.get_depth_image_age("central_depth") is not None
            )
        )
        self.recorder.start()

    def tearDown(self) -> None:
        self.recorder.shutdown()
        self._stop_depth.set()
        self._depth_thread.join(timeout=2.0)
        self.pusher.stop()

    def test_depth_written_with_zero_based_indices_and_saved(self) -> None:
        self.assertTrue(self.depth_writer.started)
        self.assertTrue(self.recorder.request_start_episode())
        self.assertTrue(_wait_for(lambda: len(self.depth_writer.added) >= 3))
        self.assertTrue(self.recorder.request_stop_save())
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.IDLE)
        )
        # begin was called with the episode index; frame indices are 0-based
        # and match the dataset add_frame count.
        self.assertEqual(self.depth_writer.begun[0], 0)
        self.assertEqual(self.depth_writer.added[0], 0)
        self.assertEqual(len(self.depth_writer.added), len(self.dataset.frames))
        self.assertEqual(self.depth_writer.ended, 1)
        self.assertEqual(self.depth_writer.aborted, 0)


class TestEeWiring(unittest.TestCase):
    """record_ee samples measured + target EE (own base frame) per frame."""

    def _make(self, push_target: bool):
        dm = DualDataManager()
        dm.set_robot_activity_state(RobotActivityState.ENABLED)
        dm.set_current_joint_angles(np.arange(10, dtype=np.float64))
        dm.set_current_gripper_open_value("left", 0.4)
        dm.set_current_gripper_open_value("right", 0.6)
        dataset = FakeDataset()
        recorder = EpisodeRecorder(
            dataset=dataset,
            data_manager=dm,
            task="t",
            fps=_FPS,
            camera_names=list(_CAMERAS),
            sidecar=None,
            record_ee=True,
        )
        pusher = FramePusher(dm, _CAMERAS)
        pusher.start()
        stop = threading.Event()

        def push_ee() -> None:
            measured = np.eye(4)
            measured[:3, 3] = [0.3, 0.0, 0.2]
            target = np.eye(4)
            target[:3, 3] = [0.35, 0.05, 0.25]  # distinct from measured
            while not stop.is_set():
                for side in ("left", "right"):
                    dm.set_current_end_effector_pose(side, measured)
                    if push_target:
                        dm.set_target_pose(side, target)
                time.sleep(0.002)

        ee_thread = threading.Thread(target=push_ee, daemon=True)
        ee_thread.start()
        return dm, dataset, recorder, pusher, stop, ee_thread

    def test_ee_features_present_and_target_used(self):
        dm, dataset, recorder, pusher, stop, ee_thread = self._make(push_target=True)
        try:
            self.assertTrue(
                _wait_for(
                    lambda: all(dm.get_rgb_image_age(n) is not None for n in _CAMERAS)
                    and dm.get_current_end_effector_pose_at("left", time.monotonic())
                    is not None
                )
            )
            recorder.start()
            self.assertTrue(recorder.request_start_episode())
            self.assertTrue(_wait_for(lambda: len(dataset.frames) >= 3))
            frame = dataset.frames[-1]
            self.assertEqual(frame["ee_pose"].shape, (14,))
            self.assertEqual(frame["ee_target"].shape, (14,))
            # Fresh, distinct target → ee_target differs from measured ee_pose.
            self.assertFalse(np.allclose(frame["ee_target"], frame["ee_pose"]))
        finally:
            recorder.shutdown()
            stop.set()
            ee_thread.join(timeout=2.0)
            pusher.stop()

    def test_ee_target_falls_back_to_measured_when_no_target(self):
        dm, dataset, recorder, pusher, stop, ee_thread = self._make(push_target=False)
        try:
            self.assertTrue(
                _wait_for(
                    lambda: all(dm.get_rgb_image_age(n) is not None for n in _CAMERAS)
                    and dm.get_current_end_effector_pose_at("left", time.monotonic())
                    is not None
                )
            )
            recorder.start()
            self.assertTrue(recorder.request_start_episode())
            self.assertTrue(_wait_for(lambda: len(dataset.frames) >= 3))
            frame = dataset.frames[-1]
            # No target ever published → ee_target falls back to measured pose.
            np.testing.assert_allclose(frame["ee_target"], frame["ee_pose"])
        finally:
            recorder.shutdown()
            stop.set()
            ee_thread.join(timeout=2.0)
            pusher.stop()


class SpyAudioCue:
    """Records which cues were asked for, so the operator-facing signal is testable."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def play(self, event: str) -> None:
        self.events.append(event)


class TestCameraStalenessPause(RecorderTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Tighten the tolerances so the test does not wait whole seconds.
        self.recorder.camera_stale_s = 0.06
        self.recorder.camera_dead_s = 0.4
        self.cue = SpyAudioCue()
        self.recorder.audio_cue = self.cue

    def _unplug(self) -> None:
        """Stop the synthetic frame feed, as a camera dropping off the bus would."""
        self.pusher.stop()

    def _replug(self) -> None:
        self.pusher = FramePusher(self.dm, _CAMERAS)
        self.pusher.start()

    def test_stale_camera_pauses_instead_of_discarding(self) -> None:
        self._record_some_frames()
        self._unplug()
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.PAUSED)
        )
        # The episode is HELD, not thrown away.
        self.assertEqual(self.dataset.clear_calls, 0)
        self.assertEqual(self.dataset.save_calls, 0)
        self.assertEqual(self.park_calls, 0)

    def test_paused_episode_resumes_when_the_camera_returns(self) -> None:
        self._record_some_frames()
        self._unplug()
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.PAUSED)
        )
        frames_at_pause = len(self.dataset.frames)
        self._replug()
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.RECORDING)
        )
        # Same episode, still growing: nothing was discarded or saved in between.
        self.assertTrue(_wait_for(lambda: len(self.dataset.frames) > frames_at_pause))
        self.assertEqual(self.dataset.clear_calls, 0)
        self.assertEqual(self.dataset.save_calls, 0)
        # The recovery is reported so the gap cannot reach training unnoticed.
        self.assertEqual(self.recorder._pause_count, 1)
        self.assertGreater(self.recorder._paused_total_s, 0.0)

    def test_paused_episode_abandoned_after_dead_timeout_without_park(self) -> None:
        self._record_some_frames()
        self._unplug()
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

    def test_automatic_end_plays_the_stop_cue(self) -> None:
        # An operator watching the arms must hear that recording ended by itself.
        self._record_some_frames()
        self.assertEqual(self.cue.events, ["start"])
        self._unplug()
        self.assertTrue(_wait_for(lambda: self.dataset.clear_calls == 1))
        self.assertEqual(self.cue.events, ["start", "stop"])

    def test_stop_save_while_paused_keeps_the_episode(self) -> None:
        self._record_some_frames()
        self._unplug()
        self.assertTrue(
            _wait_for(lambda: self.recorder.get_state() == RecorderState.PAUSED)
        )
        self.assertTrue(self.recorder.request_stop_save())
        self.assertTrue(_wait_for(lambda: self.dataset.save_calls == 1))
        self.assertEqual(self.dataset.clear_calls, 0)
        self.assertEqual(self.cue.events, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
