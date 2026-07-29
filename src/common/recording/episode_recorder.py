"""Episode recorder: the state machine that owns the LeRobot dataset writer.

A single background thread paces at the dataset fps and is the SOLE caller of
``add_frame`` / ``save_episode`` / ``clear_episode_buffer``. Button callbacks
never touch the writer — they only post ``request_start_episode`` /
``request_stop_save`` flags, which the loop consumes.

States: IDLE -> RECORDING -> (SAVING | DISCARDING) -> IDLE.

Discard triggers while RECORDING:
* shutdown requested (``DualDataManager.is_shutdown_requested``) -> park;
* a thread error inside the loop -> park;
* robot activity became DISABLED -> NO park (a torque-off already happened and
  re-torquing unattended arms is riskier);
* an enabled camera frame staler than ``camera_stale_s`` -> NO park (a data
  problem, not a safety problem).
``park_arms`` (injected) is therefore invoked ONLY for the shutdown/thread-error
triggers. A frame staler than the tolerance but present is reused with a
throttled warning; only beyond the tolerance does the episode discard.
"""

from __future__ import annotations

import threading
import time
import traceback
from enum import Enum
from typing import Any, Callable

from common.data_manager_dual import DualDataManager, RobotActivityState
from common.recording import features as feat


class RecorderState(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    SAVING = "SAVING"
    DISCARDING = "DISCARDING"


class EpisodeRecorder:
    """Threaded, fps-paced episode recorder owning the dataset writer."""

    def __init__(
        self,
        dataset: Any,
        data_manager: DualDataManager,
        task: str,
        fps: int,
        camera_names: list[str],
        cameras: list[Any] | None = None,
        sidecar: Any | None = None,
        park_arms: Callable[[], None] | None = None,
        camera_stale_s: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.data_manager = data_manager
        self.task = task
        self.fps = int(fps)
        self.camera_names = list(camera_names)
        self.cameras = list(cameras) if cameras is not None else []
        self.sidecar = sidecar
        self.park_arms = park_arms
        self.camera_stale_s = camera_stale_s

        self._lock = threading.Lock()
        self._state = RecorderState.IDLE
        self._pending_start = False
        self._pending_stop = False
        self._stop = threading.Event()
        self._external_shutdown = False
        self._thread: threading.Thread | None = None

        # Per-episode diagnostics.
        self._tick_durations: list[float] = []
        self._frame_count = 0
        self._last_reuse_warn = 0.0
        self._last_overrun_warn = 0.0

        # Episode index tracking (kept in sync with the dataset, works on
        # resume where num_episodes > 0).
        self._episode_index = int(getattr(dataset, "num_episodes", 0) or 0)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_state(self) -> RecorderState:
        with self._lock:
            return self._state

    def request_start_episode(self) -> bool:
        """Ask the loop to start recording. Rejected unless currently IDLE."""
        with self._lock:
            if self._state != RecorderState.IDLE:
                print(f"⚠️  cannot start episode: recorder is {self._state.value}")
                return False
            self._pending_start = True
            return True

    def request_stop_save(self) -> bool:
        """Ask the loop to stop and save. Rejected unless currently RECORDING."""
        with self._lock:
            if self._state != RecorderState.RECORDING:
                print(f"⚠️  cannot stop episode: recorder is {self._state.value}")
                return False
            self._pending_stop = True
            return True

    def start(self) -> None:
        """Start the sidecar, camera threads and the record loop."""
        if self.sidecar is not None:
            self.sidecar.start()
        for cam in self.cameras:
            cam.start(self.data_manager)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Discard any in-flight episode, stop everything, finalize the dataset.

        The in-flight discard runs on the loop thread (sole writer owner); this
        method signals it, joins, then tears down the auxiliary threads and
        finalizes so the parquet footers are written.
        """
        self._external_shutdown = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self.sidecar is not None:
            self.sidecar.stop()
        for cam in self.cameras:
            cam.stop()
        try:
            self.dataset.finalize()
        except Exception:
            traceback.print_exc()

    # ── Record loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        from lerobot.utils.robot_utils import precise_sleep

        dt = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                tick_start = time.perf_counter()
                self._step()
                elapsed = time.perf_counter() - tick_start
                if self.get_state() == RecorderState.RECORDING:
                    self._tick_durations.append(elapsed)
                    if elapsed > dt:
                        self._warn_overrun(elapsed, dt)
                precise_sleep(dt - elapsed)
        except Exception:
            print("❌ record loop crashed; discarding in-flight episode")
            traceback.print_exc()
            self.data_manager.request_shutdown()
            if self.get_state() == RecorderState.RECORDING:
                self._discard(reason="thread_error", park=True)
        finally:
            # External shutdown while an episode is in flight -> discard + park.
            if self.get_state() == RecorderState.RECORDING:
                self._discard(reason="shutdown", park=self._external_shutdown)

    def _step(self) -> None:
        state = self.get_state()
        if state == RecorderState.IDLE:
            self._step_idle()
        elif state == RecorderState.RECORDING:
            self._step_recording()
        # SAVING / DISCARDING are handled inline where they are entered.

    def _step_idle(self) -> None:
        with self._lock:
            start = self._pending_start
            self._pending_start = False
        if not start:
            return
        if not self._all_cameras_fresh():
            print("⚠️  cannot start episode: a camera stream is not fresh")
            return
        if self.sidecar is not None:
            self.sidecar.begin_episode()
        self._tick_durations = []
        self._frame_count = 0
        with self._lock:
            self._state = RecorderState.RECORDING
        print(f"🔴 recording episode {self._episode_index} (task: {self.task!r})")

    def _step_recording(self) -> None:
        # Discard triggers, highest priority first.
        if self.data_manager.is_shutdown_requested():
            self._discard(reason="shutdown", park=True)
            self._stop.set()
            return
        if self.data_manager.get_robot_activity_state() == RobotActivityState.DISABLED:
            self._discard(reason="disabled", park=False)
            return
        stale = self._stale_camera()
        if stale is not None:
            self._discard(reason=f"camera_stale:{stale}", park=False)
            return

        with self._lock:
            stop = self._pending_stop
            self._pending_stop = False
        if stop:
            with self._lock:
                self._state = RecorderState.SAVING
            self._save()
            return

        self._record_frame()

    # ── Frame building ───────────────────────────────────────────────────────

    def _record_frame(self) -> None:
        dm = self.data_manager
        measured = dm.get_current_joint_angles()
        if measured is None or len(measured) < feat.BODY_DOF * len(feat.SIDES):
            return  # no joint state yet; skip this tick
        gripper_open = {
            side: (dm.get_current_gripper_open_value(side) or 0.0)
            for side in feat.SIDES
        }
        state = feat.build_observation_state(measured, gripper_open)
        last_commands = {side: dm.get_last_sent_command(side) for side in feat.SIDES}
        action = feat.build_action(state, last_commands, time.monotonic())

        images = {}
        for name in self.camera_names:
            img = dm.get_rgb_image(name)
            if img is None:
                return  # guarded by staleness, but never add a None frame
            images[name] = img

        frame = feat.assemble_frame(state, action, images, self.task)
        self.dataset.add_frame(frame)
        self._frame_count += 1

    # ── Terminal transitions ─────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self.dataset.save_episode()
        except Exception:
            traceback.print_exc()
        if self.sidecar is not None:
            self.sidecar.end_episode(self._episode_index)
        self._print_stats(outcome="saved")
        self._episode_index += 1
        with self._lock:
            self._state = RecorderState.IDLE

    def _discard(self, reason: str, park: bool) -> None:
        with self._lock:
            self._state = RecorderState.DISCARDING
        try:
            self.dataset.clear_episode_buffer()
        except Exception:
            traceback.print_exc()
        if self.sidecar is not None:
            self.sidecar.abort_episode()
        self._print_stats(outcome=f"discarded ({reason})")
        if park and self.park_arms is not None:
            try:
                self.park_arms()
            except Exception:
                traceback.print_exc()
        with self._lock:
            self._state = RecorderState.IDLE

    # ── Camera freshness ─────────────────────────────────────────────────────

    def _all_cameras_fresh(self) -> bool:
        now = time.monotonic()
        for name in self.camera_names:
            age = self.data_manager.get_rgb_image_age(name, now)
            if age is None or age > self.camera_stale_s:
                return False
        return True

    def _stale_camera(self) -> str | None:
        """Return the first camera stale beyond tolerance, else None.

        A present-but-slightly-stale frame (within tolerance) is reused with a
        throttled warning; only beyond the tolerance is a camera reported.
        """
        now = time.monotonic()
        reuse_threshold = 1.5 / self.fps
        worst_reuse = 0.0
        for name in self.camera_names:
            age = self.data_manager.get_rgb_image_age(name, now)
            if age is None or age > self.camera_stale_s:
                return name
            if age > reuse_threshold:
                worst_reuse = max(worst_reuse, age)
        if worst_reuse > 0.0 and now - self._last_reuse_warn > 1.0:
            self._last_reuse_warn = now
            print(
                f"⚠️  reusing stale camera frame ({worst_reuse * 1e3:.0f} ms old; "
                f"tolerance {self.camera_stale_s * 1e3:.0f} ms)"
            )
        return None

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _warn_overrun(self, elapsed: float, dt: float) -> None:
        now = time.monotonic()
        if now - self._last_overrun_warn > 1.0:
            self._last_overrun_warn = now
            print(
                f"⚠️  record tick overran: {elapsed * 1e3:.1f} ms > "
                f"{dt * 1e3:.1f} ms budget"
            )

    def _print_stats(self, outcome: str) -> None:
        n = len(self._tick_durations)
        if n:
            avg = sum(self._tick_durations) / n * 1e3
            worst = max(self._tick_durations) * 1e3
            tick_msg = f"avg tick {avg:.1f} ms, worst {worst:.1f} ms"
        else:
            tick_msg = "no ticks"
        print(
            f"⏹️  episode {self._episode_index} {outcome}: "
            f"{self._frame_count} frames, {tick_msg}"
        )
