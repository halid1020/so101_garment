"""Live sensor view rendered inside the real-teleop process.

Shows the tactile-camera feeds and both arms' joint state while
teleoperation runs (``tool/meta_quest_teleopration.py --sensor-view``).
Everything displayed is data the teleop process already produces: the
joint-state threads publish measured/commanded joints into
``DualDataManager`` at 100 Hz and the ``CameraCapture`` threads publish
RGB frames via ``set_rgb_image`` — the viewer adds no serial traffic
and opens no extra devices.

The loop runs on the MAIN thread (which otherwise just sleeps), so all
OpenCV HighGUI calls stay on one thread. q/Esc closes the window and
returns to the caller; teleoperation itself is untouched.
"""

from __future__ import annotations

import time
from collections import deque

import cv2  # type: ignore[import]
import numpy as np

from common.data_manager_dual import DualDataManager

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# 10-DOF joint-vector layout used throughout the teleop stack.
_SIDE_SLICE = {"left": slice(0, 5), "right": slice(5, 10)}
_PANEL_W, _PANEL_H = 320, 480


class FrameRateCounter:
    """Counts frame-object changes to estimate a stream's live Hz.

    ``set_rgb_image`` stores a fresh array object per frame, so object
    identity change == new frame (no pixel comparison needed).
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._last_obj: object | None = None
        self._stamps: deque[float] = deque()

    def tick(self, frame: object, now: float | None = None) -> None:
        if frame is None or frame is self._last_obj:
            return
        self._last_obj = frame
        self._stamps.append(time.monotonic() if now is None else now)

    def hz(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        while self._stamps and self._stamps[0] < now - self.window_s:
            self._stamps.popleft()
        return len(self._stamps) / self.window_s


def build_arm_panel_lines(
    side: str,
    measured: "np.ndarray | None",
    target: "np.ndarray | None",
    grip_meas: "float | None",
    grip_target: "float | None",
    activity: str,
    teleop_active: bool,
) -> list[str]:
    """Text lines for one arm's panel (pure — unit-tested).

    ``measured``/``target`` are the 10-DOF URDF-degree vectors from the
    data manager (left joints 0-4, right 5-9), or None before the
    first publish.
    """

    def fmt(vec: "np.ndarray | None", i: int) -> str:
        if vec is None:
            return "     --"
        return f"{float(vec[_SIDE_SLICE[side]][i]):+7.1f}"

    def fgrip(value: "float | None") -> str:
        return "  --" if value is None else f"{value:4.2f}"

    lines = [
        f"{side} arm   [{activity}{' teleop' if teleop_active else ''}]",
        f"{'joint':<14}{'meas':>8}{'cmd':>8}",
    ]
    for i, joint in enumerate(_JOINT_NAMES):
        lines.append(f"{joint:<14}{fmt(measured, i):>8}{fmt(target, i):>8}")
    lines.append(f"{'gripper':<14}{fgrip(grip_meas):>8}{fgrip(grip_target):>8}")
    return lines


def _draw_panel(lines: list[str]) -> np.ndarray:
    panel = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
    for i, text in enumerate(lines):
        colour = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(panel, text, (8, 28 + 30 * i), _FONT, 0.5, colour, 1)
    return panel


def run_sensor_view_loop(
    data_manager: DualDataManager,
    captures: list,
    window: str = "teleop sensors",
) -> None:
    """~30 Hz view loop; returns on q/Esc or shutdown request.

    ``captures`` are started ``CameraCapture`` objects — only their
    ``name``/``width``/``height`` are read here; frames come from the
    data manager.
    """
    from tool.test_sensor_rates import grid_tiles

    counters = {cam.name: FrameRateCounter() for cam in captures}
    try:
        while not data_manager.is_shutdown_requested():
            tiles = []
            for cam in captures:
                rgb = data_manager.get_rgb_image(cam.name)
                counters[cam.name].tick(rgb)
                if rgb is None:
                    tile = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                else:
                    tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    tile,
                    f"{cam.name} {counters[cam.name].hz():.0f} Hz",
                    (8, 24),
                    _FONT,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                tiles.append(tile)

            measured = data_manager.get_current_joint_angles()
            target = data_manager.get_target_joint_angles()
            activity = data_manager.get_robot_activity_state().name
            teleop_active = data_manager.get_teleop_active()
            for side in ("left", "right"):
                lines = build_arm_panel_lines(
                    side,
                    measured,
                    target,
                    data_manager.get_current_gripper_open_value(side),
                    data_manager.get_target_gripper_open_value(side),
                    activity,
                    teleop_active,
                )
                tiles.append(_draw_panel(lines))

            cv2.imshow(window, grid_tiles(tiles, max_per_row=3))
            key = cv2.waitKey(33) & 0xFF
            if key in (27, ord("q")):
                print("👁  sensor view closed (teleop keeps running)")
                break
    finally:
        cv2.destroyAllWindows()
