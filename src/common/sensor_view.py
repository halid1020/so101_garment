"""Live sensor view rendered inside the real-teleop process.

Shows the tactile-camera feeds and both arms' joint state while
teleoperation runs (``tool/meta_quest_teleopration.py --sensor-view``),
using the SAME layout as ``tool/test_sensor_rates.py --view``: a row of
tactile cameras above one cell per side, each cell showing that side's
two joint columns in a large font.

Everything displayed is data the teleop process already produces — the
joint-state threads publish measured joints into ``DualDataManager`` at
100 Hz, the leader thread publishes its mapped joints, and the
``CameraCapture`` threads publish RGB via ``set_rgb_image`` — so the
viewer adds no serial traffic and opens no extra devices.

The columns adapt to the input mode: in leader mode they are the
follower (measured) and the leader (mapped); in Quest mode they are the
follower measured and the commanded target.

The loop runs on the MAIN thread (which otherwise just sleeps), so all
OpenCV HighGUI calls stay on one thread. When ``key_callbacks`` is given
(leader mode, where this window is the control surface) the loop
dispatches those keys to the teleop callbacks and stays open until
shutdown; otherwise q/Esc closes the window and teleop keeps running.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

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


class FrameRateCounter:
    """Counts frame-object changes to estimate a stream's live Hz and drops.

    ``set_rgb_image`` stores a fresh array object per frame, so object
    identity change == new frame (no pixel comparison needed). When
    ``expected_hz`` is given, an inter-frame gap longer than ~1.5 nominal
    periods is counted as dropped frame(s) so the live monitor can flag a
    stream that is silently skipping frames (USB bandwidth starvation).
    """

    def __init__(self, window_s: float = 2.0, expected_hz: float = 0.0) -> None:
        self.window_s = window_s
        self._expected_dt = 1.0 / expected_hz if expected_hz > 0 else 0.0
        self._last_obj: object | None = None
        self._last_stamp: float | None = None
        self._stamps: deque[float] = deque()
        self.drops = 0

    def tick(self, frame: object, now: float | None = None) -> None:
        if frame is None or frame is self._last_obj:
            return
        now = time.monotonic() if now is None else now
        if (
            self._expected_dt > 0.0
            and self._last_stamp is not None
            and (now - self._last_stamp) > 1.5 * self._expected_dt
        ):
            self.drops += int(round((now - self._last_stamp) / self._expected_dt)) - 1
        self._last_obj = frame
        self._last_stamp = now
        self._stamps.append(now)

    def hz(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        while self._stamps and self._stamps[0] < now - self.window_s:
            self._stamps.popleft()
        return len(self._stamps) / self.window_s


def _age_color(age_s: float | None) -> tuple[int, int, int]:
    """BGR colour for a stream age/drift: green ≤ ½ frame, amber ≤ 1 frame, red."""
    if age_s is None:
        return (0, 0, 255)
    if age_s <= 0.016:
        return (0, 255, 0)
    if age_s <= 0.033:
        return (0, 210, 255)
    return (0, 0, 255)


def _vec5_to_dict(vec5: "np.ndarray | None", gripper: "float | None") -> dict:
    """{joint_name: value|None} for the five body joints plus the gripper."""
    if vec5 is None:
        d: dict = {name: None for name in _JOINT_NAMES}
    else:
        d = {name: float(vec5[i]) for i, name in enumerate(_JOINT_NAMES)}
    d["gripper"] = float(gripper) if gripper is not None else None
    return d


def side_joint_dict(
    vec10: "np.ndarray | None", side: str, gripper: "float | None"
) -> dict:
    """One side's ``{joint: value|None}`` from a 10-DOF vector + gripper.

    Left joints are indices 0-4, right 5-9 (the teleop stack's layout).
    Pure — unit-tested.
    """
    seg = None if vec10 is None else vec10[_SIDE_SLICE[side]]
    return _vec5_to_dict(seg, gripper)


def _handle_view_key(
    key: int, key_callbacks: "dict[str, Callable[[], None]] | None"
) -> bool:
    """Process one waitKey. Returns True when the view should close.

    With ``key_callbacks`` (leader mode) the window is the control
    surface: mapped keys fire their callback and the view stays open
    (it closes only via a shutdown request). Without it (Quest mode)
    q/Esc close the view while teleop keeps running.
    """
    if key == 255:  # no key this tick
        return False
    if key_callbacks is not None:
        ch = " " if key == 32 else (chr(key).lower() if 32 < key < 127 else "")
        cb = key_callbacks.get(ch)
        if cb is not None:
            cb()
        return False
    if key in (27, ord("q")):
        print("👁  sensor view closed (teleop keeps running)")
        return True
    return False


def run_sensor_view_loop(
    data_manager: DualDataManager,
    captures: list,
    window: str = "teleop sensors",
    leader_mode: bool = False,
    key_callbacks: "dict[str, Callable[[], None]] | None" = None,
    max_w: int = 1280,
    max_h: int = 720,
) -> None:
    """~30 Hz view loop; returns on q/Esc (Quest) or shutdown request.

    ``captures`` are started ``CameraCapture`` objects — only their
    ``name``/``width``/``height`` are read here; frames come from the
    data manager.
    """
    from tool.test_sensor_rates import (
        _CAM_TILE_H,
        _CAM_TILE_W,
        _camera_short_label,
        _fit_to_screen,
        _side_panel,
        _vstack_pad,
        grid_tiles,
    )

    counters = {
        cam.name: FrameRateCounter(expected_hz=float(getattr(cam, "fps", 0) or 0))
        for cam in captures
    }
    col2_label = "leader" if leader_mode else "cmd"
    try:
        while not data_manager.is_shutdown_requested():
            now = time.monotonic()
            cam_tiles = []
            for cam in captures:
                rgb = data_manager.get_rgb_image(cam.name)
                counters[cam.name].tick(rgb, now)
                age = data_manager.get_rgb_image_age(cam.name, now)
                if rgb is None:
                    tile = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                else:
                    tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                tile = cv2.resize(tile, (_CAM_TILE_W, _CAM_TILE_H))
                cv2.putText(
                    tile,
                    f"{_camera_short_label(cam.name)} {counters[cam.name].hz():.0f}Hz",
                    (6, 26),
                    _FONT,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # Live drift line: age from the collection reference (now) +
                # cumulative dropped-frame count, colour-coded by frame budget.
                age_ms = "--" if age is None else f"{age * 1e3:.0f}ms"
                cv2.putText(
                    tile,
                    f"drift {age_ms}  drop {counters[cam.name].drops}",
                    (6, 52),
                    _FONT,
                    0.6,
                    _age_color(age),
                    2,
                )
                cam_tiles.append(tile)

            measured = data_manager.get_current_joint_angles()
            target = None if leader_mode else data_manager.get_target_joint_angles()
            cells = []
            for side in ("left", "right"):
                c1 = side_joint_dict(
                    measured, side, data_manager.get_current_gripper_open_value(side)
                )
                if leader_mode:
                    lvec, lgrip = data_manager.get_leader_mapped_state(side)
                    c2 = _vec5_to_dict(lvec, lgrip)
                else:
                    c2 = side_joint_dict(
                        target, side, data_manager.get_target_gripper_open_value(side)
                    )
                cells.append(
                    _side_panel(side, "follower", c1, None, col2_label, c2, None)
                )
            joint_row = np.hstack(cells)

            # Proprio drift strip: how stale the measured joints are at the
            # collection reference time (100 Hz stream → normally < 10 ms).
            joints_res = data_manager.get_current_joint_angles_at(now)
            joint_age = None if joints_res is None else joints_res[1]
            strip = np.zeros((34, joint_row.shape[1], 3), dtype=np.uint8)
            j_ms = "--" if joint_age is None else f"{joint_age * 1e3:.0f}ms"
            cv2.putText(
                strip,
                f"joints drift {j_ms}",
                (8, 24),
                _FONT,
                0.7,
                _age_color(joint_age),
                2,
            )

            blocks = []
            if cam_tiles:
                blocks.append(grid_tiles(cam_tiles, max_per_row=max(len(cam_tiles), 1)))
            blocks.append(strip)
            blocks.append(joint_row)
            cv2.imshow(window, _fit_to_screen(_vstack_pad(blocks), max_w, max_h))

            key = cv2.waitKey(33) & 0xFF
            if _handle_view_key(key, key_callbacks):
                break
    finally:
        cv2.destroyAllWindows()
