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
from dataclasses import dataclass, field
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

# Cameras that are opened for recording/teleop but NOT drawn in the live
# monitor. "scene" is hidden from the view (its tile crowded the panel without
# aiding teleoperation); it is still recorded like any other stream.
VIEW_HIDDEN_CAMERAS = {"scene"}


def visible_view_captures(captures: list, hidden: "set[str]" = VIEW_HIDDEN_CAMERAS):
    """Captures to draw in the live view, dropping ``hidden`` camera names.

    Filters the display only — hidden cameras are still recorded. Pure —
    unit-tested.
    """
    return [c for c in captures if getattr(c, "name", None) not in hidden]


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


def _age_ms(age_s: float | None) -> str:
    return "--" if age_s is None else f"{age_s * 1e3:.0f}ms"


def depth_range_from_frame(
    depth16: np.ndarray,
    scale_m: float = 0.001,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    min_span_m: float = 0.1,
) -> "tuple[float, float] | None":
    """Near/far metric range (metres) for colourising, from one depth frame.

    Uses the ``lo_pct``/``hi_pct`` percentiles of the valid (>0) pixels so a few
    stray near/far returns do not stretch the scale, then widens to at least
    ``min_span_m`` so a flat scene still spans the colour map. Returns ``None``
    when the frame has no valid pixels (caller keeps waiting for a good frame).
    This range is computed once from the first good frame and then held fixed
    (session-first live, episode-first in replay). Pure — unit-tested.
    """
    depth = np.asarray(depth16)
    valid = depth[depth > 0]
    if valid.size == 0:
        return None
    metres = valid.astype(np.float32) * float(scale_m or 0.001)
    near = float(np.percentile(metres, lo_pct))
    far = float(np.percentile(metres, hi_pct))
    if far - near < min_span_m:
        mid = 0.5 * (near + far)
        near, far = mid - 0.5 * min_span_m, mid + 0.5 * min_span_m
    return near, far


def colourise_depth(
    depth16: np.ndarray,
    scale_m: float = 0.001,
    near_m: float = 0.2,
    far_m: float = 2.0,
) -> np.ndarray:
    """Turn a 16-bit aligned-depth frame into a viewable BGR heat map.

    ``depth16`` holds raw depth units; ``scale_m`` (metres per unit, from the
    RealSense) converts to metres, which are clipped to ``[near_m, far_m]`` and
    mapped through a JET colour map (blue = near, red = far). Zero (no return)
    pixels stay black. Callers pass ``near_m``/``far_m`` locked to the first
    good frame (see ``depth_range_from_frame``) so the scene's own depth spread
    fills the colour map instead of a fixed metric window. Pure — unit-tested;
    used by both the live view and the replay viewer so depth reads the same in
    both.
    """
    depth = np.asarray(depth16)
    valid = depth > 0
    metres = depth.astype(np.float32) * float(scale_m or 0.001)
    clipped = np.clip(metres, near_m, far_m)
    norm = ((clipped - near_m) / max(far_m - near_m, 1e-6) * 255.0).astype(np.uint8)
    colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colour[~valid] = 0
    return colour


@dataclass
class ViewPanel:
    """One camera tile for the composited sensor view.

    ``image_bgr`` is the already-BGR frame (RGB converted / depth colourised)
    or ``None`` for a not-yet-arrived stream (drawn as a black tile of
    ``fallback_hw``). ``line1`` is the green header (name + rate); ``line2``
    is the optional drift/drop status in ``line2_color`` (omitted in replay).
    """

    label: str
    image_bgr: "np.ndarray | None"
    fallback_hw: tuple[int, int]
    line1: str
    line2: "str | None" = None
    line2_color: tuple[int, int, int] = field(default=(0, 255, 0))


def compose_sensor_view_frame(
    panels: "list[ViewPanel]",
    col1_by_side: dict,
    col2_by_side: dict,
    col2_label: str,
    joint_strip: "tuple[str, tuple[int, int, int]] | None",
    col1_label: str = "follower",
    max_w: int = 1280,
    max_h: int = 720,
) -> np.ndarray:
    """Composite one sensor-view frame: a camera row over the joint cells.

    ``panels`` are the ordered camera tiles (RGB then any depth). ``col1_by_side``
    / ``col2_by_side`` map each side to its ``{joint: value|None}`` dict (column
    one is the follower/measured state, column two the leader or command).
    ``joint_strip`` is the optional ``(text, colour)`` proprio-drift line (live
    only; ``None`` in replay). Returns a screen-fitted BGR image. Pure (no data
    manager, no hardware) so the live loop and the replay viewer render an
    identical layout.
    """
    from tool.test_sensor_rates import (
        _CAM_TILE_H,
        _CAM_TILE_W,
        _fit_to_screen,
        _side_panel,
        _vstack_pad,
        grid_tiles,
    )

    cam_tiles = []
    for p in panels:
        if p.image_bgr is None:
            tile = np.zeros((p.fallback_hw[0], p.fallback_hw[1], 3), dtype=np.uint8)
        else:
            tile = np.ascontiguousarray(p.image_bgr)
        tile = cv2.resize(tile, (_CAM_TILE_W, _CAM_TILE_H))
        cv2.putText(tile, p.line1, (6, 26), _FONT, 0.7, (0, 255, 0), 2)
        if p.line2 is not None:
            cv2.putText(tile, p.line2, (6, 52), _FONT, 0.6, p.line2_color, 2)
        cam_tiles.append(tile)

    cells = [
        _side_panel(
            side,
            col1_label,
            col1_by_side[side],
            None,
            col2_label,
            col2_by_side[side],
            None,
        )
        for side in ("left", "right")
    ]
    joint_row = np.hstack(cells)

    blocks = []
    if cam_tiles:
        blocks.append(grid_tiles(cam_tiles, max_per_row=max(len(cam_tiles), 1)))
    if joint_strip is not None:
        text, colour = joint_strip
        strip = np.zeros((34, joint_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(strip, text, (8, 24), _FONT, 0.7, colour, 2)
        blocks.append(strip)
    blocks.append(joint_row)
    return _fit_to_screen(_vstack_pad(blocks), max_w, max_h)


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

    ``captures`` are started camera objects — their ``name``/``width``/
    ``height``/``fps`` are read here; a ``RealSenseCapture`` additionally
    exposes ``depth_name``/``depth_scale``, whose depth stream is shown as an
    extra colourised tile right after its RGB. Frames come from the data
    manager; the layout itself is built by ``compose_sensor_view_frame``.
    """
    from tool.test_sensor_rates import _camera_short_label

    # Drop cameras hidden from the monitor (e.g. "scene"); still recorded.
    captures = visible_view_captures(captures)

    # One rate counter per displayed stream: each camera's RGB plus, for a
    # RealSense, its depth stream (a distinct key so both rates are tracked).
    counters: dict = {}
    depth_scales: dict = {}
    # Per-depth-stream colour range, locked to the first frame that has valid
    # pixels and then held fixed so the tile does not flicker as the scene moves.
    depth_ranges: dict = {}
    for cam in captures:
        exp_hz = float(getattr(cam, "fps", 0) or 0)
        counters[cam.name] = FrameRateCounter(expected_hz=exp_hz)
        dname = getattr(cam, "depth_name", None)
        if dname:
            counters[dname] = FrameRateCounter(expected_hz=exp_hz)
            depth_scales[dname] = float(getattr(cam, "depth_scale", 0.0) or 0.0)

    col2_label = "leader" if leader_mode else "cmd"
    try:
        while not data_manager.is_shutdown_requested():
            now = time.monotonic()
            panels: list[ViewPanel] = []
            for cam in captures:
                rgb = data_manager.get_rgb_image(cam.name)
                counters[cam.name].tick(rgb, now)
                age = data_manager.get_rgb_image_age(cam.name, now)
                label = _camera_short_label(cam.name)
                panels.append(
                    ViewPanel(
                        label=label,
                        image_bgr=(
                            None
                            if rgb is None
                            else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        ),
                        fallback_hw=(cam.height, cam.width),
                        line1=f"{label} {counters[cam.name].hz(now):.0f}Hz",
                        # Live drift line: age from the collection reference +
                        # cumulative dropped frames, colour-coded by budget.
                        line2=f"drift {_age_ms(age)}  drop {counters[cam.name].drops}",
                        line2_color=_age_color(age),
                    )
                )
                dname = getattr(cam, "depth_name", None)
                if dname:
                    depth = data_manager.get_depth_image(dname)
                    counters[dname].tick(depth, now)
                    dage = data_manager.get_depth_image_age(dname, now)
                    dlabel = _camera_short_label(dname)
                    dscale = depth_scales.get(dname, 0.0)
                    if depth is not None and dname not in depth_ranges:
                        rng = depth_range_from_frame(depth, dscale)
                        if rng is not None:
                            depth_ranges[dname] = rng
                    drng = depth_ranges.get(dname)
                    panels.append(
                        ViewPanel(
                            label=dlabel,
                            image_bgr=(
                                None
                                if depth is None or drng is None
                                else colourise_depth(depth, dscale, drng[0], drng[1])
                            ),
                            fallback_hw=(cam.height, cam.width),
                            line1=f"{dlabel} {counters[dname].hz(now):.0f}Hz",
                            line2=f"drift {_age_ms(dage)}  drop {counters[dname].drops}",
                            line2_color=_age_color(dage),
                        )
                    )

            measured = data_manager.get_current_joint_angles()
            target = None if leader_mode else data_manager.get_target_joint_angles()
            col1_by_side, col2_by_side = {}, {}
            for side in ("left", "right"):
                col1_by_side[side] = side_joint_dict(
                    measured, side, data_manager.get_current_gripper_open_value(side)
                )
                if leader_mode:
                    lvec, lgrip = data_manager.get_leader_mapped_state(side)
                    col2_by_side[side] = _vec5_to_dict(lvec, lgrip)
                else:
                    col2_by_side[side] = side_joint_dict(
                        target, side, data_manager.get_target_gripper_open_value(side)
                    )

            # Proprio drift strip: how stale the measured joints are at the
            # collection reference time (100 Hz stream → normally < 10 ms).
            joints_res = data_manager.get_current_joint_angles_at(now)
            joint_age = None if joints_res is None else joints_res[1]
            joint_strip = (f"joints drift {_age_ms(joint_age)}", _age_color(joint_age))

            frame = compose_sensor_view_frame(
                panels,
                col1_by_side,
                col2_by_side,
                col2_label,
                joint_strip,
                max_w=max_w,
                max_h=max_h,
            )
            cv2.imshow(window, frame)

            key = cv2.waitKey(33) & 0xFF
            if _handle_view_key(key, key_callbacks):
                break
    finally:
        cv2.destroyAllWindows()
