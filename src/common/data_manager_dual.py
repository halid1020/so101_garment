#!/usr/bin/env python3
"""Thread-safe state management for dual-arm SO101 teleoperation.

Mirrors the openarm DataManager pattern: per-arm keyed state for
controller transforms, gripper values, and EEF poses; shared IK and
robot activity state.
"""

import threading
import time
from enum import Enum
from typing import Any, Callable

import numpy as np

from .one_euro_filter import OneEuroFilterTransform
from .sync import TimestampedHistory, interp_pose, lerp, select_interp

# How long the timestamped stream histories retain samples (seconds). The
# data-collection recorder aligns every stream to one reference time per frame
# by sampling these; ~0.5 s comfortably brackets a 30 Hz tick against the
# ~100 Hz proprio streams while keeping memory bounded.
_HISTORY_MAX_AGE_S = 0.5


class RobotActivityState(Enum):
    ENABLED = "ENABLED"
    HOMING = "HOMING"
    DISABLED = "DISABLED"
    POLICY_CONTROLLED = "POLICY_CONTROLLED"


class ControllerState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.min_cutoff: float = 1.0
        self.beta: float = 0.0
        self.d_cutoff: float = 1.0
        self.transform_raw: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.transform: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.grip_value: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.trigger_value: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._filter: dict[str, OneEuroFilterTransform | None] = {
            "left": None,
            "right": None,
        }


class TeleopState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active: bool = False
        self.translation_scale: float = 1.0
        self.rotation_scale: float = 1.0
        self.mirror_control_enabled: bool = False
        self.height_lock_enabled: bool = False
        # Pending per-hand roll-ratchet resets (consumed at the next grip).
        self.roll_reset_requests: dict[str, bool] = {"left": False, "right": False}
        self.gizmo_target_poses: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        # Named 3D points (robot/world coords) for visualization — e.g. the
        # notional headset center published by the IK thread at calibration.
        self.frame_markers: dict[str, np.ndarray] = {}


class RobotState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.joint_angles: np.ndarray | None = (
            None  # 10-DOF body-only [left×5, right×5]
        )
        self.end_effector_poses: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self.current_gripper_open_values: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.target_gripper_open_values: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.activity_state: RobotActivityState = RobotActivityState.DISABLED
        # Timestamped histories for reference-time alignment (see common.sync).
        self.joint_history = TimestampedHistory(_HISTORY_MAX_AGE_S)
        self.ee_history: dict[str, TimestampedHistory] = {
            "left": TimestampedHistory(_HISTORY_MAX_AGE_S),
            "right": TimestampedHistory(_HISTORY_MAX_AGE_S),
        }
        self.cur_gripper_history: dict[str, TimestampedHistory] = {
            "left": TimestampedHistory(_HISTORY_MAX_AGE_S),
            "right": TimestampedHistory(_HISTORY_MAX_AGE_S),
        }


class IKState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.target_joint_angles: np.ndarray | None = None  # 10-DOF body-only
        self.target_poses: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.solve_time_ms: float = 0.0
        self.success: bool = True
        # Timestamped history of the projected+constrained EE target per side.
        self.target_pose_history: dict[str, TimestampedHistory] = {
            "left": TimestampedHistory(_HISTORY_MAX_AGE_S),
            "right": TimestampedHistory(_HISTORY_MAX_AGE_S),
        }


class CameraState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rgb_images: dict[str, np.ndarray] = {}
        # time.monotonic() of the most recent frame per camera name, used by
        # the recorder to detect stale streams.
        self.rgb_timestamps: dict[str, float] = {}
        # Aligned 16-bit depth (uint16) per stream name + capture stamps.
        self.depth_images: dict[str, np.ndarray] = {}
        self.depth_timestamps: dict[str, float] = {}


class LeaderMappedStateDual:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.joint_angles: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.gripper_open: dict[str, float | None] = {"left": None, "right": None}


class LastSentCommandState:
    """Latest joint-space command actually written to the motors, per side.

    Published by ``threads/dual_joint_state.py`` after each successful
    ``sync_write`` of Goal_Position, and consumed by the recorder's ``action``
    feature (used only while fresh; see the recorder's action-fallback rule).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # URDF-space 5-DOF body targets (degrees) per side.
        self.urdf_deg: dict[str, np.ndarray | None] = {"left": None, "right": None}
        # Commanded gripper open fraction (0-1) per side.
        self.gripper_open: dict[str, float | None] = {"left": None, "right": None}
        # time.monotonic() timestamp of the write per side (for freshness).
        self.t_mono: dict[str, float | None] = {"left": None, "right": None}


class DualDataManager:
    """State container for dual-arm SO101 teleoperation.

    Matches the openarm DataManager API so the openarm IK solver thread
    and joint state thread can be ported with minimal changes.
    """

    def __init__(self) -> None:
        self._controller_state = ControllerState()
        self._teleop_state = TeleopState()
        self._robot_state = RobotState()
        self._ik_state = IKState()
        self._camera_state = CameraState()
        self._leader_mapped_state = LeaderMappedStateDual()
        self._last_sent_command = LastSentCommandState()
        self._shutdown_event = threading.Event()
        # Set when a motor bus stops existing (the USB serial adapter left the
        # bus), as opposed to returning a protocol error. Teardown consults this
        # instead of guessing from an exception message: once the bus is gone the
        # port handle itself misbehaves, so the error the teardown path sees is a
        # consequence ("port is in use") rather than the cause.
        self._bus_lost_event = threading.Event()
        self._on_change_callback: Callable[
            [str, dict[str, Any], float], None
        ] | None = None

    def set_on_change_callback(
        self, callback: Callable[[str, dict[str, Any], float], None]
    ) -> None:
        self._on_change_callback = callback

    # ── Camera ──────────────────────────────────────────────────────────────────

    def get_rgb_image(self, camera_name: str) -> np.ndarray | None:
        with self._camera_state._lock:
            img = self._camera_state.rgb_images.get(camera_name)
            return img.copy() if img is not None else None

    def set_rgb_image(
        self, image: np.ndarray, camera_name: str, t_capture: float | None = None
    ) -> None:
        """Publish the latest frame for ``camera_name``.

        ``t_capture`` is the ``time.monotonic()`` stamp taken at the moment the
        device was read (the capture thread passes it so alignment reflects the
        true capture instant, not the post-decode publish); it defaults to now.
        Only the latest frame is kept — for a causal 30 Hz collector the
        newest frame is always the nearest available sample to the tick, so a
        deeper ring would not improve image alignment (unsynchronised UVC
        cameras cannot be re-phased in software; the drift is logged instead).
        """
        with self._camera_state._lock:
            self._camera_state.rgb_images[camera_name] = image.copy()
            self._camera_state.rgb_timestamps[camera_name] = (
                time.monotonic() if t_capture is None else float(t_capture)
            )

    def get_rgb_image_at(
        self, camera_name: str, t_ref: float
    ) -> tuple[np.ndarray, float] | None:
        """Return ``(frame, drift_s)`` for ``camera_name`` at reference ``t_ref``.

        The nearest available frame is the latest one (see
        :meth:`set_rgb_image`); ``drift_s = t_ref - t_capture``. ``None`` if no
        frame has been published for that camera.
        """
        with self._camera_state._lock:
            img = self._camera_state.rgb_images.get(camera_name)
            ts = self._camera_state.rgb_timestamps.get(camera_name)
            if img is None or ts is None:
                return None
            return img.copy(), t_ref - ts

    def get_rgb_image_age(
        self, camera_name: str, now_mono: float | None = None
    ) -> float | None:
        """Age in seconds of the latest frame for ``camera_name``.

        Returns ``None`` if no frame has ever been published for that camera
        (used by the recorder to reject stale/absent streams). ``now_mono``
        defaults to ``time.monotonic()``.
        """
        if now_mono is None:
            now_mono = time.monotonic()
        with self._camera_state._lock:
            ts = self._camera_state.rgb_timestamps.get(camera_name)
            return None if ts is None else now_mono - ts

    def set_depth_image(
        self, depth: np.ndarray, name: str, t_capture: float | None = None
    ) -> None:
        """Publish the latest aligned 16-bit depth frame for ``name``.

        ``depth`` is a ``uint16`` (H, W) array in the camera's native depth
        units (see the dataset's ``realsense.json`` for the metres-per-unit
        scale). ``t_capture`` is the ``time.monotonic()`` capture stamp (the
        RealSense thread passes the same stamp used for its colour frame so RGB
        and depth stay co-timed); it defaults to now.
        """
        with self._camera_state._lock:
            self._camera_state.depth_images[name] = depth.copy()
            self._camera_state.depth_timestamps[name] = (
                time.monotonic() if t_capture is None else float(t_capture)
            )

    def get_depth_image(self, name: str) -> np.ndarray | None:
        with self._camera_state._lock:
            img = self._camera_state.depth_images.get(name)
            return img.copy() if img is not None else None

    def get_depth_image_age(
        self, name: str, now_mono: float | None = None
    ) -> float | None:
        """Age in seconds of the latest depth frame for ``name`` (None if never)."""
        if now_mono is None:
            now_mono = time.monotonic()
        with self._camera_state._lock:
            ts = self._camera_state.depth_timestamps.get(name)
            return None if ts is None else now_mono - ts

    def get_depth_image_at(
        self, name: str, t_ref: float
    ) -> tuple[np.ndarray, float] | None:
        """Return ``(depth, drift_s)`` for ``name`` at reference ``t_ref``.

        The latest depth frame is the nearest causal sample (same rationale as
        :meth:`get_rgb_image_at`). ``None`` if no depth has been published.
        """
        with self._camera_state._lock:
            img = self._camera_state.depth_images.get(name)
            ts = self._camera_state.depth_timestamps.get(name)
            if img is None or ts is None:
                return None
            return img.copy(), t_ref - ts

    # ── Controller ──────────────────────────────────────────────────────────────

    def get_controller_state(self, hand: str) -> tuple[np.ndarray | None, float, float]:
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        with self._controller_state._lock:
            tf = self._controller_state.transform[hand]
            return (
                tf.copy() if tf is not None else None,
                self._controller_state.grip_value[hand],
                self._controller_state.trigger_value[hand],
            )

    def get_controller_state_raw(
        self, hand: str
    ) -> tuple[np.ndarray | None, float, float]:
        """Return the UNFILTERED controller transform, grip and trigger.

        Mirrors :meth:`get_controller_state` but returns the raw handle pose
        (before the One-Euro filter), used by the sidecar to log the operator
        input at full fidelity.
        """
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        with self._controller_state._lock:
            tf = self._controller_state.transform_raw[hand]
            return (
                tf.copy() if tf is not None else None,
                self._controller_state.grip_value[hand],
                self._controller_state.trigger_value[hand],
            )

    def set_controller_state(
        self, hand: str, transform: np.ndarray | None, grip: float, trigger: float
    ) -> None:
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        with self._controller_state._lock:
            self._controller_state.grip_value[hand] = grip
            self._controller_state.trigger_value[hand] = trigger
            if transform is not None:
                t = time.time()
                self._controller_state.transform_raw[hand] = transform.copy()
                hand_filter = self._controller_state._filter[hand]
                if hand_filter is None:
                    self._controller_state._filter[hand] = OneEuroFilterTransform(
                        t,
                        transform,
                        self._controller_state.min_cutoff,
                        self._controller_state.beta,
                        self._controller_state.d_cutoff,
                    )
                    self._controller_state.transform[hand] = transform.copy()
                else:
                    hand_filter.update_params(
                        self._controller_state.min_cutoff,
                        self._controller_state.beta,
                        self._controller_state.d_cutoff,
                    )
                    self._controller_state.transform[hand] = hand_filter(t, transform)
            else:
                self._controller_state.transform[hand] = None
                self._controller_state.transform_raw[hand] = None
                self._controller_state._filter[hand] = None

    def set_controller_filter_params(
        self, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
        with self._controller_state._lock:
            self._controller_state.min_cutoff = min_cutoff
            self._controller_state.beta = beta
            self._controller_state.d_cutoff = d_cutoff

    def get_controller_filter_params(self) -> tuple[float, float, float]:
        with self._controller_state._lock:
            return (
                self._controller_state.min_cutoff,
                self._controller_state.beta,
                self._controller_state.d_cutoff,
            )

    # ── Teleop ──────────────────────────────────────────────────────────────────

    def set_teleop_state(self, active: bool) -> None:
        with self._teleop_state._lock:
            self._teleop_state.active = active

    def get_teleop_active(self) -> bool:
        with self._teleop_state._lock:
            return self._teleop_state.active

    def set_teleop_scaling(
        self, translation_scale: float, rotation_scale: float
    ) -> None:
        if translation_scale <= 0.0 or rotation_scale <= 0.0:
            return
        with self._teleop_state._lock:
            self._teleop_state.translation_scale = translation_scale
            self._teleop_state.rotation_scale = rotation_scale

    def get_teleop_scaling(self) -> tuple[float, float]:
        with self._teleop_state._lock:
            return (
                self._teleop_state.translation_scale,
                self._teleop_state.rotation_scale,
            )

    def set_mirror_control_enabled(self, enabled: bool) -> None:
        with self._teleop_state._lock:
            self._teleop_state.mirror_control_enabled = enabled

    def get_mirror_control_enabled(self) -> bool:
        with self._teleop_state._lock:
            return self._teleop_state.mirror_control_enabled

    def toggle_height_lock(self) -> bool:
        """Toggle the height lock (targets keep their current z). Returns
        the new state."""
        with self._teleop_state._lock:
            self._teleop_state.height_lock_enabled = (
                not self._teleop_state.height_lock_enabled
            )
            return self._teleop_state.height_lock_enabled

    def get_height_lock_enabled(self) -> bool:
        with self._teleop_state._lock:
            return self._teleop_state.height_lock_enabled

    def request_roll_reset(self, side: str) -> None:
        """Queue a wrist-roll reset for one hand; the IK thread consumes it
        at the next clutch engagement (see common/roll_ratchet.py)."""
        with self._teleop_state._lock:
            self._teleop_state.roll_reset_requests[side] = True

    def consume_roll_reset(self, side: str) -> bool:
        """Return-and-clear the pending roll-reset flag for one hand."""
        with self._teleop_state._lock:
            pending = self._teleop_state.roll_reset_requests[side]
            self._teleop_state.roll_reset_requests[side] = False
            return pending

    def set_frame_marker(self, name: str, position: np.ndarray) -> None:
        """Publish a named 3D point (robot/world coords) for visualization."""
        with self._teleop_state._lock:
            self._teleop_state.frame_markers[name] = np.asarray(
                position, dtype=float
            ).copy()

    def get_frame_markers(self) -> dict[str, np.ndarray]:
        with self._teleop_state._lock:
            return {k: v.copy() for k, v in self._teleop_state.frame_markers.items()}

    def set_gizmo_target_pose(self, side: str, transform: np.ndarray | None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._teleop_state._lock:
            self._teleop_state.gizmo_target_poses[side] = (
                transform.copy() if transform is not None else None
            )

    def get_gizmo_target_pose(self, side: str) -> np.ndarray | None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._teleop_state._lock:
            tf = self._teleop_state.gizmo_target_poses[side]
            return tf.copy() if tf is not None else None

    # ── Robot State ─────────────────────────────────────────────────────────────

    def get_robot_activity_state(self) -> RobotActivityState:
        with self._robot_state._lock:
            return self._robot_state.activity_state

    def set_robot_activity_state(self, state: RobotActivityState) -> None:
        with self._robot_state._lock:
            self._robot_state.activity_state = state

    def get_current_joint_angles(self) -> np.ndarray | None:
        with self._robot_state._lock:
            return (
                self._robot_state.joint_angles.copy()
                if self._robot_state.joint_angles is not None
                else None
            )

    def set_current_joint_angles(self, angles: np.ndarray) -> None:
        snapshot = angles.copy()
        with self._robot_state._lock:
            self._robot_state.joint_angles = snapshot
        # Stamp on publish (≈ read time: the joint thread calls this straight
        # after the hardware read) so the collector can align to a tick time.
        self._robot_state.joint_history.append(time.monotonic(), snapshot)

    def get_current_joint_angles_at(
        self, t_ref: float
    ) -> tuple[np.ndarray, float] | None:
        """Interpolate the measured 10-DOF joints to reference time ``t_ref``.

        Returns ``(angles, drift_s)`` where ``drift_s = t_ref - t_nearest`` (the
        residual to the closest real sample), or ``None`` before any sample.
        """
        return select_interp(self._robot_state.joint_history.snapshot(), t_ref, lerp)

    def get_current_end_effector_pose(self, side: str) -> np.ndarray | None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._robot_state._lock:
            pose = self._robot_state.end_effector_poses[side]
            return pose.copy() if pose is not None else None

    def set_current_end_effector_pose(self, side: str, pose: np.ndarray | None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        snapshot = pose.copy() if pose is not None else None
        with self._robot_state._lock:
            self._robot_state.end_effector_poses[side] = snapshot
        if snapshot is not None:
            self._robot_state.ee_history[side].append(time.monotonic(), snapshot)

    def get_current_end_effector_pose_at(
        self, side: str, t_ref: float
    ) -> tuple[np.ndarray, float] | None:
        """Interpolate the measured EE 4×4 pose to reference time ``t_ref``.

        LERP translation, SLERP rotation. Returns ``(pose, drift_s)`` or
        ``None`` before any sample.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return select_interp(
            self._robot_state.ee_history[side].snapshot(), t_ref, interp_pose
        )

    def get_current_gripper_open_value(self, side: str) -> float | None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._robot_state._lock:
            return self._robot_state.current_gripper_open_values[side]

    def set_current_gripper_open_value(self, side: str, value: float) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._robot_state._lock:
            self._robot_state.current_gripper_open_values[side] = float(value)
        self._robot_state.cur_gripper_history[side].append(
            time.monotonic(), np.array([float(value)])
        )

    def get_current_gripper_open_value_at(
        self, side: str, t_ref: float
    ) -> tuple[float, float] | None:
        """Interpolate the measured gripper open fraction to ``t_ref``.

        Returns ``(open_fraction, drift_s)`` or ``None`` before any sample.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        result = select_interp(
            self._robot_state.cur_gripper_history[side].snapshot(), t_ref, lerp
        )
        if result is None:
            return None
        value, drift = result
        return float(value[0]), drift

    def get_target_gripper_open_value(self, side: str) -> float | None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._robot_state._lock:
            return self._robot_state.target_gripper_open_values[side]

    def set_target_gripper_open_value(self, side: str, value: float) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._robot_state._lock:
            self._robot_state.target_gripper_open_values[side] = float(value)

    # ── IK State ────────────────────────────────────────────────────────────────

    def get_target_joint_angles(self) -> np.ndarray | None:
        with self._ik_state._lock:
            return (
                self._ik_state.target_joint_angles.copy()
                if self._ik_state.target_joint_angles is not None
                else None
            )

    def set_target_joint_angles(self, angles: np.ndarray) -> None:
        with self._ik_state._lock:
            self._ik_state.target_joint_angles = angles.copy()

    def set_target_pose(self, side: str, transform: np.ndarray | None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        snapshot = transform.copy() if transform is not None else None
        with self._ik_state._lock:
            self._ik_state.target_poses[side] = snapshot
        if snapshot is not None:
            self._ik_state.target_pose_history[side].append(time.monotonic(), snapshot)

    def get_target_pose(self, side: str) -> np.ndarray | None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._ik_state._lock:
            tf = self._ik_state.target_poses[side]
            return tf.copy() if tf is not None else None

    def get_target_pose_at(
        self, side: str, t_ref: float
    ) -> tuple[np.ndarray, float] | None:
        """Interpolate the projected+constrained EE target to ``t_ref``.

        LERP translation, SLERP rotation. Returns ``(pose, drift_s)`` or
        ``None`` before any sample.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return select_interp(
            self._ik_state.target_pose_history[side].snapshot(), t_ref, interp_pose
        )

    def set_ik_solve_time_ms(self, time_ms: float) -> None:
        with self._ik_state._lock:
            self._ik_state.solve_time_ms = time_ms

    def set_ik_success(self, success: bool) -> None:
        with self._ik_state._lock:
            self._ik_state.success = success

    def get_ik_solve_time_ms(self) -> float:
        with self._ik_state._lock:
            return self._ik_state.solve_time_ms

    def get_ik_success(self) -> bool:
        with self._ik_state._lock:
            return self._ik_state.success

    # ── Leader Mapped State ─────────────────────────────────────────────────────

    def set_leader_mapped_state(
        self, side: str, joint_angles: np.ndarray, gripper_open: float
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._leader_mapped_state._lock:
            self._leader_mapped_state.joint_angles[side] = joint_angles.copy()
            self._leader_mapped_state.gripper_open[side] = float(gripper_open)

    def get_leader_mapped_state(
        self, side: str
    ) -> tuple[np.ndarray | None, float | None]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._leader_mapped_state._lock:
            angles = self._leader_mapped_state.joint_angles[side]
            return (
                angles.copy() if angles is not None else None,
                self._leader_mapped_state.gripper_open[side],
            )

    # ── Last Sent Command ───────────────────────────────────────────────────────

    def set_last_sent_command(
        self,
        side: str,
        arm_urdf_deg_5: np.ndarray,
        gripper_open: float,
        t_mono: float,
    ) -> None:
        """Publish the joint-space command just written to one arm's motors.

        Called from ``threads/dual_joint_state.py`` after a successful
        ``sync_write``. ``arm_urdf_deg_5`` are the 5 body-joint targets in
        URDF degrees, ``gripper_open`` the commanded 0-1 open fraction, and
        ``t_mono`` a ``time.monotonic()`` timestamp for freshness checks.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._last_sent_command._lock:
            self._last_sent_command.urdf_deg[side] = np.asarray(
                arm_urdf_deg_5, dtype=np.float64
            ).copy()
            self._last_sent_command.gripper_open[side] = float(gripper_open)
            self._last_sent_command.t_mono[side] = float(t_mono)

    def get_last_sent_command(
        self, side: str
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        """Return ``(urdf_deg_5, gripper_open, t_mono)`` for one arm.

        Any element is ``None`` before the first command has been sent.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._last_sent_command._lock:
            angles = self._last_sent_command.urdf_deg[side]
            return (
                angles.copy() if angles is not None else None,
                self._last_sent_command.gripper_open[side],
                self._last_sent_command.t_mono[side],
            )

    # ── System ──────────────────────────────────────────────────────────────────

    def request_shutdown(self) -> None:
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def note_bus_lost(self) -> None:
        """Record that a motor bus is no longer present on the system."""
        self._bus_lost_event.set()

    def is_bus_lost(self) -> bool:
        """Whether a motor bus was seen to disappear during this session."""
        return self._bus_lost_event.is_set()
