"""Full-rate sidecar sampler for data collection.

A ~100 Hz thread that reads (never mutates) the shared ``DualDataManager`` and
the Quest reader, buffering one row per tick between ``begin_episode`` and
``end_episode``. On ``end_episode`` it writes ``<root>/extra/episode_XXXXXX
.parquet`` via pyarrow; ``abort_episode`` drops the buffer. Missing values are
NaN-filled — rows are never dropped, so wall-clock timing stays intact for
offline drift analysis.

Per-arm end-effector poses are logged in that arm's OWN base frame. The dual
URDF fixes each arm base to the world, so ``T_world_base`` is constant: it is
computed once with pinocchio at construction and ``T_base_ee =
inv(T_world_base) @ T_world_ee``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore[import]
import pyarrow as pa  # type: ignore[import]
import pyarrow.parquet as pq  # type: ignore[import]

from common.configs import (
    DUAL_URDF_PATH,
    LEFT_ARM_BASE_FRAME_NAME,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_BASE_FRAME_NAME,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from common.data_manager_dual import DualDataManager
from common.recording.features import BODY_JOINTS, SIDES

_NAN = float("nan")
_HW_OFFSETS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
}
_HW_SIGNS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
}
_BASE_FRAMES = {
    "left": LEFT_ARM_BASE_FRAME_NAME,
    "right": RIGHT_ARM_BASE_FRAME_NAME,
}


def _quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion in (w, x, y, z) order."""
    quat = pin.Quaternion(np.asarray(rotation, dtype=np.float64))
    quat.normalize()
    return (float(quat.w), float(quat.x), float(quat.y), float(quat.z))


def compute_world_base_transforms(
    urdf_path: str = DUAL_URDF_PATH,
) -> dict[str, np.ndarray]:
    """Return the constant 4x4 ``T_world_base`` for each arm from the URDF.

    Each arm base is a fixed joint in the dual URDF, so its world placement does
    not depend on the configuration; it is evaluated once at the neutral pose.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    transforms: dict[str, np.ndarray] = {}
    for side, frame_name in _BASE_FRAMES.items():
        fid = model.getFrameId(frame_name)
        transforms[side] = np.array(data.oMf[fid].homogeneous, dtype=np.float64)
    return transforms


class SidecarSampler:
    """~100 Hz reader thread writing a per-episode full-rate parquet sidecar."""

    def __init__(
        self,
        data_manager: DualDataManager,
        quest_reader: Any,
        root: str | Path,
        rate_hz: float = 100.0,
        include_hw_frame_goal: bool = True,
        urdf_path: str = DUAL_URDF_PATH,
    ) -> None:
        self.data_manager = data_manager
        self.quest_reader = quest_reader
        self.root = Path(root)
        self.rate_hz = float(rate_hz)
        self.include_hw_frame_goal = include_hw_frame_goal

        self._world_base = compute_world_base_transforms(urdf_path)
        self._world_base_inv = {
            side: np.linalg.inv(tf) for side, tf in self._world_base.items()
        }

        self._lock = threading.Lock()
        self._recording = False
        self._rows: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── Base-frame helpers ───────────────────────────────────────────────────

    def base_transform(self, side: str) -> np.ndarray:
        """Return the constant 4x4 ``T_world_base`` for one arm."""
        return self._world_base[side].copy()

    def to_base_frame(self, side: str, world_pose: np.ndarray) -> np.ndarray:
        """Map a 4x4 world EE pose into that arm's own base frame."""
        return self._world_base_inv[side] @ np.asarray(world_pose, dtype=np.float64)

    # ── Episode control (called from the recorder thread) ────────────────────

    def begin_episode(self) -> None:
        with self._lock:
            self._rows = []
            self._recording = True

    def end_episode(self, ep_idx: int) -> Path | None:
        """Stop buffering and flush the buffered rows to a parquet file."""
        with self._lock:
            self._recording = False
            rows = self._rows
            self._rows = []
        return self._write_parquet(ep_idx, rows)

    def abort_episode(self) -> None:
        with self._lock:
            self._recording = False
            self._rows = []

    # ── Thread lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        dt = 1.0 / self.rate_hz
        while not self._stop.is_set() and not self.data_manager.is_shutdown_requested():
            start = time.perf_counter()
            with self._lock:
                recording = self._recording
            if recording:
                row = self._sample_row()
                with self._lock:
                    if self._recording:
                        self._rows.append(row)
            sleep_time = dt - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Row assembly ─────────────────────────────────────────────────────────

    def _pose_cols(
        self, prefix: str, side: str, world_pose: np.ndarray | None
    ) -> dict[str, float]:
        keys = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        if world_pose is None:
            return {f"{prefix}_{side}_{k}": _NAN for k in keys}
        base_pose = self.to_base_frame(side, world_pose)
        pos = base_pose[:3, 3]
        w, x, y, z = _quat_wxyz(base_pose[:3, :3])
        return {
            f"{prefix}_{side}_x": float(pos[0]),
            f"{prefix}_{side}_y": float(pos[1]),
            f"{prefix}_{side}_z": float(pos[2]),
            f"{prefix}_{side}_qw": w,
            f"{prefix}_{side}_qx": x,
            f"{prefix}_{side}_qy": y,
            f"{prefix}_{side}_qz": z,
        }

    def _controller_cols(
        self, prefix: str, side: str, tf: np.ndarray | None
    ) -> dict[str, float]:
        keys = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        if tf is None:
            return {f"{prefix}_{side}_{k}": _NAN for k in keys}
        pos = np.asarray(tf, dtype=np.float64)[:3, 3]
        w, x, y, z = _quat_wxyz(np.asarray(tf, dtype=np.float64)[:3, :3])
        return {
            f"{prefix}_{side}_x": float(pos[0]),
            f"{prefix}_{side}_y": float(pos[1]),
            f"{prefix}_{side}_z": float(pos[2]),
            f"{prefix}_{side}_qw": w,
            f"{prefix}_{side}_qx": x,
            f"{prefix}_{side}_qy": y,
            f"{prefix}_{side}_qz": z,
        }

    def _sample_row(self) -> dict[str, Any]:
        dm = self.data_manager
        row: dict[str, Any] = {
            "t_wall": time.time(),
            "t_mono": time.monotonic(),
            "activity_state": dm.get_robot_activity_state().value,
            "teleop_active": 1.0 if dm.get_teleop_active() else 0.0,
        }

        measured = dm.get_current_joint_angles()
        for s, side in enumerate(SIDES):
            if measured is not None and len(measured) >= 5 * (s + 1):
                q_side = np.asarray(measured[s * 5 : (s + 1) * 5], dtype=np.float64)
            else:
                q_side = np.full(5, _NAN)
            for j, joint in enumerate(BODY_JOINTS):
                row[f"q_{side}_{joint}"] = float(q_side[j])

            cmd_urdf, cmd_grip_last, _ = dm.get_last_sent_command(side)
            if cmd_urdf is not None:
                cmd_arr = np.asarray(cmd_urdf, dtype=np.float64)
            else:
                cmd_arr = np.full(5, _NAN)
            for j, joint in enumerate(BODY_JOINTS):
                row[f"cmd_q_{side}_{joint}"] = float(cmd_arr[j])
            if self.include_hw_frame_goal:
                hw = _HW_SIGNS[side] * (cmd_arr - _HW_OFFSETS[side])
                for j, joint in enumerate(BODY_JOINTS):
                    row[f"cmd_hw_{side}_{joint}"] = float(hw[j])

            grip_meas = dm.get_current_gripper_open_value(side)
            grip_cmd = dm.get_target_gripper_open_value(side)
            row[f"grip_meas_{side}"] = _NAN if grip_meas is None else float(grip_meas)
            row[f"grip_cmd_{side}"] = _NAN if grip_cmd is None else float(grip_cmd)

            row.update(
                self._pose_cols("ee", side, dm.get_current_end_effector_pose(side))
            )
            row.update(self._pose_cols("tgt_ee", side, dm.get_target_pose(side)))

            tf_f, grip_val, trig_val = dm.get_controller_state(side)
            tf_raw, _, _ = dm.get_controller_state_raw(side)
            row.update(self._controller_cols("ctrl", side, tf_f))
            row.update(self._controller_cols("ctrlraw", side, tf_raw))
            row[f"grip_value_{side}"] = float(grip_val)
            row[f"trigger_value_{side}"] = float(trig_val)

            if self.quest_reader is not None:
                js_x, js_y = self.quest_reader.get_joystick_value(side)
            else:
                js_x, js_y = _NAN, _NAN
            row[f"js_x_{side}"] = float(js_x)
            row[f"js_y_{side}"] = float(js_y)

        return row

    # ── Parquet output ───────────────────────────────────────────────────────

    def _write_parquet(self, ep_idx: int, rows: list[dict[str, Any]]) -> Path | None:
        extra_dir = self.root / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        out_path = extra_dir / f"episode_{ep_idx:06d}.parquet"
        if not rows:
            print(f"⚠️  sidecar: episode {ep_idx} had no rows; nothing written")
            return None
        # Stable column order taken from the first row; every row shares keys.
        columns = list(rows[0].keys())
        table = pa.Table.from_pydict({col: [r[col] for r in rows] for col in columns})
        pq.write_table(table, out_path)
        n_rows = len(rows)
        span = rows[-1]["t_wall"] - rows[0]["t_wall"] if n_rows > 1 else 0.0
        rate = (n_rows - 1) / span if span > 0 else float(n_rows)
        print(
            f"  💾 sidecar: wrote {n_rows} rows " f"(~{rate:.0f} Hz) -> {out_path.name}"
        )
        return out_path
