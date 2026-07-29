"""Leader-arm reader thread for joint-space dual-arm teleoperation.

Polls two passive SO-101 leader arms (LeRobot ``SOLeader``) at the joint
streaming rate and publishes 10-DOF URDF-degree targets plus synthesised
trigger values into the shared ``DualDataManager``. The existing per-arm
joint-state threads remain the sole motor writers, so the recorder's
``action`` feature and the sidecar command columns keep exactly the Quest
semantics.

Unit mapping: a leader in DEGREES normalisation reads degrees about its
calibrated mid-range — the same convention as the follower's hardware
frame — so the leader reading is treated as the follower's HW reading and
converted with the follower's own hw→URDF signs/offsets.

Engage safety: whenever teleoperation (re)activates, the published command
seeds from the followers' measured joints and slews toward the leader pose
at a bounded joint velocity. The per-tick clamp stays on permanently, so
it doubles as a joint-velocity limit during tracking.
"""

import math
import time
import traceback
from typing import Mapping

import numpy as np

from common.configs import (
    GRIPPER_OPEN_MAX_FRAC,
    JOINT_STATE_STREAMING_RATE,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    MAX_JOINT_VEL_HW_RAD_S,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from common.data_manager_dual import DualDataManager, RobotActivityState

_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
_HW_TO_URDF = {
    "left": (
        np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
        np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    ),
    "right": (
        np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
        np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    ),
}


def leader_action_to_urdf(action: Mapping[str, float], side: str) -> np.ndarray:
    """Map one leader's ``get_action()`` body joints to follower URDF degrees.

    The leader's DEGREES reading shares the follower's hardware-frame
    convention (zero at the calibrated mid-range), so the follower's own
    conversion applies: ``urdf = sign * hw + offset``.
    """
    signs, offsets = _HW_TO_URDF[side]
    hw = np.array([action[f"{j}.pos"] for j in _BODY_JOINTS], dtype=np.float64)
    return signs * hw + offsets


def leader_gripper_to_trigger(gripper_0_100: float) -> float:
    """Map the leader jaw (0–100, 0 = closed) to Quest trigger semantics.

    The joint-state threads compute the follower opening as
    ``(1 - trigger) * GRIPPER_OPEN_MAX_FRAC``, so a fully open leader jaw
    yields the capped opening and a closed jaw closes fully — identical
    command semantics to Quest recordings.
    """
    frac = min(max(gripper_0_100 / 100.0, 0.0), 1.0)
    return 1.0 - frac


def slew_toward(cmd: np.ndarray, target: np.ndarray, max_step_deg: float) -> np.ndarray:
    """Advance ``cmd`` toward ``target`` by at most ``max_step_deg`` per joint."""
    return cmd + np.clip(target - cmd, -max_step_deg, max_step_deg)


def leader_arm_thread(
    data_manager: DualDataManager,
    leaders: Mapping[str, object],
    rate_hz: float = JOINT_STATE_STREAMING_RATE,
    max_joint_vel_rad_s: float = MAX_JOINT_VEL_HW_RAD_S,
) -> None:
    """Poll both leaders and publish joint targets while teleop is active.

    Args:
        data_manager: Shared dual-arm DataManager.
        leaders: {"left": SOLeader, "right": SOLeader} — sides follow the
                 tool's bus wiring (left = bus_0). Each leader owns its own
                 serial port, so no bus lock is shared with the followers.
        rate_hz: Poll/publish rate (default: the joint streaming rate).
        max_joint_vel_rad_s: Per-joint slew limit — bounds the engage
                 catch-up and doubles as a permanent velocity limit.
    """
    print("🕹️  Leader-arm thread started")
    dt = 1.0 / rate_hz
    max_step_deg = math.degrees(max_joint_vel_rad_s) * dt
    cmd: np.ndarray | None = None  # slewed 10-DOF URDF-degree command
    # Same serial-robustness policy as the joint-state threads: a dropped
    # Feetech status packet surfaces as ConnectionError; skip the tick and
    # only give up after a full second of consecutive failures.
    _MAX_CONSECUTIVE_FAILURES = 100
    consecutive_failures = 0

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            try:
                urdf: dict[str, np.ndarray] = {}
                trigger: dict[str, float] = {}
                for side, leader in leaders.items():
                    action = leader.get_action()
                    urdf[side] = leader_action_to_urdf(action, side)
                    trigger[side] = leader_gripper_to_trigger(action["gripper.pos"])
                    # Observability publish (no consumer in the write path).
                    data_manager.set_leader_mapped_state(
                        side,
                        urdf[side],
                        (1.0 - trigger[side]) * GRIPPER_OPEN_MAX_FRAC,
                    )
            except ConnectionError as e:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise
                if consecutive_failures in (1, 10) or consecutive_failures % 50 == 0:
                    print(
                        f"⚠️  leader read failed ({consecutive_failures} in a "
                        f"row, tolerating up to {_MAX_CONSECUTIVE_FAILURES}): {e}"
                    )
                time.sleep(dt)
                continue
            consecutive_failures = 0
            leader_10 = np.concatenate([urdf["left"], urdf["right"]])

            active = (
                data_manager.get_teleop_active()
                and data_manager.get_robot_activity_state()
                == RobotActivityState.ENABLED
            )
            if active:
                if cmd is None:
                    # Engage: start from where the FOLLOWERS are, then slew
                    # toward the leader pose — no snap on the first tick.
                    measured = data_manager.get_current_joint_angles()
                    if measured is not None and len(measured) == 10:
                        cmd = np.array(measured, dtype=np.float64)
                    else:
                        cmd = leader_10.copy()
                cmd = slew_toward(cmd, leader_10, max_step_deg)
                data_manager.set_target_joint_angles(cmd)
                for side in ("left", "right"):
                    # transform=None: no IK thread reads controller transforms
                    # in leader mode; only the trigger feeds the gripper path.
                    data_manager.set_controller_state(side, None, 0.0, trigger[side])
            else:
                cmd = None  # force a fresh engage slew next activation

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Leader-arm thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🕹️  Leader-arm thread stopped")
