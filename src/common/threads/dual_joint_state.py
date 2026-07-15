"""Joint state thread for one arm in a dual-arm SO101 setup (LeRobot backend).

Adapted from example_so101's dual_joint_state.py: reads joint angles from one
FeetechMotorsBus, writes them into the shared 10-DOF DualDataManager array
(left: [0:5], right: [5:10]), and forwards IK target angles + gripper trigger
commands to the physical arm when teleop is active.
"""

import threading
import time
import traceback

import numpy as np

from common.configs import (
    GRIPPER_OPEN_MAX_FRAC,
    JOINT_STATE_STREAMING_RATE,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from common.data_manager_dual import DualDataManager, RobotActivityState

_BODY_DOF = 5  # SO101 has 5 actuated body joints per arm
_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# Everything in DualDataManager / IK / visualizer lives in URDF joint space;
# only this thread talks to the motors, so the hw<->URDF sign+zero-offset
# conversion happens here on read and write. Both are per arm (each servo
# was zeroed and oriented independently during LeRobot calibration).
# urdf = sign * hw + offset ; hw = sign * (urdf - offset)  [sign is +-1]
_HW_TO_URDF_OFFSETS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
}
_HW_TO_URDF_SIGNS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
}


def dual_joint_state_thread(
    data_manager: DualDataManager,
    bus,
    arm_side: str,
    bus_lock: threading.Lock,
) -> None:
    """Read/write joint state for one arm in the dual-arm setup.

    Args:
        data_manager: Shared dual-arm DataManager.
        bus: FeetechMotorsBus for this arm (motors in degrees, gripper 0-100).
        arm_side: "left" or "right" — determines which slice of the
                  10-DOF array this thread owns.
        bus_lock: Lock serializing all access to this bus's serial port —
                  the Feetech port handler is not thread-safe, so button
                  callbacks touching the same bus must hold this lock too.
    """
    if arm_side not in ("left", "right"):
        raise ValueError("arm_side must be 'left' or 'right'")

    print(f"🔧 Dual joint state thread started ({arm_side})")
    dt: float = 1.0 / JOINT_STATE_STREAMING_RATE
    hw_to_urdf = _HW_TO_URDF_OFFSETS[arm_side]
    hw_signs = _HW_TO_URDF_SIGNS[arm_side]

    # Serial robustness: the Feetech bus occasionally drops a status packet
    # (EMI / USB latency), which LeRobot surfaces as a ConnectionError. A
    # single dropped packet must NOT kill the thread (and with it the whole
    # teleop session) — skip the tick and retry. Only give up after
    # _MAX_CONSECUTIVE_FAILURES in a row (a genuinely dead bus).
    _MAX_CONSECUTIVE_FAILURES = 100  # = 1 s at 100 Hz
    consecutive_failures = 0

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # ── Read current state from hardware ─────────────────────────────
            try:
                with bus_lock:
                    positions = bus.sync_read("Present_Position", num_retry=2)
            except ConnectionError as e:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise
                if consecutive_failures in (1, 10) or consecutive_failures % 50 == 0:
                    print(
                        f"⚠️  {arm_side} bus read failed "
                        f"({consecutive_failures} in a row, tolerating up to "
                        f"{_MAX_CONSECUTIVE_FAILURES}): {e}"
                    )
                time.sleep(dt)
                continue
            consecutive_failures = 0
            current_joint_angles = (
                hw_signs
                * np.array([positions[j] for j in _BODY_JOINTS], dtype=np.float64)
                + hw_to_urdf
            )

            combined = data_manager.get_current_joint_angles()
            if combined is None or len(combined) != _BODY_DOF * 2:
                combined = np.zeros(_BODY_DOF * 2, dtype=np.float64)
            else:
                combined = combined.copy()
            if arm_side == "left":
                combined[:_BODY_DOF] = current_joint_angles
            else:
                combined[_BODY_DOF:] = current_joint_angles
            data_manager.set_current_joint_angles(combined)

            # Gripper position is normalized 0-100 (0 = closed, 100 = open)
            data_manager.set_current_gripper_open_value(
                arm_side, positions["gripper"] / 100.0
            )

            # ── Commands ─────────────────────────────────────────────────────
            target_joint_angles = data_manager.get_target_joint_angles()
            _, _, trigger_value = data_manager.get_controller_state(arm_side)

            robot_activity_state = data_manager.get_robot_activity_state()

            if robot_activity_state == RobotActivityState.ENABLED:
                if (
                    target_joint_angles is not None
                    and data_manager.get_teleop_active()
                    and len(target_joint_angles) >= _BODY_DOF * 2
                ):
                    if arm_side == "left":
                        arm_targets = target_joint_angles[:_BODY_DOF]
                    else:
                        arm_targets = target_joint_angles[_BODY_DOF:]

                    goal = dict(
                        zip(_BODY_JOINTS, hw_signs * (arm_targets - hw_to_urdf))
                    )

                    # Trigger controls gripper: fully pressed = fully closed,
                    # fully released = open_max_frac of the jaw range (capped
                    # below 100 % — see GRIPPER_OPEN_MAX_FRAC / teleop_shared).
                    gripper_target = (1.0 - trigger_value) * GRIPPER_OPEN_MAX_FRAC
                    goal["gripper"] = gripper_target * 100.0
                    data_manager.set_target_gripper_open_value(arm_side, gripper_target)

                    try:
                        with bus_lock:
                            bus.sync_write(
                                "Goal_Position", goal, normalize=True, num_retry=2
                            )
                        # Publish the command actually sent (URDF-space body
                        # targets + gripper open fraction) for the recorder's
                        # 'action' feature. Only after a successful write.
                        data_manager.set_last_sent_command(
                            arm_side,
                            np.asarray(arm_targets, dtype=np.float64),
                            gripper_target,
                            time.monotonic(),
                        )
                    except ConnectionError as e:
                        # Transient write failure: drop this command tick —
                        # the next one supersedes it anyway.
                        print(f"⚠️  {arm_side} bus write failed (skipped): {e}")

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Dual joint state thread error ({arm_side}): {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print(f"🔧 Dual joint state thread stopped ({arm_side})")
