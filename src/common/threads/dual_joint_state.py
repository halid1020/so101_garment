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

from common.configs import JOINT_STATE_STREAMING_RATE
from common.data_manager_dual import DualDataManager, RobotActivityState

_BODY_DOF = 5  # SO101 has 5 actuated body joints per arm
_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


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

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # ── Read current state from hardware ─────────────────────────────
            with bus_lock:
                positions = bus.sync_read("Present_Position")
            current_joint_angles = np.array(
                [positions[j] for j in _BODY_JOINTS], dtype=np.float64
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

                    goal = dict(zip(_BODY_JOINTS, arm_targets))

                    # Trigger controls gripper: fully pressed = fully closed.
                    gripper_target = 1.0 - trigger_value
                    goal["gripper"] = gripper_target * 100.0
                    data_manager.set_target_gripper_open_value(arm_side, gripper_target)

                    with bus_lock:
                        bus.sync_write("Goal_Position", goal, normalize=True)

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
