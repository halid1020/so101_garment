#!/usr/bin/env python3
"""Live side-by-side check of joint directions: real arm vs 3D model.

Why: the model believes your ready pose points the gripper DOWN while the
real gripper points UP — a joint moving in the OPPOSITE direction between
hardware and model (a sign flip the offset calibration cannot detect).

What happens: the motors turn off (hold the arms!), a browser view opens at
http://localhost:8080 showing the model driven by the real joint angles.
Bend each joint of each arm slowly by hand and watch the screen:

  - screen moves the SAME way as your hand  -> that joint is fine
  - screen moves the OPPOSITE way (mirror)  -> sign flip! note the joint

Fix: in src/common/configs.py set that joint's entry to -1.0 in
LEFT_ARM_HW_TO_URDF_SIGNS / RIGHT_ARM_HW_TO_URDF_SIGNS, re-run this tool to
confirm, then RE-RUN tool/fit_joint_offsets.py for that arm (offsets fitted
under a wrong sign are invalid).

Ctrl+C to exit.
"""

import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import yaml  # noqa: E402

from common.configs import (  # noqa: E402
    DUAL_URDF_PATH,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from common.robot_visualizer import RobotVisualizer  # noqa: E402
from src.so101_dual_arm import SO101DualArm  # noqa: E402

_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
    }
    print("Connecting to the arms...")
    dual_arm = SO101DualArm(config)

    print("\n⚠️  Motors will turn OFF — hold or support both arms!")
    input("Ready? Press Enter to release the motors... ")
    dual_arm.disable_torque()

    print("\n🖥️  Starting the 3D view — open http://localhost:8080")
    visualizer = RobotVisualizer(urdf_path=str(DUAL_URDF_PATH))
    visualizer.update_ghost_robot_visibility(False)

    print()
    print("Bend each joint slowly by hand and watch the screen:")
    print("  SAME direction on screen      -> joint OK")
    print("  OPPOSITE direction (mirror)   -> sign flip: set that joint to")
    print("     -1.0 in *_ARM_HW_TO_URDF_SIGNS in src/common/configs.py")
    print("Also check the POSE matches overall (offsets).")
    print("Ctrl+C to exit.")
    print()

    conv = {
        "left": (
            np.array(LEFT_ARM_HW_TO_URDF_SIGNS),
            np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG),
        ),
        "right": (
            np.array(RIGHT_ARM_HW_TO_URDF_SIGNS),
            np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG),
        ),
    }
    buses = {"left": dual_arm.bus_0, "right": dual_arm.bus_1}

    try:
        while True:
            cfg = np.zeros(12)
            for side, (i0, ig) in (("left", (0, 5)), ("right", (6, 11))):
                pos = buses[side].sync_read("Present_Position")
                signs, offs = conv[side]
                hw = np.array([pos[j] for j in _BODY_JOINTS])
                cfg[i0 : i0 + 5] = np.radians(signs * hw + offs)
                cfg[ig] = 0.0
            visualizer.update_robot_pose(cfg)
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n👋 Exiting (motors stay off — arms are limp).")
    finally:
        visualizer.stop()


if __name__ == "__main__":
    main()
