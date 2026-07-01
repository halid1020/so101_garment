#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest.
Left Quest hand -> left SO101 arm. Right Quest hand -> right SO101 arm.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

import yaml
from meta_quest_teleop.reader import MetaQuestReader

# Unified imports
from common.configs import (
    DUAL_URDF_PATH,
    END_EFFECTOR_FRAME_NAMES,
    IK_SOLVER_RATE,
    NEUTRAL_JOINT_ANGLES_DUAL,
    SOLVER_NAME,
)
from common.data_manager import DataManager
from common.pink_ik_solver import PinkIKSolver
from common.threads.ik_solver import ik_solver_thread
from src.so101_dual_arm import SO101DualArm


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation")
    parser.add_argument("--ip-address", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOPERATION (LeRobot Backend)")
    print("=" * 60)

    # 1. Initialize State Manager
    data_manager = DataManager()

    # 2. Initialize LeRobot Dual Arm Hardware
    config = {
        "robot": load_yaml("src/conf/robot.yaml"),
        "rest_pos": load_yaml("src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml("src/conf/mid_pos.yaml"),
    }
    dual_arm = SO101DualArm(config)

    # 3. Initialize Pink IK Solver
    ik_solver = PinkIKSolver(
        urdf_path=DUAL_URDF_PATH,
        end_effector_frames=END_EFFECTOR_FRAME_NAMES,
        solver_name=SOLVER_NAME,
        integration_time_step=1.0 / IK_SOLVER_RATE,
        initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
    )

    # 4. Initialize Quest Reader & Threads
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    ik_thread = threading.Thread(
        target=ik_solver_thread,
        args=(data_manager, ik_solver),
        daemon=True,
    )
    ik_thread.start()

    print("\n🚀 Ready. Hold LEFT + RIGHT GRIP to activate teleoperation.")

    try:
        while True:
            # Main sync loop mapping VR to LeRobot hardware
            if data_manager.get_teleop_active():
                target_pose = data_manager.get_target_pose()
                if target_pose is not None:
                    # Pass the computed IK directly to the dual arm Cartesian wrapper
                    dual_arm.move_to_ee_pose(
                        target_pose, target_pose, duration=0.02, ik_solver=ik_solver
                    )
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        data_manager.request_shutdown()
        quest_reader.stop()
        dual_arm.disable_torque()


if __name__ == "__main__":
    main()
