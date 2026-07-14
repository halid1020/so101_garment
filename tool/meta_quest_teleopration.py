#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest (LeRobot backend).

Left Meta Quest hand → left SO101 arm (arm 0, PORT_ID_0).
Right Meta Quest hand → right SO101 arm (arm 1, PORT_ID_1).
Single 10-DOF IK solver on the dual-arm URDF.

Controls:
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Hold triggers           - close grippers
  Thumbstick (mymethod)   - deflect to trim that arm's wrist (x = roll,
                            y = flex); the arm's other joints freeze while
                            the stick is deflected and the handle is ignored
                            for that arm, then resumes from the new pose on
                            release
  Button A                - enable / disable both arms
  Button B                - move both arms to middle pose
  Joystick clicks (LJ/RJ) - glide that gripper's roll back to neutral
                            at the next grip
  Ctrl+C                  - exit
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import yaml
from meta_quest_teleop.reader import MetaQuestReader

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    IK_SOLVER_RATE,
    MAX_JOINT_VEL_HW_RAD_S,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.teleop_setup import add_teleop_cli_args, create_teleop_stack
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from src.so101_dual_arm import SO101DualArm


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation")
    parser.add_argument("--ip-address", type=str, default=None)
    add_teleop_cli_args(
        parser, default_max_joint_vel=MAX_JOINT_VEL_HW_RAD_S, default_method="armplane"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOPERATION (LeRobot Backend)")
    print("=" * 60)

    # 1. Shared state
    data_manager = DualDataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    # 2. LeRobot dual arm hardware (arm 0 = left, arm 1 = right)
    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
        "ready_pos": load_yaml(_root / "src/conf/ready_pos.yaml"),
    }
    dual_arm = SO101DualArm(config)
    ready_pos = config["ready_pos"]
    rest_pos = config["rest_pos"]

    # 3. IK layer (10 body DOF, grippers locked): built by the shared helper so
    # this tool and the sim rehearsal (tool/quest_sim_teleop.py) cannot drift.
    # 'mymethod' reuses the pink_relaxed solver plus the thumbstick wrist trims
    # (--wrist-mode); armplane keeps the tuned Pink solver + armplane mapping.
    ik_solver, thread_kwargs = create_teleop_stack(args, dt=1.0 / IK_SOLVER_RATE)

    # 4. Quest reader (IK thread reads it directly)
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    # 5. Threads: per-arm joint state I/O + dual IK solver.
    # The Feetech serial port handler is not thread-safe: every bus access
    # (joint threads AND quest button callbacks) must hold that bus's lock.
    left_bus_lock = threading.Lock()
    right_bus_lock = threading.Lock()
    left_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_0, "left", left_bus_lock),
        daemon=True,
    )
    right_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_1, "right", right_bus_lock),
        daemon=True,
    )
    ik_thread = threading.Thread(
        target=dual_ik_solver_thread,
        args=(data_manager, ik_solver, quest_reader),
        kwargs=thread_kwargs,
        daemon=True,
    )
    left_joint_thread.start()
    right_joint_thread.start()
    ik_thread.start()

    # 6. Quest button callbacks.
    # MUST be crash-proof: the quest reader dispatches callbacks without an
    # except clause, so a raised exception kills its thread (no more buttons
    # OR hand tracking), and a crash mid-move would leave the state stuck in
    # HOMING, making the A button silently dead.
    def _safe_button(name, fn):
        def wrapped() -> None:
            print(
                f"[{name}] pressed "
                f"(state={data_manager.get_robot_activity_state().value})"
            )
            try:
                fn()
            except Exception:
                traceback.print_exc()
                print(
                    f"❌ [{name}] handler failed (see traceback above). "
                    "Torque off, state reset to DISABLED — press A to retry."
                )
                try:
                    with left_bus_lock, right_bus_lock:
                        dual_arm.disable_torque()
                except Exception:
                    traceback.print_exc()
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

        return wrapped

    def toggle_robot_enabled_status() -> None:
        state = data_manager.get_robot_activity_state()
        if state == RobotActivityState.ENABLED:
            # HOMING first so the joint threads stop sending teleop commands
            # while we drive the arms to the rest pose, then cut torque.
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            data_manager.set_teleop_state(False)
            print("🔴 Disabling: moving both arms to rest pose...")
            with left_bus_lock, right_bus_lock:
                dual_arm.move_to_joint_pose(rest_pos, rest_pos, 2.0)
                dual_arm.bus_0.disable_torque()
                dual_arm.bus_1.disable_torque()
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            print("✓ 🔴 Both arms at rest and disabled (torque off)")
        elif state == RobotActivityState.DISABLED:
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            print("🟢 Enabling: moving both arms to ready pose...")
            with left_bus_lock, right_bus_lock:
                dual_arm.bus_0.enable_torque()
                dual_arm.bus_1.enable_torque()
                dual_arm.move_to_joint_pose(ready_pos, ready_pos, 2.0)
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Both arms at ready pose and enabled")

    def on_go_home() -> None:
        state = data_manager.get_robot_activity_state()
        if state in (RobotActivityState.ENABLED, RobotActivityState.HOMING):
            print("🏠 Moving both arms to middle pose...")
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            data_manager.set_teleop_state(False)
            with left_bus_lock, right_bus_lock:
                dual_arm.send_to_middle(2.0)
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ Both arms at middle pose and re-enabled")
        else:
            print("⚠️  Cannot home: arms not enabled")

    quest_reader.on(
        "button_a_pressed", _safe_button("Button A", toggle_robot_enabled_status)
    )
    quest_reader.on("button_b_pressed", _safe_button("Button B", on_go_home))
    quest_reader.on(
        "button_lj_pressed",
        _safe_button(
            "Left joystick click",
            lambda: data_manager.request_roll_reset("left"),
        ),
    )
    quest_reader.on(
        "button_rj_pressed",
        _safe_button(
            "Right joystick click",
            lambda: data_manager.request_roll_reset("right"),
        ),
    )

    print()
    print("🚀 Dual-arm teleoperation ready.")
    print("   1. Press BUTTON A to enable/disable both arms")
    print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
    print("   3. Move controllers — arms follow!")
    print("   4. Hold triggers to close grippers")
    print("   5. Press BUTTON B to move both arms to the middle pose")
    if args.method == "mymethod":
        print(
            "   6. Deflect a THUMBSTICK to trim that arm's wrist (x = roll, "
            "y = flex); its other joints freeze while deflected, then the "
            "handle resumes from the new pose on release"
        )
    print("⚠️  Press Ctrl+C to exit")
    print()

    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received — shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        data_manager.request_shutdown()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        quest_reader.stop()
        ik_thread.join(timeout=3.0)
        left_joint_thread.join(timeout=3.0)
        right_joint_thread.join(timeout=3.0)
        with left_bus_lock, right_bus_lock:
            dual_arm.disable_torque()
        print("👋 Done.")


if __name__ == "__main__":
    main()
