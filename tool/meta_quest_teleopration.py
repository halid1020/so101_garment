#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest (LeRobot backend).

Left Meta Quest hand → left SO101 arm (arm 0, PORT_ID_0).
Right Meta Quest hand → right SO101 arm (arm 1, PORT_ID_1).
Single 10-DOF IK solver on the dual-arm URDF.

Controls:
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Hold triggers           - close grippers
  Button A                - enable / disable both arms
  Button B                - move both arms to middle pose
  Button Y                - toggle height lock (flat table strokes)
  Ctrl+C                  - exit
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import yaml
from meta_quest_teleop.reader import MetaQuestReader

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    DUAL_URDF_PATH,
    EE_ORIENTATION_COST_MASK,
    END_EFFECTOR_FRAME_NAMES,
    FRAME_TASK_GAIN,
    IK_SOLVER_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR_DUAL,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
    WORKSPACE_OOB_MODE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.pink_ik_solver import PinkIKSolver
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from common.workspace_envelope import OOE_POLICIES
from src.so101_dual_arm import SO101DualArm


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default="production",
        choices=[
            "production",
            "pink_full",
            "pink_relaxed",
            "dls",
            "mink",
            "scipy_ls",
            "telegrip",
        ],
        help=(
            "IK layer: 'production' is the tuned Pink solver this tool has "
            "always used; the others are the sim-benchmark methods, wrapped "
            "in a joint-space rate limiter. Rehearse in simulation first "
            "with tool/quest_sim_teleop.py."
        ),
    )
    parser.add_argument(
        "--max-joint-vel",
        type=float,
        default=2.0,
        help="Joint-space rate limit (rad/s) applied to benchmark methods",
    )
    parser.add_argument(
        "--oob-mode",
        type=str,
        default=WORKSPACE_OOB_MODE,
        choices=sorted(OOE_POLICIES),
        help="Out-of-envelope target policy (see common/workspace_envelope.py)",
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

    # 3. IK layer (10 body DOF, grippers locked): production Pink solver or
    # one of the sim-benchmark methods behind the PinkIKSolver interface.
    if args.method == "production":
        print("\n🔧 Creating dual-arm Pink IK solver (production)...")
        ik_solver = PinkIKSolver(
            urdf_path=DUAL_URDF_PATH,
            end_effector_frames=END_EFFECTOR_FRAME_NAMES,
            solver_name=SOLVER_NAME,
            position_cost=POSITION_COST,
            # Anisotropic: zero cost on the EE-local yaw axis (no wrist-yaw joint)
            orientation_cost=ORIENTATION_COST * np.asarray(EE_ORIENTATION_COST_MASK),
            frame_task_gain=FRAME_TASK_GAIN,
            lm_damping=LM_DAMPING,
            damping_cost=DAMPING_COST,
            solver_damping_value=SOLVER_DAMPING_VALUE,
            integration_time_step=1.0 / IK_SOLVER_RATE,
            initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
            posture_cost_vector=np.array(POSTURE_COST_VECTOR_DUAL, dtype=float),
        )
    else:
        from sim_benchmark.method_adapter import MethodIKAdapter

        print(f"\n🔧 Creating benchmark IK method '{args.method}'...")
        ik_solver = MethodIKAdapter(
            args.method,
            dt=1.0 / IK_SOLVER_RATE,
            max_joint_vel=args.max_joint_vel,
            initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
        )

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
        kwargs={"oob_mode": args.oob_mode},
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

    def toggle_height_lock() -> None:
        enabled = data_manager.toggle_height_lock()
        print(
            "📏 Height lock "
            + ("ON — hand height ignored, strokes stay flat" if enabled else "OFF")
        )

    quest_reader.on(
        "button_a_pressed", _safe_button("Button A", toggle_robot_enabled_status)
    )
    quest_reader.on("button_b_pressed", _safe_button("Button B", on_go_home))
    quest_reader.on(
        "button_y_pressed", _safe_button("Button Y", toggle_height_lock)
    )  # TODO: let's remove this functionality.

    print()
    print("🚀 Dual-arm teleoperation ready.")
    print("   1. Press BUTTON A to enable/disable both arms")
    print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
    print("   3. Move controllers — arms follow!")
    print("   4. Hold triggers to close grippers")
    print("   5. Press BUTTON B to move both arms to the middle pose")
    print("   6. Press BUTTON Y to lock/unlock the height (flat strokes)")
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
