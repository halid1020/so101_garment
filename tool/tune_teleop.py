#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation tuning - real robot control.

Runs the same Quest teleoperation pipeline as tool/meta_quest_teleopration.py
plus a Viser web UI (http://localhost:8080) where IK weights, 1-Euro filter
coefficients, and motion scaling can be adjusted live while teleoperating.
A "Save Config" button writes the current slider values to
src/ik_conf/tuned_teleop_configs.yaml.

Controls:
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Hold triggers           - close grippers
  Button A                - enable (go to ready pose) / disable (go to rest)
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

from common.config_parser import load_ik_config
from common.configs import (
    DUAL_URDF_PATH,
    EE_ORIENTATION_COST_MASK,
    END_EFFECTOR_FRAME_NAMES,
    IK_SOLVER_RATE,
    NEUTRAL_JOINT_ANGLES_DUAL,
    SOLVER_NAME,
    VISUALIZATION_RATE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.pink_ik_solver import PinkIKSolver
from common.robot_visualizer import RobotVisualizer
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from src.so101_dual_arm import SO101DualArm

_BODY_DOF = 5

# yourdfpy actuated joint order in the dual URDF:
# [left x5, left_gripper, right x5, right_gripper]
_LEFT_GRIPPER_IDX = 5
_RIGHT_GRIPPER_IDX = 11


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def to_urdf_config(body_deg: np.ndarray) -> np.ndarray:
    """Expand a 10-DOF body-joint vector (degrees) to the 12-joint URDF
    configuration (radians) the visualizer expects, with grippers at 0."""
    cfg = np.zeros(12, dtype=np.float64)
    cfg[0:_BODY_DOF] = np.radians(body_deg[0:_BODY_DOF])
    cfg[_LEFT_GRIPPER_IDX + 1 : _RIGHT_GRIPPER_IDX] = np.radians(
        body_deg[_BODY_DOF : 2 * _BODY_DOF]
    )
    return cfg


def save_config_to_yaml(visualizer: RobotVisualizer, filepath: Path) -> None:
    """Extract all current UI slider values and save them to a YAML file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        pink_params = visualizer.get_pink_parameters()
        min_c, beta, d_c = visualizer.get_controller_filter_params()

        # Cast to plain python types so PyYAML doesn't choke on numpy scalars
        config_dict = {
            "ik_parameters": {
                "position_cost": float(pink_params["position_cost"]),
                "orientation_cost": float(pink_params["orientation_cost"]),
                "frame_task_gain": float(pink_params["frame_task_gain"]),
                "lm_damping": float(pink_params["lm_damping"]),
                "damping_cost": float(pink_params["damping_cost"]),
                "solver_damping_value": float(pink_params["solver_damping_value"]),
                "posture_cost_vector": [
                    float(v) for v in pink_params["posture_cost_vector"]
                ],
            },
            "filter_parameters": {
                "controller_min_cutoff": float(min_c),
                "controller_beta": float(beta),
                "controller_d_cutoff": float(d_c),
            },
            "teleop_parameters": {
                "translation_scale": float(visualizer.get_translation_scale()),
                "rotation_scale": float(visualizer.get_rotation_scale()),
                # NEW: Save the advanced parameters from the GUI
                "slow_translation_scale": float(
                    visualizer.get_slow_translation_scale()
                ),
                "slow_rotation_scale": float(visualizer.get_slow_rotation_scale()),
                "wrist_joint_button_step_degrees": float(
                    visualizer.get_wrist_step_degrees()
                ),
            },
        }

        with open(filepath, "w") as file:
            yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
        print(f"\n✅ SUCCESS: Configuration saved to {filepath}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save YAML config: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation tuning")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--ik-config",
        type=str,
        default=str(_root / "src/ik_conf/so101_dual_default.yaml"),
        help="Path to IK/teleop YAML config.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOPERATION TUNING (LeRobot Backend)")
    print("=" * 60)

    ik_config = load_ik_config(args.ik_config)
    ik_p = ik_config.get("ik_parameters", {})
    filt_p = ik_config.get("filter_parameters", {})
    tele_p = ik_config.get("teleop_parameters", {})

    # 1. Shared state
    data_manager = DualDataManager()
    data_manager.set_controller_filter_params(
        filt_p.get("controller_min_cutoff", 0.8),
        filt_p.get("controller_beta", 5.0),
        filt_p.get("controller_d_cutoff", 0.9),
    )
    data_manager.set_teleop_scaling(
        tele_p.get("translation_scale", 1.0),
        tele_p.get("rotation_scale", 1.0),
    )

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

    # 3. Dual-arm Pink IK solver (10 body DOF, grippers locked)
    posture_cost_vector = np.array(
        ik_p.get("posture_cost_vector", [0.0] * len(NEUTRAL_JOINT_ANGLES_DUAL)),
        dtype=float,
    )
    print("\n🔧 Creating dual-arm Pink IK solver...")
    ik_solver = PinkIKSolver(
        urdf_path=DUAL_URDF_PATH,
        end_effector_frames=END_EFFECTOR_FRAME_NAMES,
        solver_name=SOLVER_NAME,
        position_cost=ik_p.get("position_cost", 1.0),
        # Anisotropic: zero cost on the EE-local yaw axis (no wrist-yaw joint)
        orientation_cost=ik_p.get("orientation_cost", 0.75)
        * np.asarray(EE_ORIENTATION_COST_MASK),
        frame_task_gain=ik_p.get("frame_task_gain", 0.4),
        lm_damping=ik_p.get("lm_damping", 0.0),
        damping_cost=ik_p.get("damping_cost", 0.25),
        solver_damping_value=ik_p.get("solver_damping_value", 1e-12),
        integration_time_step=1.0 / IK_SOLVER_RATE,
        initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
        posture_cost_vector=posture_cost_vector,
    )

    # 4. Quest reader
    print("\n🎮 Initializing Meta Quest reader...")
    try:
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
    except (Exception, SystemExit):
        print("\n" + "!" * 60)
        print("❌ FAILED TO ACCESS META QUEST")
        print("!" * 60)
        print("PLEASE FOLLOW THESE STEPS:")
        print("  1. Put the Meta Quest headset on your head.")
        print("  2. Look for a 'USB Detected' notification.")
        print("  3. Click it and select 'Allow' to grant data access.")
        print("  4. Rerun this script.")
        print("!" * 60 + "\n")
        dual_arm.disable_torque()
        sys.exit(1)

    # 5. Threads: per-arm joint state I/O + dual IK solver.
    # Feetech serial ports are not thread-safe: every bus access (joint
    # threads AND button callbacks) must hold that bus's lock.
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
        daemon=True,
    )
    left_joint_thread.start()
    right_joint_thread.start()
    ik_thread.start()

    # 6. Visualizer web UI with tuning sliders
    print("\n🖥️  Starting visualization...")
    visualizer = RobotVisualizer(urdf_path=DUAL_URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_robot_status_controls()
    visualizer.add_teleop_controls()
    visualizer.add_ee_pose_displays()
    visualizer.add_homing_controls()
    visualizer.add_toggle_robot_enabled_status_button()
    # NEW: Add advanced tuning controls (which implicitly adds the Save Config button)
    visualizer.add_advanced_tuning_controls(
        initial_slow_t=tele_p.get("slow_translation_scale", 0.6),
        initial_slow_r=tele_p.get("slow_rotation_scale", 0.6),
        initial_wrist_step=tele_p.get("wrist_joint_button_step_degrees", 5.0),
    )

    visualizer.add_controller_filter_controls(
        filt_p.get("controller_min_cutoff", 0.8),
        filt_p.get("controller_beta", 5.0),
        filt_p.get("controller_d_cutoff", 0.9),
    )
    visualizer.add_scaling_controls(
        tele_p.get("translation_scale", 1.0),
        tele_p.get("rotation_scale", 1.0),
    )
    visualizer.add_pink_parameter_controls(
        ik_p.get("position_cost", 1.0),
        ik_p.get("orientation_cost", 0.75),
        ik_p.get("frame_task_gain", 0.4),
        ik_p.get("lm_damping", 0.0),
        ik_p.get("damping_cost", 0.25),
        ik_p.get("solver_damping_value", 1e-12),
        list(posture_cost_vector),
    )

    # 7. Button callbacks (Quest buttons + web UI buttons).
    # MUST be crash-proof: the quest reader dispatches callbacks without an
    # except clause, so a raised exception kills its thread (no more buttons
    # OR hand tracking). And any early-exit path must restore a state the
    # toggle recognizes, or the button goes permanently silent.
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
        else:  # HOMING — a move is in progress (or a crash left it stuck)
            print(
                "⚠️  Busy (HOMING) — if this persists, the previous move "
                "failed; state auto-resets on error, try again."
            )

    def on_go_home() -> None:
        state = data_manager.get_robot_activity_state()
        if state == RobotActivityState.ENABLED:
            print("🏠 Moving both arms to middle pose...")
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            data_manager.set_teleop_state(False)
            with left_bus_lock, right_bus_lock:
                dual_arm.send_to_middle(2.0)
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ Both arms at middle pose and re-enabled")
        else:
            print("⚠️  Cannot home: arms not enabled")

    toggle_robot_enabled_status = _safe_button("Button A", toggle_robot_enabled_status)
    on_go_home = _safe_button("Button B", on_go_home)

    def toggle_height_lock() -> None:
        enabled = data_manager.toggle_height_lock()
        print(
            f"📏 Height lock {'ON — hand height ignored, strokes stay flat' if enabled else 'OFF'}"
        )

    quest_reader.on("button_a_pressed", toggle_robot_enabled_status)
    quest_reader.on("button_b_pressed", on_go_home)
    quest_reader.on("button_y_pressed", _safe_button("Button Y", toggle_height_lock))
    visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled_status)
    visualizer.set_go_home_callback(on_go_home)
    visualizer.set_save_config_callback(
        lambda: save_config_to_yaml(
            visualizer, _root / "src/ik_conf/tuned_teleop_configs.yaml"
        )
    )

    print()
    print("🚀 Tuning ready. Open http://localhost:8080 to tune.")
    print("   1. Press BUTTON A (or web button) to enable/disable both arms")
    print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
    print("   3. Adjust sliders live, then click 'Save Config' in the web UI")
    print("   4. Press BUTTON Y to lock/unlock the height (flat strokes)")
    print("⚠️  Press Ctrl+C to exit")
    print()

    # 8. Main visualization/tuning loop
    dt = 1.0 / VISUALIZATION_RATE
    last_pink_params: dict = {}
    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # Push slider values into the live system
            data_manager.set_teleop_scaling(
                visualizer.get_translation_scale(),
                visualizer.get_rotation_scale(),
            )
            data_manager.set_controller_filter_params(
                *visualizer.get_controller_filter_params()
            )
            pink_params = visualizer.get_pink_parameters()
            comparable = {
                k: (tuple(v) if isinstance(v, np.ndarray) else v)
                for k, v in pink_params.items()
            }
            if comparable != last_pink_params:
                # The slider is a scalar; expand it through the anisotropic
                # mask so the EE-local yaw axis stays zero-costed.
                masked = dict(pink_params)
                masked["orientation_cost"] = pink_params[
                    "orientation_cost"
                ] * np.asarray(EE_ORIENTATION_COST_MASK)
                ik_solver.update_task_parameters(**masked)
                last_pink_params = comparable

            # Update GUI displays (grip/trigger: strongest of either hand)
            _, left_grip, left_trigger = data_manager.get_controller_state("left")
            _, right_grip, right_trigger = data_manager.get_controller_state("right")
            visualizer.set_grip_value(max(left_grip, right_grip))
            visualizer.set_trigger_value(max(left_trigger, right_trigger))
            visualizer.update_timing(data_manager.get_ik_solve_time_ms())
            visualizer.update_teleop_status(data_manager.get_teleop_active())

            state = data_manager.get_robot_activity_state()
            visualizer.update_robot_status(f"Robot Status: {state.value.capitalize()}")
            visualizer.update_toggle_robot_enabled_status(
                state == RobotActivityState.ENABLED
            )

            # Render actual robot
            current_joints = data_manager.get_current_joint_angles()
            if current_joints is not None:
                cfg = to_urdf_config(current_joints)
                visualizer.update_robot_pose(cfg)
                visualizer.update_joint_angles_display(cfg, show_gripper=True)

            # Render ghost robot at the IK target
            target_joints = data_manager.get_target_joint_angles()
            if target_joints is not None and state == RobotActivityState.ENABLED:
                visualizer.update_ghost_robot_visibility(True)
                visualizer.update_ghost_robot_pose(to_urdf_config(target_joints))
            else:
                visualizer.update_ghost_robot_visibility(False)

            target_left = data_manager.get_target_pose("left")
            target_right = data_manager.get_target_pose("right")

            # Get actual IK calculated poses directly from the solver
            ik_poses = ik_solver.get_current_end_effector_poses()
            ik_left = ik_poses.get(END_EFFECTOR_FRAME_NAMES[0])
            ik_right = ik_poses.get(END_EFFECTOR_FRAME_NAMES[1])

            visualizer.update_ee_poses_display("left", target_left, ik_left)
            visualizer.update_ee_poses_display("right", target_right, ik_right)

            time.sleep(max(0, dt - (time.time() - iteration_start)))

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
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
        visualizer.stop()
        print("👋 Done.")


if __name__ == "__main__":
    main()
