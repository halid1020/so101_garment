"""Common robot action helpers shared by example scripts."""

from typing import Any

from piper_controller import PiperController

from so101_garment.src.common.data_manager import DataManager, RobotActivityState


def toggle_robot_enabled(
    data_manager: DataManager,
    robot_controller: PiperController,
    visualizer: Any = None,
) -> None:
    """Safely toggles the robot between ENABLED and DISABLED states."""
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        if robot_controller:
            robot_controller.graceful_stop()
        data_manager.set_teleop_state(False, None, None)
        if visualizer:
            visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Robot disabled")

    elif state in (RobotActivityState.DISABLED, RobotActivityState.HOMING):
        if not robot_controller:
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Robot enabled (Headless)")
            return

        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            if visualizer:
                visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Robot enabled")
        else:
            print("✗ Failed to enable robot")


def move_robot_home(
    data_manager: DataManager, robot_controller: PiperController
) -> None:
    """Safely commands the robot to return to its home position."""
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        print("🏠 Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)

        if robot_controller:
            if not robot_controller.move_to_home():
                print("✗ Failed to initiate home move")
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        else:
            print("✓ 🏠 Robot homed (Headless)")
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️ Robot is not enabled, cannot go home")
