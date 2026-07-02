"""Dual-arm IK solver thread for SO101 Quest teleoperation.

Ported from example_openarm's ik_solver.py. Reads both hand transforms
directly from the MetaQuestReader and drives a single 10-DOF PinkIKSolver
(dual-arm URDF, grippers locked). Each arm has its own calibration transform
computed on teleop activation.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np
import pinocchio as pin

from common.configs import (
    IK_SOLVER_RATE,
    LEFT_END_EFFECTOR_FRAME_NAME,
    ORIENTATION_BLEND_TIME_S,
    RIGHT_END_EFFECTOR_FRAME_NAME,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.pink_ik_solver import PinkIKSolver
from common.utils import (
    blend_rotations,
    compute_hand_to_robot_calibration,
    hand_to_gripper_orientation_armplane,
    map_quest_hands_to_robot_arms,
)

_DIVERGENCE_TOLERANCE_DEG = 0.1


def _sync_targets_from_ik(
    data_manager: DualDataManager,
    ik_solver: PinkIKSolver,
    joint_angles_deg: np.ndarray,
) -> None:
    """Push joint and EEF targets from the current IK config into DataManager."""
    data_manager.set_target_joint_angles(joint_angles_deg)
    ee_poses = ik_solver.get_current_end_effector_poses()
    data_manager.set_target_pose("left", ee_poses.get(LEFT_END_EFFECTOR_FRAME_NAME))
    data_manager.set_target_pose("right", ee_poses.get(RIGHT_END_EFFECTOR_FRAME_NAME))
    data_manager.set_ik_success(True)
    data_manager.set_ik_solve_time_ms(0.0)


def dual_ik_solver_thread(
    data_manager: DualDataManager,
    ik_solver: PinkIKSolver,
    quest_reader: Any | None = None,
) -> None:
    """Dual-arm IK solver thread.

    With quest_reader: absolute pointer poses in the head frame map to robot
    TCP targets each frame. A one-time calibration at grip-press aligns head
    frame to the robot. Left hand → left arm, right hand → right arm.
    Without quest_reader: Viser gizmos drive absolute TCP targets via DataManager.
    """
    input_label = "Quest" if quest_reader is not None else "Viser gizmo"
    print(f"🧮 Dual IK solver thread started ({input_label})")

    # Fixed base positions of each arm (for the tip-azimuth computation)
    _m = ik_solver.urdf_model
    _d = _m.createData()
    _q0 = pin.neutral(_m)
    pin.forwardKinematics(_m, _d, _q0)
    pin.updateFramePlacements(_m, _d)
    base_xy = {
        side: _d.oMf[_m.getFrameId(f"{side}_base_link")].translation[:2].copy()
        for side in ("left", "right")
    }
    # Arm azimuths (compass direction of the EE from its base), kept between
    # cycles so a near-vertical/retracted pose doesn't produce noise.
    azimuths: dict[str, float] = {}
    last_debug_time = 0.0

    # Handle axes are captured PER HAND at the FIRST grip of the session:
    # the operator holds both handles pointing straight down, and whatever
    # body-frame direction is "world down" at that instant is that hand's
    # handle axis. (The two Quest controllers are mirrored hardware, and
    # their body frames are not stable across app sessions, so a config
    # constant cannot express this.)
    handle_axes: dict[str, np.ndarray] = {}
    knuckle_axes: dict[str, np.ndarray] = {}
    # Height lock: when toggled on, target z freezes at each arm's current
    # height — hand z is ignored, so table-plane strokes stay perfectly flat.
    height_lock_prev = False
    locked_z: dict[str, float] = {}
    _WORLD_DOWN = np.array([0.0, 0.0, -1.0])
    if quest_reader is not None:
        print(
            "🖐️  FIRST GRIP CALIBRATES: when you first hold both grips, "
            "point both HANDLES straight down (like two nails) — "
            "handle-down = gripper-down for the rest of the session."
        )

    def _update_azimuth(side: str, ee_pose: np.ndarray | None) -> float:
        if ee_pose is not None:
            vec = ee_pose[:2, 3] - base_xy[side]
            if np.linalg.norm(vec) > 0.04:
                azimuths[side] = float(np.arctan2(vec[1], vec[0]))
        return azimuths.get(side, 0.0)

    dt: float = 1.0 / IK_SOLVER_RATE
    left_hand_to_robot: np.ndarray | None = None
    right_hand_to_robot: np.ndarray | None = None
    left_hand_reference: np.ndarray | None = None
    right_hand_reference: np.ndarray | None = None
    # Orientation blend state: EE rotations at activation, ramped toward the
    # absolute hand orientation over ORIENTATION_BLEND_TIME_S.
    left_rot_at_activation: np.ndarray | None = None
    right_rot_at_activation: np.ndarray | None = None
    activation_time: float | None = None
    teleop_active_prev = False
    mirror_control_prev = False

    def _reset_teleop_calibration() -> None:
        nonlocal left_hand_to_robot, right_hand_to_robot
        nonlocal left_hand_reference, right_hand_reference
        nonlocal left_rot_at_activation, right_rot_at_activation
        nonlocal activation_time
        left_rot_at_activation = None
        right_rot_at_activation = None
        activation_time = None
        left_hand_to_robot = None
        right_hand_to_robot = None
        left_hand_reference = None
        right_hand_reference = None

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()
            mirror_control = data_manager.get_mirror_control_enabled()
            if mirror_control != mirror_control_prev:
                if left_hand_to_robot is not None or right_hand_to_robot is not None:
                    _reset_teleop_calibration()
                    print(
                        "↔️  Mirror control "
                        + ("enabled" if mirror_control else "disabled")
                        + " — release grips and re-press to recalibrate teleop"
                    )
                mirror_control_prev = mirror_control

            if quest_reader is not None:
                quest_left_tf = quest_reader.get_hand_controller_transform_ros(
                    hand="left"
                )
                quest_right_tf = quest_reader.get_hand_controller_transform_ros(
                    hand="right"
                )
                right_grip = quest_reader.get_grip_value("right")
                right_trigger = quest_reader.get_trigger_value("right")
                left_grip_for_dm = quest_reader.get_grip_value("left")
                left_trigger_for_dm = quest_reader.get_trigger_value("left")

                if quest_left_tf is not None and quest_right_tf is not None:
                    left_tf_for_dm, right_tf_for_dm = map_quest_hands_to_robot_arms(
                        quest_left_tf, quest_right_tf, mirror_control=mirror_control
                    )
                else:
                    left_tf_for_dm = quest_left_tf
                    right_tf_for_dm = quest_right_tf

                data_manager.set_controller_state(
                    "right", right_tf_for_dm, right_grip, right_trigger
                )
                data_manager.set_controller_state(
                    "left", left_tf_for_dm, left_grip_for_dm, left_trigger_for_dm
                )

            current_joint_angles = data_manager.get_current_joint_angles()
            robot_activity_state = data_manager.get_robot_activity_state()

            if quest_reader is not None:
                left_grip = quest_reader.get_grip_value("left")
                right_grip = quest_reader.get_grip_value("right")
                teleop_active = (
                    left_tf_for_dm is not None
                    and right_tf_for_dm is not None
                    and left_grip >= 0.9
                    and right_grip >= 0.9
                )
                data_manager.set_teleop_state(teleop_active)
                # Use the One-Euro-FILTERED transforms from the DataManager
                # for target mapping (set_controller_state above filtered
                # them). Using the raw reader transforms here would bypass
                # the smoothing entirely and shake the arms.
                left_tf, _, _ = data_manager.get_controller_state("left")
                right_tf, _, _ = data_manager.get_controller_state("right")
            else:
                teleop_active = data_manager.get_teleop_active()
                left_tf = None
                right_tf = None

            # Anchor IK to measured joints whenever teleop is not active.
            if current_joint_angles is not None:
                if not teleop_active:
                    ik_solver.set_configuration_no_task_update(
                        np.radians(current_joint_angles)
                    )
                else:
                    current_ik_joint_angles = np.degrees(
                        ik_solver.get_current_configuration()
                    )
                    if current_ik_joint_angles is not None and np.all(
                        np.abs(current_joint_angles - current_ik_joint_angles)
                        <= _DIVERGENCE_TOLERANCE_DEG
                    ):
                        ik_solver.set_configuration_no_task_update(
                            np.radians(current_joint_angles)
                        )

            current_poses = ik_solver.get_current_end_effector_poses()
            left_pose = current_poses.get(LEFT_END_EFFECTOR_FRAME_NAME)
            right_pose = current_poses.get(RIGHT_END_EFFECTOR_FRAME_NAME)
            if right_pose is not None:
                data_manager.set_current_end_effector_pose("right", right_pose)
            if left_pose is not None:
                data_manager.set_current_end_effector_pose("left", left_pose)

            # On rising edge of teleop, compute per-arm calibration transforms.
            if (
                quest_reader is not None
                and teleop_active
                and not teleop_active_prev
                and robot_activity_state == RobotActivityState.ENABLED
                and left_tf is not None
                and right_tf is not None
                and left_pose is not None
                and right_pose is not None
            ):
                translation_scale, rotation_scale = data_manager.get_teleop_scaling()
                left_hand_reference = left_tf.copy()
                right_hand_reference = right_tf.copy()
                left_hand_to_robot = compute_hand_to_robot_calibration(
                    left_pose,
                    left_tf,
                    left_hand_reference,
                    translation_scale,
                    rotation_scale,
                )
                right_hand_to_robot = compute_hand_to_robot_calibration(
                    right_pose,
                    right_tf,
                    right_hand_reference,
                    translation_scale,
                    rotation_scale,
                )
                # Blend from the EE's orientation at activation toward the
                # absolute hand orientation, so activation never jerks.
                left_rot_at_activation = left_pose[:3, :3].copy()
                right_rot_at_activation = right_pose[:3, :3].copy()
                activation_time = time.time()
                # First grip of the session: capture each hand's handle axis
                # (operator is holding the handles pointing straight down)
                # and its roll reference ("knuckle" axis) — a horizontal
                # world direction taken from the gripper's current roll, so
                # it is perpendicular to the handle by construction and the
                # roll response is strong in every session.
                if "left" not in handle_axes:
                    for _key, _tf, _pose in (
                        ("left", left_tf, left_pose),
                        ("right", right_tf, right_pose),
                    ):
                        handle_axes[_key] = _tf[:3, :3].T @ _WORLD_DOWN
                        khat = _pose[:3, 2].copy()
                        khat[2] = 0.0
                        n = np.linalg.norm(khat)
                        khat = khat / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
                        knuckle_axes[_key] = _tf[:3, :3].T @ khat
                    print(
                        "🖐️  Handle axes captured (handles assumed pointing "
                        f"straight down): L={np.round(handle_axes['left'], 2)} "
                        f"R={np.round(handle_axes['right'], 2)}"
                    )
                mode = "mirror" if mirror_control else "direct"
                print(f"✓ Dual-arm teleop activated (absolute handle mapping, {mode})")
                # Mapping diagnostics: the world direction each handle axis
                # maps to right now (at first grip this is exactly [0,0,-1]
                # by construction; on re-grips it shows the true reading).
                for _side, _key, _tf, _pose in (
                    ("L", "left", left_tf, left_pose),
                    ("R", "right", right_tf, right_pose),
                ):
                    _hd = _tf[:3, :3] @ handle_axes[_key]
                    print(
                        f"  🧭 {_side}: handle world dir={np.round(_hd, 2)} "
                        f"(z<0=down) | current tip={np.round(_pose[:3, 0], 2)}"
                    )

            if quest_reader is not None and not teleop_active and teleop_active_prev:
                _reset_teleop_calibration()
                print("✗ Dual-arm teleop deactivated")

            teleop_active_prev = teleop_active

            if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    _sync_targets_from_ik(data_manager, ik_solver, current_joint_angles)

            elif (
                quest_reader is not None
                and teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
                and left_tf is not None
                and right_tf is not None
                and left_hand_to_robot is not None
                and right_hand_to_robot is not None
                and left_hand_reference is not None
                and right_hand_reference is not None
                and left_rot_at_activation is not None
                and right_rot_at_activation is not None
                and activation_time is not None
            ):
                translation_scale, rotation_scale = data_manager.get_teleop_scaling()

                # Orientation: absolute and reachable-by-construction — tip
                # elevation from the handle, tip azimuth following the arm,
                # roll from the hand twist. Blended in over
                # ORIENTATION_BLEND_TIME_S after activation to avoid a jerk.
                left_abs_rot = hand_to_gripper_orientation_armplane(
                    left_tf[:3, :3],
                    _update_azimuth("left", left_pose),
                    0.0,
                    handle_axes["left"],
                    knuckle_axes.get("left"),
                )
                right_abs_rot = hand_to_gripper_orientation_armplane(
                    right_tf[:3, :3],
                    _update_azimuth("right", right_pose),
                    0.0,
                    handle_axes["right"],
                    knuckle_axes.get("right"),
                )
                blend_alpha = min(
                    1.0, (time.time() - activation_time) / ORIENTATION_BLEND_TIME_S
                )
                left_target_rot = blend_rotations(
                    left_rot_at_activation, left_abs_rot, blend_alpha
                )
                right_target_rot = blend_rotations(
                    right_rot_at_activation, right_abs_rot, blend_alpha
                )

                # Position: world-frame delta from the calibration anchor.
                left_target = np.eye(4)
                left_target[:3, 3] = (
                    left_hand_to_robot[:3, 3]
                    + (left_tf[:3, 3] - left_hand_reference[:3, 3]) * translation_scale
                )
                left_target[:3, :3] = left_target_rot
                right_target = np.eye(4)
                right_target[:3, 3] = (
                    right_hand_to_robot[:3, 3]
                    + (right_tf[:3, 3] - right_hand_reference[:3, 3])
                    * translation_scale
                )
                right_target[:3, :3] = right_target_rot

                # Height lock: freeze target z at the height each arm had
                # when the lock was engaged — flat table strokes for free.
                height_lock = data_manager.get_height_lock_enabled()
                if height_lock and not height_lock_prev:
                    locked_z["left"] = float(left_target[2, 3])
                    locked_z["right"] = float(right_target[2, 3])
                    print(
                        f"📏 Height locked (L z={locked_z['left']:.3f} m, "
                        f"R z={locked_z['right']:.3f} m)"
                    )
                elif not height_lock and height_lock_prev:
                    print("📏 Height unlocked")
                height_lock_prev = height_lock
                if height_lock:
                    left_target[2, 3] = locked_z["left"]
                    right_target[2, 3] = locked_z["right"]

                ik_solver.set_target_poses(
                    {
                        LEFT_END_EFFECTOR_FRAME_NAME: (
                            left_target[:3, 3],
                            left_target[:3, :3],
                        ),
                        RIGHT_END_EFFECTOR_FRAME_NAME: (
                            right_target[:3, 3],
                            right_target[:3, :3],
                        ),
                    }
                )

                success = ik_solver.solve_ik()
                if success:
                    joint_config = np.degrees(ik_solver.get_current_configuration())
                    data_manager.set_target_joint_angles(joint_config)
                    data_manager.set_target_pose("left", left_target)
                    data_manager.set_target_pose("right", right_target)
                    data_manager.set_ik_success(True)
                    data_manager.set_ik_solve_time_ms(
                        float(ik_solver.get_statistics()["last_solve_time_ms"])
                    )
                else:
                    data_manager.set_ik_success(False)
                    data_manager.set_ik_solve_time_ms(0.0)

                # Periodic mapping diagnostics (2 s): target tip direction vs
                # the tip the solver currently achieves. Divergent target =
                # mapping/frame problem; matching-but-wrong-on-robot =
                # hardware-side problem.
                if time.time() - last_debug_time > 2.0:
                    last_debug_time = time.time()
                    cur = ik_solver.get_current_end_effector_poses()
                    lc = cur[LEFT_END_EFFECTOR_FRAME_NAME][:3, 0]
                    rc = cur[RIGHT_END_EFFECTOR_FRAME_NAME][:3, 0]
                    print(
                        f"🧭 tip target L={np.round(left_target[:3, 0], 2)} "
                        f"ik L={np.round(lc, 2)} | "
                        f"target R={np.round(right_target[:3, 0], 2)} "
                        f"ik R={np.round(rc, 2)} (z<0=down)"
                    )

            elif (
                quest_reader is None
                and teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
            ):
                left_target = data_manager.get_gizmo_target_pose("left")
                right_target = data_manager.get_gizmo_target_pose("right")
                if left_target is not None and right_target is not None:
                    ik_solver.set_target_poses(
                        {
                            LEFT_END_EFFECTOR_FRAME_NAME: (
                                left_target[:3, 3],
                                left_target[:3, :3],
                            ),
                            RIGHT_END_EFFECTOR_FRAME_NAME: (
                                right_target[:3, 3],
                                right_target[:3, :3],
                            ),
                        }
                    )
                    success = ik_solver.solve_ik()
                    if success:
                        joint_config = np.degrees(ik_solver.get_current_configuration())
                        data_manager.set_target_joint_angles(joint_config)
                        data_manager.set_target_pose("left", left_target)
                        data_manager.set_target_pose("right", right_target)
                        data_manager.set_ik_success(True)
                        data_manager.set_ik_solve_time_ms(
                            float(ik_solver.get_statistics()["last_solve_time_ms"])
                        )
                    else:
                        data_manager.set_ik_success(False)
                        data_manager.set_ik_solve_time_ms(0.0)

            elif robot_activity_state in (
                RobotActivityState.HOMING,
                RobotActivityState.DISABLED,
            ):
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    _sync_targets_from_ik(data_manager, ik_solver, current_joint_angles)

            elif not teleop_active:
                joint_config = np.degrees(ik_solver.get_current_configuration())
                _sync_targets_from_ik(data_manager, ik_solver, joint_config)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as exc:
        print(f"❌ Dual IK solver thread error: {exc}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🧮 Dual IK solver thread stopped")
