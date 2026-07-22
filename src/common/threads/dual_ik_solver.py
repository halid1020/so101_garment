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
    JOYSTICK_DEADZONE,
    JOYSTICK_EXPO,
    JOYSTICK_FLEX_RATE_DEG_S,
    JOYSTICK_FLEX_SIGN,
    JOYSTICK_ROLL_RATE_DEG_S,
    JOYSTICK_ROLL_SIGN,
    LEFT_END_EFFECTOR_FRAME_NAME,
    ORIENTATION_BLEND_TIME_S,
    RATCHET_LIMIT_GUARD_DEG,
    RIGHT_END_EFFECTOR_FRAME_NAME,
    WORKSPACE_OOB_MODE,
    WORKSPACE_Z_FLOOR,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.envelope_feedback import EnvelopeFeedback
from common.joystick_wrist import JoystickWristTrim
from common.pink_ik_solver import PinkIKSolver
from common.roll_ratchet import KEEP, NEUTRAL, REWRAP, RollRatchet
from common.utils import (
    blend_rotations,
    compute_hand_to_robot_calibration,
    compute_operator_frame,
    gripper_orientation_from_pitch_roll,
    gripper_pitch_roll_from_rotation,
    hand_to_gripper_orientation_armplane,
    map_quest_hands_to_robot_arms,
    to_operator_frame,
    wrist_roll_pitch_delta,
)
from common.workspace_envelope import build_envelopes, make_policies

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
    oob_mode: str | None = None,
    envelope_feedback: EnvelopeFeedback | None = None,
    orientation_mode: str = "armplane",
    joystick_wrist: bool = False,
    envelope_z_floor: float | None = None,
) -> None:
    """Dual-arm IK solver thread.

    With quest_reader: absolute pointer poses in the head frame map to robot
    TCP targets each frame. A one-time calibration at grip-press aligns head
    frame to the robot. Left hand → left arm, right hand → right arm.
    Without quest_reader: Viser gizmos drive absolute TCP targets via DataManager.

    oob_mode selects the out-of-envelope policy applied to every target
    position before it reaches the IK ("warn"|"project"|"freeze"|"slow",
    see common/workspace_envelope.py); None uses WORKSPACE_OOB_MODE.

    envelope_feedback, if given, receives the per-arm OOEStatus every tick
    (debounced operator cues, e.g. a terminal bell — see
    common/envelope_feedback.py); its state resets with the calibration.

    orientation_mode selects how the operator's wrist maps to the gripper
    attitude: "armplane" (default) maps the ABSOLUTE hand orientation to a
    reachable gripper orientation (yaw follows the arm, elevation + roll from
    the handle); "incremental" instead anchors pitch and roll to the arm's
    current orientation at each grip and tracks only the CHANGE of the
    operator's wrist pitch/roll from there — a clutched, ratchetable mapping
    that lets teleop start with the controllers in any pose; "hold" freezes
    the gripper attitude at the rotation captured on grip and changes it only
    through the thumbstick wrist trims (so the handle drives position only).

    joystick_wrist enables the thumbstick wrist trims (the "mymethod" clutch,
    see common/joystick_wrist.py): while a controller stick is deflected, that
    arm's other joints freeze and the handle is ignored while stick x trims
    wrist_roll and stick y trims wrist_flex in joint space; releasing the
    stick leaves the wrist where it was and re-anchors the handle so control
    resumes with no jump. Requires quest_reader to expose get_joystick_value.

    envelope_z_floor overrides the envelope's table-clearance floor (see
    common/workspace_envelope.py build_envelopes); None keeps the shared-YAML
    value. Only sim collection scenes with a lowered table pass this.
    """
    input_label = "Quest" if quest_reader is not None else "Viser gizmo"
    print(f"🧮 Dual IK solver thread started ({input_label}, {orientation_mode})")

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
    # Wrist-roll ratcheting (see common/roll_ratchet.py): joint indices in
    # the 10-DoF configuration ([left x5, right x5], roll is joint 5 of each
    # arm), the roll joint limits from the model, and per-side decision state.
    _roll_idx = {"left": 4, "right": 9}
    # wrist_flex joint index per side in the 10-DoF config (joint 4 of each
    # arm); used by the thumbstick wrist trims alongside _roll_idx.
    _flex_idx = {"left": 3, "right": 8}
    # Side joint slices in the 10-DoF config: left 0:5, right 5:10.
    _side_slice = {"left": slice(0, 5), "right": slice(5, 10)}
    _ee_fid = {
        "left": _m.getFrameId(LEFT_END_EFFECTOR_FRAME_NAME),
        "right": _m.getFrameId(RIGHT_END_EFFECTOR_FRAME_NAME),
    }
    roll_ratchet = RollRatchet(
        lo=float(_m.lowerPositionLimit[_roll_idx["left"]]),
        hi=float(_m.upperPositionLimit[_roll_idx["left"]]),
        guard_rad=np.deg2rad(RATCHET_LIMIT_GUARD_DEG),
    )

    def _knuckle_at_neutral_roll(side: str) -> np.ndarray:
        """World knuckle axis of this arm's EE with its wrist_roll zeroed.

        Used by the operator-requested roll reset: anchoring the roll
        reference here makes the commanded roll glide back to the joint's
        neutral over the activation blend.
        """
        q = ik_solver.get_current_configuration().copy()
        q[_roll_idx[side]] = 0.0
        pin.forwardKinematics(_m, _d, q)
        pin.updateFramePlacements(_m, _d)
        rot = _d.oMf[_ee_fid[side]].rotation
        khat = rot[:, 2].copy()
        khat[2] = 0.0
        n = np.linalg.norm(khat)
        return khat / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    # Arm azimuths (compass direction of the EE from its base), kept between
    # cycles so a near-vertical/retracted pose doesn't produce noise.
    azimuths: dict[str, float] = {}
    # Whether each arm's target was inside the envelope last tick; used to
    # freeze the azimuth (and hence the roll reference) while out of envelope.
    oob_inside_prev: dict[str, bool] = {"left": True, "right": True}
    # Incremental-mapping state (orientation_mode="incremental"): at each grip
    # the *_anchor holds the gripper's actual pitch/roll and hand_rot_ref the
    # operator's hand rotation (operator frame); the commanded attitude is
    # anchor + wrist-delta-since-grip, so re-gripping continues from the
    # current pose (clutched ratchet) and the start pose is irrelevant.
    pitch_anchor: dict[str, float] = {"left": 0.0, "right": 0.0}
    roll_anchor: dict[str, float] = {"left": 0.0, "right": 0.0}
    hand_rot_ref: dict[str, np.ndarray] = {}
    # Hold-mode state (orientation_mode="hold"): the gripper rotation captured
    # at each grip, held as the absolute attitude target and changed only by
    # the thumbstick wrist trims.
    hold_rot: dict[str, np.ndarray] = {}
    # Thumbstick wrist trims (joystick_wrist): the per-side integrator plus the
    # latched 5-joint slice held while that arm's stick is deflected (its other
    # joints are frozen exactly; see common/joystick_wrist.py).
    joy_trim = (
        JoystickWristTrim(
            deadzone=JOYSTICK_DEADZONE,
            expo=JOYSTICK_EXPO,
            roll_rate_rad_s=np.deg2rad(JOYSTICK_ROLL_RATE_DEG_S),
            flex_rate_rad_s=np.deg2rad(JOYSTICK_FLEX_RATE_DEG_S),
            roll_sign=JOYSTICK_ROLL_SIGN,
            flex_sign=JOYSTICK_FLEX_SIGN,
        )
        if joystick_wrist
        else None
    )
    wrist_latch: dict[str, np.ndarray] = {}
    # Sign of the wrist->gripper mapping per axis. Wrist twist left -> gripper
    # rolls counter-clockwise; wrist up -> gripper pitches up. Flip a sign here
    # if a direction comes out reversed on the real robot.
    _ROLL_SIGN, _PITCH_SIGN = 1.0, -1.0
    last_debug_time = 0.0

    # Workspace envelope + out-of-envelope policy (per arm). Applied to every
    # target position right before it is handed to the IK.
    oob_mode = oob_mode if oob_mode is not None else WORKSPACE_OOB_MODE
    oob_policies = make_policies(
        oob_mode, build_envelopes(_m, z_floor=envelope_z_floor)
    )
    print(f"🛡️  Out-of-envelope policy: {oob_mode}")
    if envelope_z_floor is not None:
        print(f"🛡️  Envelope z-floor override: {envelope_z_floor:+.4f} m")

    # Handle axes are captured PER HAND at EVERY grip: the operator holds
    # both handles pointing straight down, and whatever body-frame direction
    # is "world down" at that instant is that hand's handle axis. (The two
    # Quest controllers are mirrored hardware, and their body frames are not
    # stable across app sessions, so a config constant cannot express this.)
    # Re-capturing on every re-grip means a bad calibration is fixed by
    # simply releasing and gripping again.
    handle_axes: dict[str, np.ndarray] = {}
    knuckle_axes: dict[str, np.ndarray] = {}
    # Operator control frame (headset-anywhere): derived from the two handle
    # poses at every grip — origin behind/above the handle midpoint, x
    # forward, y toward the operator's left, z up. All control happens in
    # this frame, so the headset's own placement/yaw never matters.
    # Identity until the first grip.
    op_frame_rot = np.eye(3)
    op_frame_origin = np.zeros(3)
    # Height lock: when toggled on, target z freezes at each arm's current
    # height — hand z is ignored, so table-plane strokes stay perfectly flat.
    height_lock_prev = False
    locked_z: dict[str, float] = {}
    _WORLD_DOWN = np.array([0.0, 0.0, -1.0])
    if quest_reader is not None:
        print(
            "🖐️  EVERY GRIP CALIBRATES: whenever you press both grips, "
            "point both HANDLES straight down (like two nails) — "
            "handle-down = gripper-down until the next re-grip. The "
            "headset can sit anywhere: the control frame comes from the "
            "handles themselves."
        )

    def _update_azimuth(
        side: str, ee_pose: np.ndarray | None, freeze: bool = False
    ) -> float:
        # ``freeze`` holds the last azimuth without updating it. The armplane
        # tip azimuth follows the arm's live EE heading, and the roll
        # reference is orthogonalised against that tip — so any azimuth change
        # rotates the commanded wrist_roll. While the target is out of the
        # envelope the arm saturates and its heading swings as it slides along
        # the boundary, which would inject roll the operator never asked for.
        # Freezing the azimuth there keeps the gripper orientation steady, so
        # roll only responds to a genuine wrist twist.
        if ee_pose is not None and not freeze:
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
        if envelope_feedback is not None:
            envelope_feedback.reset()
        roll_ratchet.reset()
        if joy_trim is not None:
            joy_trim.reset()
        wrist_latch.clear()
        hold_rot.clear()
        oob_inside_prev["left"] = True
        oob_inside_prev["right"] = True
        left_rot_at_activation = None
        right_rot_at_activation = None
        activation_time = None
        left_hand_to_robot = None
        right_hand_to_robot = None
        left_hand_reference = None
        right_hand_reference = None
        for policy in oob_policies.values():
            policy.reset()

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
                # On every grip, re-derive the operator control frame from
                # the handle poses themselves (headset-anywhere).
                if (
                    teleop_active
                    and not teleop_active_prev
                    and left_tf is not None
                    and right_tf is not None
                ):
                    op_frame_rot, op_frame_origin = compute_operator_frame(
                        left_tf, right_tf
                    )
                    fwd = np.round(op_frame_rot[:, 0], 2)
                    print(
                        "🧿 Operator frame captured: origin="
                        f"{np.round(op_frame_origin, 2)} forward={fwd} "
                        "(reader coords)"
                    )
                # All downstream control math sees operator-frame poses.
                if left_tf is not None:
                    left_tf = to_operator_frame(left_tf, op_frame_rot, op_frame_origin)
                if right_tf is not None:
                    right_tf = to_operator_frame(
                        right_tf, op_frame_rot, op_frame_origin
                    )
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
                # EVERY grip recalibrates: capture each hand's handle axis
                # (operator is holding the handles pointing straight down)
                # and its roll reference ("knuckle" axis) — a horizontal
                # world direction taken from the gripper's current roll, so
                # it is perpendicular to the handle by construction and the
                # roll response is strong in every session. Re-gripping is
                # therefore always a full, fresh calibration.
                _q_solved = ik_solver.get_current_configuration()
                for _key, _tf, _pose in (
                    ("left", left_tf, left_pose),
                    ("right", right_tf, right_pose),
                ):
                    handle_axes[_key] = _tf[:3, :3].T @ _WORLD_DOWN
                    khat = _pose[:3, 2].copy()
                    khat[2] = 0.0
                    n = np.linalg.norm(khat)
                    khat = khat / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
                    # Roll ratcheting (armplane only): decide the roll anchor
                    # for this engagement (see common/roll_ratchet.py). The
                    # activation blend below glides any change, so nothing
                    # snaps. The incremental mapping does its own clutched
                    # re-anchoring, so it skips the ratchet.
                    if orientation_mode == "armplane":
                        try:
                            _trig = float(quest_reader.get_trigger_value(_key))
                        except (AttributeError, TypeError):
                            _trig = 0.0
                        _action = roll_ratchet.decide_at_grip(
                            _key,
                            float(_q_solved[_roll_idx[_key]]),
                            data_manager.consume_roll_reset(_key),
                            _trig > 0.5,
                        )
                        if _action == NEUTRAL:
                            khat = _knuckle_at_neutral_roll(_key)
                            print(f"🔄 {_key} gripper roll gliding back to neutral")
                        elif _action == REWRAP:
                            # Jaw-equivalent half-turn: negating the horizontal
                            # roll reference rotates the target roll by 180 deg,
                            # which the parallel jaws cannot distinguish, and
                            # restores wrist_roll headroom.
                            khat = -khat
                            print(
                                f"↩️  {_key} wrist rolled back 180° (jaw-"
                                "equivalent) — roll headroom restored"
                            )
                        else:
                            assert _action == KEEP
                    knuckle_axes[_key] = _tf[:3, :3].T @ khat
                    if orientation_mode == "incremental":
                        # Clutched anchor: the gripper's actual pitch/roll now,
                        # and the operator's hand rotation as the zero. The
                        # commanded attitude then tracks the wrist change from
                        # here, so a re-grip continues from the current pose.
                        _, _ap, _ar = gripper_pitch_roll_from_rotation(_pose[:3, :3])
                        pitch_anchor[_key], roll_anchor[_key] = _ap, _ar
                        hand_rot_ref[_key] = _tf[:3, :3].copy()
                    elif orientation_mode == "hold":
                        # Freeze the gripper attitude at its current rotation;
                        # only the thumbstick trims change it from here.
                        hold_rot[_key] = _pose[:3, :3].copy()
                print(
                    "🖐️  Handle axes captured (handles assumed pointing "
                    f"straight down): L={np.round(handle_axes['left'], 2)} "
                    f"R={np.round(handle_axes['right'], 2)}"
                )
                # Publish the notional headset center in ROBOT coordinates
                # for visualization: the operator-frame origin (0 in op
                # coords) mapped through each arm's clutch correspondence.
                headset_center = 0.5 * (
                    (left_hand_to_robot[:3, 3] - left_hand_reference[:3, 3])
                    + (right_hand_to_robot[:3, 3] - right_hand_reference[:3, 3])
                )
                data_manager.set_frame_marker("headset_center", headset_center)
                mode = "mirror" if mirror_control else "direct"
                _map = (
                    "incremental" if orientation_mode == "incremental" else "absolute"
                )
                print(f"✓ Dual-arm teleop activated ({_map} handle mapping, {mode})")
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

                # Orientation target, blended in over ORIENTATION_BLEND_TIME_S
                # after activation to avoid a jerk. Tip azimuth always follows
                # the arm (frozen while out of envelope). "armplane" maps the
                # absolute hand orientation; "incremental" tracks the change of
                # the operator's wrist pitch/roll from the grip anchor.
                az_left = _update_azimuth(
                    "left", left_pose, freeze=not oob_inside_prev["left"]
                )
                az_right = _update_azimuth(
                    "right", right_pose, freeze=not oob_inside_prev["right"]
                )
                if orientation_mode == "incremental":
                    # Wrist rotation since the grip, in the operator frame ->
                    # roll (twist) and pitch (nod) deltas added to the anchor.
                    l_roll, l_pitch = wrist_roll_pitch_delta(
                        hand_rot_ref["left"].T @ left_tf[:3, :3]
                    )
                    r_roll, r_pitch = wrist_roll_pitch_delta(
                        hand_rot_ref["right"].T @ right_tf[:3, :3]
                    )
                    left_abs_rot = gripper_orientation_from_pitch_roll(
                        az_left,
                        pitch_anchor["left"] + rotation_scale * _PITCH_SIGN * l_pitch,
                        roll_anchor["left"] + rotation_scale * _ROLL_SIGN * l_roll,
                    )
                    right_abs_rot = gripper_orientation_from_pitch_roll(
                        az_right,
                        pitch_anchor["right"] + rotation_scale * _PITCH_SIGN * r_pitch,
                        roll_anchor["right"] + rotation_scale * _ROLL_SIGN * r_roll,
                    )
                elif orientation_mode == "hold":
                    # Attitude held from the grip; the thumbstick trims are the
                    # only thing that changes it (they move the wrist joints,
                    # and the falling edge writes the new attitude back here).
                    left_abs_rot = hold_rot["left"]
                    right_abs_rot = hold_rot["right"]
                else:
                    left_abs_rot = hand_to_gripper_orientation_armplane(
                        left_tf[:3, :3],
                        az_left,
                        0.0,
                        handle_axes["left"],
                        knuckle_axes.get("left"),
                    )
                    right_abs_rot = hand_to_gripper_orientation_armplane(
                        right_tf[:3, :3],
                        az_right,
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

                # Out-of-envelope policy — runs AFTER height lock so a locked
                # z below the floor still gets clamped (safety wins).
                now = time.time()
                left_target[:3, 3], left_oob = oob_policies["left"].apply(
                    left_target[:3, 3], now
                )
                right_target[:3, 3], right_oob = oob_policies["right"].apply(
                    right_target[:3, 3], now
                )
                if envelope_feedback is not None:
                    envelope_feedback.notify("left", left_oob, now)
                    envelope_feedback.notify("right", right_oob, now)
                # Remember envelope state so next tick can freeze the azimuth
                # (and roll reference) while a target stays out of envelope.
                oob_inside_prev["left"] = left_oob.inside
                oob_inside_prev["right"] = right_oob.inside

                # Wrist-roll limit hint (armplane only): a hold that parks the
                # roll in the guard band gets a throttled release-and-regrip
                # reminder (the rewrap itself only ever happens at a grip edge).
                _q_now = ik_solver.get_current_configuration()
                for _side in (
                    ("left", "right") if orientation_mode == "armplane" else ()
                ):
                    if roll_ratchet.should_warn_mid_hold(
                        _side, float(_q_now[_roll_idx[_side]]), now
                    ):
                        print(
                            f"⚠️  {_side} wrist roll near its limit — "
                            "release, untwist your wrist, and re-grip to "
                            "keep rolling"
                        )

                # Thumbstick wrist trims (mymethod): a deflected stick clutches
                # that arm — its other joints freeze exactly and the handle is
                # ignored — while stick x trims wrist_roll and stick y trims
                # wrist_flex in joint space. On release the wrist stays put and
                # the handle re-anchors so control resumes with no jump. See
                # common/joystick_wrist.py.
                frozen_sides: list[str] = []
                if joy_trim is not None:
                    _targets = {"left": left_target, "right": right_target}
                    _tfs = {"left": left_tf, "right": right_tf}
                    _get_js = getattr(quest_reader, "get_joystick_value", None)
                    _lo, _hi = _m.lowerPositionLimit, _m.upperPositionLimit
                    for _side in ("left", "right"):
                        if _get_js is not None:
                            # Under mirror control the sticks swap arms and the
                            # roll axis reflects, so read the other hand and
                            # negate x. SIGN UNTESTED on hardware — flip here (or
                            # in the YAML roll_sign) if roll comes out reversed.
                            _hand = _side
                            if mirror_control:
                                _hand = "right" if _side == "left" else "left"
                            _jx, _jy = _get_js(_hand)
                            _jx, _jy = float(_jx), float(_jy)
                            if mirror_control:
                                _jx = -_jx
                        else:
                            _jx, _jy = 0.0, 0.0
                        _res = joy_trim.update(_side, _jx, _jy, dt)
                        if not (_res.engaged or _res.just_released):
                            continue
                        if _res.just_engaged:
                            wrist_latch[_side] = ik_solver.get_current_configuration()[
                                _side_slice[_side]
                            ].copy()
                            print(f"🕹️  {_side} wrist trim engaged (arm frozen)")
                        _latch = wrist_latch[_side]
                        if _res.engaged:
                            _new_flex = float(
                                np.clip(
                                    _latch[3] + _res.d_flex,
                                    _lo[_flex_idx[_side]],
                                    _hi[_flex_idx[_side]],
                                )
                            )
                            _new_roll = float(
                                np.clip(
                                    _latch[4] + _res.d_roll,
                                    _lo[_roll_idx[_side]],
                                    _hi[_roll_idx[_side]],
                                )
                            )
                            # Floor guard: FK the candidate; if a flex change
                            # drives the tip below the table, revert it (roll
                            # cannot lower the tip, so it is always kept).
                            _q_probe = ik_solver.get_current_configuration().copy()
                            _q_probe[_side_slice[_side]] = _latch
                            _q_probe[_flex_idx[_side]] = _new_flex
                            _q_probe[_roll_idx[_side]] = _new_roll
                            pin.forwardKinematics(_m, _d, _q_probe)
                            pin.updateFramePlacements(_m, _d)
                            if (
                                _d.oMf[_ee_fid[_side]].translation[2]
                                < WORKSPACE_Z_FLOOR
                            ):
                                _new_flex = _latch[3]
                            _latch[3] = _new_flex
                            _latch[4] = _new_roll
                        # Build the frozen config and set this arm's target to
                        # its FK pose, so the solve is self-consistent and the
                        # handle is ignored for this arm.
                        _q_full = ik_solver.get_current_configuration().copy()
                        _q_full[_side_slice[_side]] = _latch
                        pin.forwardKinematics(_m, _d, _q_full)
                        pin.updateFramePlacements(_m, _d)
                        _fk = _d.oMf[_ee_fid[_side]]
                        _fk_pose = np.eye(4)
                        _fk_pose[:3, :3] = _fk.rotation.copy()
                        _fk_pose[:3, 3] = _fk.translation.copy()
                        _targets[_side] = _fk_pose
                        ik_solver.set_configuration_no_task_update(_q_full)
                        frozen_sides.append(_side)
                        if _res.just_released:
                            # Re-anchor so nothing jumps: zero the position
                            # delta against the frozen pose, and carry the
                            # trimmed attitude into the mode's orientation ref.
                            _tf = _tfs[_side]
                            _cal = compute_hand_to_robot_calibration(
                                _fk_pose, _tf, _tf, translation_scale, rotation_scale
                            )
                            if _side == "left":
                                left_hand_reference = _tf.copy()
                                left_hand_to_robot = _cal
                            else:
                                right_hand_reference = _tf.copy()
                                right_hand_to_robot = _cal
                            _fk_rot = _fk_pose[:3, :3]
                            if orientation_mode == "hold":
                                hold_rot[_side] = _fk_rot.copy()
                            elif orientation_mode == "incremental":
                                _, _ap, _ar = gripper_pitch_roll_from_rotation(_fk_rot)
                                pitch_anchor[_side], roll_anchor[_side] = _ap, _ar
                                hand_rot_ref[_side] = _tf[:3, :3].copy()
                            print(f"🕹️  {_side} wrist trim released (handle resumed)")
                    left_target = _targets["left"]
                    right_target = _targets["right"]

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
                    if frozen_sides:
                        # Exact freeze: re-impose the latched joints on the
                        # solved config before publishing, so a clutched arm's
                        # non-wrist joints do not drift by an IK residual.
                        _q_solved = ik_solver.get_current_configuration().copy()
                        for _side in frozen_sides:
                            _q_solved[_side_slice[_side]] = wrist_latch[_side]
                        ik_solver.set_configuration_no_task_update(_q_solved)
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
                        f"ik R={np.round(rc, 2)} (z<0=down) | "
                        f"envelope margin L={left_oob.margin_m * 1000:+.0f}mm "
                        f"R={right_oob.margin_m * 1000:+.0f}mm"
                    )

            elif (
                quest_reader is None
                and teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
            ):
                left_target = data_manager.get_gizmo_target_pose("left")
                right_target = data_manager.get_gizmo_target_pose("right")
                if left_target is not None and right_target is not None:
                    now = time.time()
                    left_target = left_target.copy()
                    right_target = right_target.copy()
                    left_target[:3, 3], left_oob = oob_policies["left"].apply(
                        left_target[:3, 3], now
                    )
                    right_target[:3, 3], right_oob = oob_policies["right"].apply(
                        right_target[:3, 3], now
                    )
                    if envelope_feedback is not None:
                        envelope_feedback.notify("left", left_oob, now)
                        envelope_feedback.notify("right", right_oob, now)
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
