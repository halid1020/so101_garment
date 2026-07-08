"""Utility functions for the examples."""

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def transform_from_position_wxyz(
    position: Sequence[float], wxyz: Sequence[float]
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from Viser position and wxyz quaternion."""
    quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position, dtype=float)
    transform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return transform


def position_wxyz_from_transform(
    transform: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a 4x4 transform to Viser (position, wxyz quaternion)."""
    position = tuple(float(v) for v in transform[:3, 3])
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    wxyz = (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )
    return position, wxyz


def se3_inverse(transform: np.ndarray) -> np.ndarray:
    """Invert a proper 4x4 rigid-body transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


_MIRROR_HEAD_FRAME_Y = np.diag([1.0, -1.0, 1.0, 1.0])


def mirror_head_frame_pose(transform: np.ndarray) -> np.ndarray:
    """Mirror a head-frame hand pose for face-to-face (mirror) teleoperation."""
    return _MIRROR_HEAD_FRAME_Y @ transform @ _MIRROR_HEAD_FRAME_Y


# Offset of the operator-frame origin from the handle midpoint at grip:
# 20 cm behind (toward the operator) and 20 cm above — roughly where the
# operator's head sits, making the origin a natural "headset center" proxy.
OPERATOR_FRAME_BACK_M = 0.20
OPERATOR_FRAME_UP_M = 0.20


def compute_operator_frame(
    left_hand_tf: np.ndarray, right_hand_tf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Operator control frame from the two handle poses at grip time.

    The Quest app streams controller poses in a reference frame whose yaw
    and origin depend on where the HEADSET was when the app started — so
    with raw poses, the operator has to place the headset carefully or
    "forward" on the hand stops being "forward" on the robot. This frame
    removes that dependence: at every grip, a robot-aligned frame is built
    from the handles themselves (gravity gives z; the left→right handle
    line gives the lateral axis), and all control happens in it. The
    headset can sit anywhere.

    Returns (rotation, origin) in the reader frame: `rotation` columns are
    the operator frame's x (forward, away from the operator), y (operator's
    left), z (up); `origin` sits OPERATOR_FRAME_BACK_M behind and
    OPERATOR_FRAME_UP_M above the handle midpoint — a headset-center proxy.
    Assumes the reader frame's z is gravity-aligned (OpenXR spaces are).
    """
    p_left = left_hand_tf[:3, 3]
    p_right = right_hand_tf[:3, 3]
    midpoint = 0.5 * (p_left + p_right)
    y_axis = p_left - p_right
    y_axis[2] = 0.0  # lateral axis is horizontal by construction
    norm = np.linalg.norm(y_axis)
    y_axis = y_axis / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)  # forward, right-handed with y left
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    origin = midpoint - OPERATOR_FRAME_BACK_M * x_axis + OPERATOR_FRAME_UP_M * z_axis
    return rotation, origin


def to_operator_frame(
    hand_tf: np.ndarray, frame_rot: np.ndarray, frame_origin: np.ndarray
) -> np.ndarray:
    """Re-express a reader-frame hand pose in the operator control frame."""
    out = np.eye(4)
    out[:3, :3] = frame_rot.T @ hand_tf[:3, :3]
    out[:3, 3] = frame_rot.T @ (hand_tf[:3, 3] - frame_origin)
    return out


def map_quest_hands_to_robot_arms(
    left_hand_transform: np.ndarray,
    right_hand_transform: np.ndarray,
    *,
    mirror_control: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Quest left/right hands to robot left/right arms.

    Normal mode: Quest left → robot left, Quest right → robot right.
    Mirror mode: swap hands and reflect lateral motion.
    """
    if not mirror_control:
        return left_hand_transform, right_hand_transform
    return (
        mirror_head_frame_pose(right_hand_transform),
        mirror_head_frame_pose(left_hand_transform),
    )


def compute_hand_to_robot_calibration(
    robot_ee_pose: np.ndarray,
    hand_in_chest_frame: np.ndarray,
    hand_reference_at_calibration: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """Return the robot EE pose at calibration time, used as the delta anchor."""
    return robot_ee_pose.copy()


def map_head_frame_hand_to_robot_target(
    hand_in_head_frame: np.ndarray,
    hand_to_robot_calibration: np.ndarray,
    hand_reference_at_calibration: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """Map an absolute head-frame hand pose to a robot-base TCP target.

    Translation: world-frame delta from the calibration reference, scaled and
    added to the robot EE position at calibration — no rotation of the delta.

    Rotation: phosphobot-style scalar Euler mapping using only the two DOFs a
    human wrist shares with the SO-101 wrist — pitch and roll. Hand pitch and
    roll deltas (yaw-pitch-roll / intrinsic 'zyx' decomposition) are added to
    the EE's calibration pitch and roll; the hand's yaw is discarded and the
    target yaw is held at its calibration value. The SO-101 has no wrist-yaw
    joint — yaw comes only from panning the whole arm — so the IK task's yaw
    axis is also zero-costed (EE_ORIENTATION_COST_MASK) and the gripper's yaw
    simply follows the arm.
    """
    # Translation — world-frame delta, added directly (matches single-arm ik_solver.py)
    delta_pos = hand_in_head_frame[:3, 3] - hand_reference_at_calibration[:3, 3]
    target_pos = hand_to_robot_calibration[:3, 3] + delta_pos * translation_scale

    # Rotation — scalar pitch/roll deltas, yaw dropped
    hand_ypr = Rotation.from_matrix(hand_in_head_frame[:3, :3]).as_euler("zyx")
    ref_ypr = Rotation.from_matrix(hand_reference_at_calibration[:3, :3]).as_euler(
        "zyx"
    )
    calib_ypr = Rotation.from_matrix(hand_to_robot_calibration[:3, :3]).as_euler("zyx")

    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    delta_pitch = _wrap(hand_ypr[1] - ref_ypr[1]) * rotation_scale
    delta_roll = _wrap(hand_ypr[2] - ref_ypr[2]) * rotation_scale
    target_rot = Rotation.from_euler(
        "zyx",
        [calib_ypr[0], calib_ypr[1] + delta_pitch, calib_ypr[2] + delta_roll],
    ).as_matrix()

    target = np.eye(4)
    target[:3, 3] = target_pos
    target[:3, :3] = target_rot
    return target


def _handle_to_gripper_offset(
    handle_pitch_offset_deg: float,
    handle_axis: Sequence[float] | None = None,
) -> np.ndarray:
    """Fixed body-frame rotation mapping the Quest controller frame to the
    gripper (EE) frame.

    The reader's "ros" transform converts only the WORLD basis; the
    controller's local columns remain OpenXR aim-pose axes:
    x = right, y = up, z = backward (pointer = -z).

    ``handle_axis``, when given (measured with tool/calibrate_handle.py),
    is the handle's top->bottom direction in that body frame and overrides
    the analytic guess of -y tilted backward by ``handle_pitch_offset_deg``.
    We build an orthonormal triad and re-label:
      EE x (wrist->tip)  <- handle axis
      EE z (free/yaw)    <- pointer direction, orthogonalized to the handle
      EE y (pitch axis)  <- completes the right-handed frame
    """
    if handle_axis is not None:
        e1 = np.asarray(handle_axis, dtype=float)
        e1 = e1 / np.linalg.norm(e1)
    else:
        theta = np.radians(handle_pitch_offset_deg)
        e1 = np.array([0.0, -np.cos(theta), np.sin(theta)])
    pointer = np.array([0.0, 0.0, -1.0])
    e3 = pointer - (pointer @ e1) * e1
    norm = np.linalg.norm(e3)
    if norm < 1e-6:
        raise ValueError("handle axis is parallel to the pointer axis")
    e3 = e3 / norm
    e2 = np.cross(e3, e1)
    return np.column_stack([e1, e2, e3])


def hand_to_gripper_orientation(
    hand_rot: np.ndarray,
    handle_pitch_offset_deg: float,
    handle_axis: Sequence[float] | None = None,
) -> np.ndarray:
    """Absolute hand->gripper orientation: the gripper's long axis (EE local
    x, wrist -> tip) mirrors the controller's handle axis (top -> bottom),
    1:1 and independent of grip-press history.
    """
    return hand_rot @ _handle_to_gripper_offset(handle_pitch_offset_deg, handle_axis)


def hand_to_gripper_orientation_armplane(
    hand_rot: np.ndarray,
    azimuth: float,
    handle_pitch_offset_deg: float,
    handle_axis: Sequence[float] | None = None,
    knuckle_axis: Sequence[float] | None = None,
) -> np.ndarray:
    """Build a FULLY-REACHABLE gripper orientation target for a 5-DOF arm.

    The SO-101 wrist can only point the gripper within the vertical plane
    the arm is panned to. So the target is assembled from three sources:
      - tip ELEVATION (up/down angle): from the controller handle axis,
      - tip AZIMUTH (compass direction): the arm's own current ``azimuth``
        (yaw follows the arm; the hand cannot and need not command it),
      - ROLL about the tip: from the controller's twist.

    Because the target always lies in the arm's reachable orientation set,
    all three orientation axes can be fully costed — unlike a zero-costed
    local axis, which silently permits a 180-degree tip flip (tip up when
    the hand says down).

    Hold the handle vertically -> gripper points straight down; twist the
    wrist -> the gripper rolls; lift the handle -> the gripper rises to
    face forward, in whatever direction the arm is panned.

    ``knuckle_axis``, when given (captured at first grip, perpendicular to
    the handle axis by construction), is the body-frame roll reference.
    Without it the reference is derived from an assumed pointer axis, which
    can be nearly parallel to the handle axis — the orthogonalized residual
    is then tiny and the roll response becomes weak and noisy.
    """
    if knuckle_axis is not None and handle_axis is not None:
        h = np.asarray(handle_axis, dtype=float)
        handle_world = hand_rot @ (h / np.linalg.norm(h))
        knuckle_world = hand_rot @ np.asarray(knuckle_axis, dtype=float)
    else:
        offset = _handle_to_gripper_offset(handle_pitch_offset_deg, handle_axis)
        handle_world = hand_rot @ offset[:, 0]
        knuckle_world = hand_rot @ offset[:, 2]

    # Tip: handle elevation, arm azimuth
    elev = np.arctan2(handle_world[2], max(np.linalg.norm(handle_world[:2]), 1e-9))
    ce, se = np.cos(elev), np.sin(elev)
    tip = np.array([ce * np.cos(azimuth), ce * np.sin(azimuth), se])

    # Roll: hand knuckle direction, orthogonalized against the tip
    z_axis = knuckle_world - (knuckle_world @ tip) * tip
    norm = np.linalg.norm(z_axis)
    if norm < 1e-6:
        # Degenerate (knuckles along the tip); fall back to world up
        z_axis = np.array([0.0, 0.0, 1.0]) - tip[2] * tip
        norm = np.linalg.norm(z_axis)
    z_axis = z_axis / norm
    y_axis = np.cross(z_axis, tip)
    return np.column_stack([tip, y_axis, z_axis])


def blend_rotations(
    rot_from: np.ndarray, rot_to: np.ndarray, alpha: float
) -> np.ndarray:
    """Geodesic interpolation between two rotation matrices (alpha in [0, 1])."""
    if alpha >= 1.0:
        return rot_to.copy()
    delta = Rotation.from_matrix(rot_from.T @ rot_to).as_rotvec()
    return rot_from @ Rotation.from_rotvec(alpha * delta).as_matrix()


def scale_and_add_delta_transform(
    delta_position: np.ndarray,
    delta_orientation: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
    initial_transform: np.ndarray,
) -> np.ndarray:
    """Scale and add delta position and orientation to an initial transform.

    Args:
        delta_position: 3D position vector
        delta_orientation: 3x3 rotation matrix
        translation_scale: translation scale
        rotation_scale: rotation scale
        initial_transform: 4x4 transformation matrix

    Returns:
        4x4 transformation matrix
    """
    scaled_delta_position = delta_position * translation_scale

    # Scale the delta rotation by converting to axis-angle, scaling the angle, then converting back
    delta_rotation = Rotation.from_matrix(delta_orientation)
    delta_axis_angle = delta_rotation.as_rotvec()
    scaled_delta_axis_angle = delta_axis_angle * rotation_scale
    scaled_delta_rotation = Rotation.from_rotvec(scaled_delta_axis_angle).as_matrix()

    # Compose rotations by matrix multiplication (correct way to combine rotations)
    initial_rotation = initial_transform[:3, :3]
    new_rotation = initial_rotation @ scaled_delta_rotation

    target_transform = np.eye(4)
    target_transform[:3, 3] = initial_transform[:3, 3] + scaled_delta_position
    target_transform[:3, :3] = new_rotation

    return target_transform
