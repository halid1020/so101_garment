"""Configuration parameters for AgileX Piper robot demos."""

from pathlib import Path

# Hardware & Kinematics Paths
URDF_PATH = str(
    Path(__file__).parent.parent.parent
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)
GRIPPER_FRAME_NAME = "gripper_center"


# Two-arm IK configuration (left/right hand TCP frames from URDF)
DUAL_URDF_PATH = str(
    Path(__file__).parent.parent / "so101_dual_description" / "robot.urdf"
)
END_EFFECTOR_FRAME_NAMES = ["left_eef_link", "right_eef_link"]
LEFT_END_EFFECTOR_FRAME_NAME = "left_eef_link"
RIGHT_END_EFFECTOR_FRAME_NAME = "right_eef_link"
LEFT_ARM_BASE_FRAME_NAME = "left_base_link"
RIGHT_ARM_BASE_FRAME_NAME = "right_base_link"

# SO101 neutral pose (degrees): 5 body joints [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
NEUTRAL_JOINT_ANGLES = [0.0, -10.0, 20.0, 25.0, 0.0]
NEUTRAL_JOINT_ANGLES_2 = [0.0, -10.0, 20.0, 25.0, 0.0]
# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0]
POSTURE_COST_VECTOR_2 = [0.0, 0.0, 0.0, 0.0, 0.0]

# Dual-arm neutral configuration (10 body joints: left×5, right×5)
NEUTRAL_JOINT_ANGLES_DUAL = [*NEUTRAL_JOINT_ANGLES, *NEUTRAL_JOINT_ANGLES]
POSTURE_COST_VECTOR_DUAL = [*POSTURE_COST_VECTOR, *POSTURE_COST_VECTOR]


# Hardware -> URDF joint zero offsets (degrees), one list per arm:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
# The LeRobot calibration zero does not coincide with the URDF zero
# (observed: wrist_roll reads 90 deg off in the visualizer; the lift /
# elbow / wrist_flex values were measured per arm with
# tool/fit_joint_offsets.py on 2026-07-02 — table-drag fit, model error
# reduced from +-8.5/13.8 mm to +-4 mm).
# urdf_deg = hw_deg + offset ; hw_deg = urdf_deg - offset.
LEFT_ARM_HW_TO_URDF_OFFSETS_DEG = [0.0, -3.45, 12.26, -9.72, 90.0]
RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG = [0.0, -3.97, -5.43, -1.02, 90.0]

# Per-joint direction signs for the same mapping (+1 or -1):
# urdf_deg = sign * hw_deg + offset ; hw_deg = sign * (urdf_deg - offset).
# A -1 means the servo's positive direction is OPPOSITE to the URDF's for
# that joint (the LeRobot calibration does not guarantee they agree).
# Check with `python tool/check_mirror.py`: bend each joint by hand and
# flip the sign of any joint where the on-screen robot moves the WRONG way.
# NOTE: after changing a sign, re-run tool/fit_joint_offsets.py — offsets
# fitted under the wrong sign are invalid for that joint.
LEFT_ARM_HW_TO_URDF_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0]
RIGHT_ARM_HW_TO_URDF_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0]

# IK Solver
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.75

# Per-axis orientation cost mask, in the eef_link LOCAL frame.
# All axes are costed. (An earlier design zero-costed local z so "yaw
# follows the arm", but the zero-cost family R_target*Rz(phi) includes
# phi=180 deg — a full tip flip, so the gripper could face UP when the
# hand said down. Yaw-follows-arm is now achieved by constructing the
# target inside the arm's reachable set instead: see
# hand_to_gripper_orientation_armplane in common/utils.py.)
EE_ORIENTATION_COST_MASK = [1.0, 1.0, 1.0]

# Absolute hand->gripper orientation mapping. The gripper's long axis
# (wrist -> tip) mirrors the controller's HANDLE axis (top -> bottom).
# The Quest's tracked aim pose has local -z along the POINTER ray and -y
# roughly along the handle; the physical handle is tilted backward from
# -y by about this angle. With it, holding the handle plumb-vertical
# (natural upright fist) points the gripper straight down at the table.
# Tune: this angle sets the gripper's neutral fore/aft tilt one-to-one.
# If at your natural grip the gripper leans N degrees toward the FRONT of
# the setup, ADD N here (it should lean toward the operator, like the
# handle does); if it leans too far back toward you, subtract.
HANDLE_PITCH_OFFSET_DEG = 65.0

# Measured handle top->bottom axis in the controller body frame.
# Run `python tool/calibrate_handle.py` and paste its output here; when
# set, it overrides HANDLE_PITCH_OFFSET_DEG entirely.
# Measured 2026-07-02 (right controller, 148 samples, 4.7 deg spread).
# Note the dominant +x: the tracked frame's axes are nothing like the
# OpenXR aim convention we first assumed — measuring was the right call.
HANDLE_AXIS = [0.8242, 0.2110, -0.5255]

# Seconds to blend from the gripper's orientation at grip-press to the
# absolute hand orientation, so activating teleop never jerks the wrist.
ORIENTATION_BLEND_TIME_S = 1.0
FRAME_TASK_GAIN = 0.4
LM_DAMPING = 0.0
DAMPING_COST = 0.25
SOLVER_DAMPING_VALUE = 1e-12

# Teleoperation Thresholds
GRIP_THRESHOLD = 0.9

# Workspace envelope (see common/workspace_envelope.py). The annulus radii
# are the EE-to-shoulder-lift-pivot distance extremes from a FK grid sweep
# of elbow_flex x wrist_flex over the URDF limits (invariant to
# shoulder_lift). Re-derived by test/test_workspace_envelope.py so URDF
# edits cannot silently stale them. Values from robot.urdf, 2026-07-08.
WORKSPACE_R_MIN = 0.0837  # m, swept minimum reach from the lift pivot
WORKSPACE_R_MAX = 0.4110  # m, swept maximum reach from the lift pivot
WORKSPACE_Z_FLOOR = 0.01  # m, table clearance floor for EE targets
WORKSPACE_SAFETY_MARGIN = 0.01  # m, shaved off both radii in build_envelopes
WORKSPACE_SOFT_MARGIN = 0.04  # m, slowdown band width ("slow" policy)
# Default out-of-envelope policy: "warn" | "project" | "freeze" | "slow".
WORKSPACE_OOB_MODE = "warn"

# One-Euro filter parameters for controller pose smoothing
CONTROLLER_MIN_CUTOFF = 0.8
CONTROLLER_BETA = 5.0
CONTROLLER_D_CUTOFF = 0.9

# Teleop motion scaling (controller motion -> robot motion)
TRANSLATION_SCALE = 1.0
ROTATION_SCALE = 1.0

# Thread Execution Rates (Hz)
CONTROLLER_DATA_RATE = 50.0
IK_SOLVER_RATE = 100
VISUALIZATION_RATE = 60.0
ROBOT_RATE = 100.0
JOINT_STATE_STREAMING_RATE = 100.0
CAMERA_FRAME_STREAMING_RATE = 30.0

# Hardware Context
META_QUEST_AXIS_MASK = [1, 1, 1, 1, 1, 1]
CAMERA_DEVICE_INDEX = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["rgb_scene", "rgb_wrist"]

# Default Robot Poses
NEUTRAL_JOINT_ANGLES = [-1.6, 52.2, -54.3, -3.2, 43.1, 4.7]
NEUTRAL_END_EFFECTOR_POSE = [455.257, -46.344, 172.213, 176.205, -14.545, 29.621]
# NEUTRAL_JOINT_ANGLES = [-9.3, 86.7, -86.6, 1.8, 61.7, -5.1] #Lemon pick task pruthvi

# AI Policy Execution Parameters
POLICY_EXECUTION_RATE = 20.0
PREDICTION_HORIZON_EXECUTION_RATIO = 1.0
MAX_SAFETY_THRESHOLD = 200.0
MAX_ACTION_ERROR_THRESHOLD = 3.0
TARGETING_POSE_TIME_THRESHOLD = 1.0

GRIPPER_NAME = "gripper"
GRIPPER_LOGGING_NAME = GRIPPER_NAME
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
