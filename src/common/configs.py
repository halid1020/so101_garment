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


# IK Solver
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.75
FRAME_TASK_GAIN = 0.4
LM_DAMPING = 0.0
DAMPING_COST = 0.25
SOLVER_DAMPING_VALUE = 1e-12

# Teleoperation Thresholds
GRIP_THRESHOLD = 0.9

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
