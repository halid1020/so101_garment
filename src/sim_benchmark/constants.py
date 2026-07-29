"""Shared constants for the teleop simulation benchmark."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DUAL_URDF_PATH = REPO_ROOT / "src" / "so101_dual_description" / "robot.urdf"
DESCRIPTION_DIR = DUAL_URDF_PATH.parent

SIDES = ("left", "right")

# Per-arm actuated joints, in kinematic order (grippers excluded from IK).
ARM_JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Flat joint ordering used for every q vector in this package:
# 5 left-arm joints then 5 right-arm joints.
ARM_JOINTS = tuple(
    f"{side}_{suffix}" for side in SIDES for suffix in ARM_JOINT_SUFFIXES
)

GRIPPER_JOINTS = tuple(f"{side}_gripper" for side in SIDES)

EE_FRAMES = {side: f"{side}_eef_link" for side in SIDES}

# Neutral pose (degrees) mirroring common.configs.NEUTRAL_JOINT_ANGLES_DUAL.
NEUTRAL_ARM_ANGLES_DEG = (0.0, -10.0, 20.0, 25.0, 0.0)

# Control loop rate for the IK/teleop layer (matches CONTROLLER_DATA_RATE).
CONTROL_RATE_HZ = 50.0

# Position-servo gains for the simulated STS3215 bus servos.
ACTUATOR_KP = 25.0
ACTUATOR_KV = 1.5

# Joint dynamics matching the official SO-ARM100 MJCF (STS3215 servos).
# The URDF carries none; both the benchmark scene and the sim twin apply
# these programmatically.
JOINT_DAMPING = 0.60
JOINT_ARMATURE = 0.028
JOINT_FRICTIONLOSS = 0.05
