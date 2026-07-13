"""Teleoperation configuration for the dual SO-101 rig.

Teleop *tuning* values (filtering, clutch, handle geometry, workspace envelope,
rates, scaling, rate limits, operator anthropometry) are the single source of
truth in ``src/ik_conf/teleop_shared.yaml`` and are bound onto the historical
constant names below at import time via ``load_teleop_shared`` (a strict load —
an unknown or missing key in that YAML is a hard error). Per-method IK weights
live in ``src/ik_conf/methods/<name>.yaml`` and are loaded where the solver is
built (tools + benchmark methods), not here.

What deliberately stays as plain Python here: per-machine calibration
(HW->URDF joint offsets/signs) and model wiring (URDF path, frame names). These
are not teleop tuning — they describe a specific robot and its URDF — so they do
not belong in the shared YAML.
"""

from pathlib import Path

from common.config_parser import load_teleop_shared

# Strict load of the shared teleop parameters (raises on any missing/unknown
# key, naming the file and key). Bound onto named constants just below.
_SHARED = load_teleop_shared()

# ---------------------------------------------------------------------------
# Model wiring (NOT teleop tuning): dual-arm URDF and its TCP/base frame names.
# ---------------------------------------------------------------------------
DUAL_URDF_PATH = str(
    Path(__file__).parent.parent / "so101_dual_description" / "robot.urdf"
)
END_EFFECTOR_FRAME_NAMES = ["left_eef_link", "right_eef_link"]
LEFT_END_EFFECTOR_FRAME_NAME = "left_eef_link"
RIGHT_END_EFFECTOR_FRAME_NAME = "right_eef_link"
LEFT_ARM_BASE_FRAME_NAME = "left_base_link"
RIGHT_ARM_BASE_FRAME_NAME = "right_base_link"

# ---------------------------------------------------------------------------
# Neutral pose / posture (also mirrored in methods/armplane.yaml, which the
# real-robot tool reads to build the solver). Kept here because the DUAL
# derivations below are consumed by the tools and the benchmark adapter as the
# IK integrator's initial configuration.
# ---------------------------------------------------------------------------
# SO101 neutral pose (degrees): 5 body joints
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
NEUTRAL_JOINT_ANGLES = [0.0, -10.0, 20.0, 25.0, 0.0]
# Posture-task cost vector (one weight per joint); zero = no posture pull.
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0]

# Dual-arm neutral configuration (10 body joints: left x5, right x5), composed
# in Python from the per-arm values above.
NEUTRAL_JOINT_ANGLES_DUAL = [*NEUTRAL_JOINT_ANGLES, *NEUTRAL_JOINT_ANGLES]
POSTURE_COST_VECTOR_DUAL = [*POSTURE_COST_VECTOR, *POSTURE_COST_VECTOR]

# ---------------------------------------------------------------------------
# Per-machine calibration (NOT teleop tuning): HW -> URDF joint zero offsets
# (degrees), one list per arm: [shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll].
# The LeRobot calibration zero does not coincide with the URDF zero
# (observed: wrist_roll reads 90 deg off in the visualizer; the lift /
# elbow / wrist_flex values were measured per arm with
# tool/fit_joint_offsets.py on 2026-07-02 — table-drag fit, model error
# reduced from +-8.5/13.8 mm to +-4 mm).
# urdf_deg = hw_deg + offset ; hw_deg = urdf_deg - offset.
# ---------------------------------------------------------------------------
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

# ===========================================================================
# Teleop tuning constants — bound from teleop_shared.yaml. Change values in the
# YAML (each carries its own detailed comment/rationale), NOT here.
# ===========================================================================
_filtering = _SHARED["filtering"]
_clutch = _SHARED["clutch"]
_handle = _SHARED["handle"]
_operator_frame = _SHARED["operator_frame"]
_envelope = _SHARED["envelope"]
_rates = _SHARED["rates"]
_scaling = _SHARED["scaling"]
_rate_limit = _SHARED["rate_limit"]
_operator = _SHARED["operator"]

# One-Euro filter parameters for controller-pose smoothing.
CONTROLLER_MIN_CUTOFF = _filtering["min_cutoff"]
CONTROLLER_BETA = _filtering["beta"]
CONTROLLER_D_CUTOFF = _filtering["d_cutoff"]

# Grip clutch.
GRIP_THRESHOLD = _clutch["grip_threshold"]
ORIENTATION_BLEND_TIME_S = _clutch["orientation_blend_time_s"]

# Handle geometry (controller frame -> gripper orientation).
HANDLE_PITCH_OFFSET_DEG = _handle["pitch_offset_deg"]
HANDLE_AXIS = _handle["axis"]

# Operator control frame origin offsets (moved here from common/utils.py so
# utils imports them from configs; configs does not import utils, so no cycle).
OPERATOR_FRAME_BACK_M = _operator_frame["back_m"]
OPERATOR_FRAME_UP_M = _operator_frame["up_m"]

# Workspace envelope (see common/workspace_envelope.py).
WORKSPACE_R_MIN = _envelope["r_min"]
WORKSPACE_R_MAX = _envelope["r_max"]
WORKSPACE_Z_FLOOR = _envelope["z_floor"]
WORKSPACE_SAFETY_MARGIN = _envelope["safety_margin"]
WORKSPACE_SOFT_MARGIN = _envelope["soft_margin"]
WORKSPACE_OOB_MODE = _envelope["oob_mode"]

# Teleop motion scaling (controller motion -> robot motion).
TRANSLATION_SCALE = _scaling["translation_scale"]
ROTATION_SCALE = _scaling["rotation_scale"]

# Thread execution rates (Hz).
CONTROLLER_DATA_RATE = _rates["controller_data"]
IK_SOLVER_RATE = _rates["ik_solver"]
VISUALIZATION_RATE = _rates["visualization"]
ROBOT_RATE = _rates["robot"]
JOINT_STATE_STREAMING_RATE = _rates["joint_state_streaming"]
CAMERA_FRAME_STREAMING_RATE = _rates["camera_frame_streaming"]

# Joint-space rate-limit defaults (rad/s): simulation vs real hardware. Consumed
# as defaults by the benchmark method adapter (sim) and the real-robot tool
# (hw); a CLI --max-joint-vel still overrides either.
MAX_JOINT_VEL_SIM_RAD_S = _rate_limit["max_joint_vel_sim_rad_s"]
MAX_JOINT_VEL_HW_RAD_S = _rate_limit["max_joint_vel_hw_rad_s"]

# Operator anthropometry (see the YAML for the full derivation chain).
OPERATOR_HEIGHT_M = _operator["height_m"]
