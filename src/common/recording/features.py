"""Pure feature-schema and frame builders for the LeRobot recorder.

No hardware, no dataset, no threads — just the mapping from the dual-arm state
(measured URDF-degree joints, gripper fractions, last sent command, camera
frames) to the LeRobotDataset feature specification and per-frame dicts. Kept
dependency-light so it unit-tests fast.

Design notes:
* ``observation.state`` (12,) = measured joints in URDF degrees for the five
  body joints per arm, each arm's gripper open fraction appended (0 = closed,
  1 = open). Order: left five body joints, left gripper, then the right arm.
* ``action`` (12,) = the joint-space command actually sent to the motors, in
  the SAME layout and units, used only while fresh (age < ACTION_FRESH_S);
  otherwise each stale side falls back to that side's measured state (covers
  HOMING moves and a released clutch, both of which bypass the command path).
* Cameras become ``observation.images.<name>`` video features (H, W, 3).
The whole internal pipeline is URDF degrees; the hardware conversion is
confined to the bus boundary, and pi0.5 normalises with dataset statistics.
"""

from __future__ import annotations

import numpy as np
from actoris_harena.sync import mat_to_quat

# The five actuated body joints per SO-101 arm, in URDF order.
BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SIDES = ["left", "right"]
BODY_DOF = 5

# The 12 state/action channel names: per side the five body joints then the
# gripper, left arm first. e.g. "left_shoulder_pan.pos", ..., "right_gripper.pos".
STATE_NAMES: list[str] = [
    f"{side}_{joint}.pos" for side in SIDES for joint in (*BODY_JOINTS, "gripper")
]

# A command is used as the action only while fresher than this (seconds). The
# joint threads write at ~100 Hz, so 30 ms comfortably admits the latest write.
ACTION_FRESH_S = 0.030

# Non-policy phase flag recorded per frame. It is a bare top-level key (not
# under ``observation.`` and not ``action``), so LeRobot's feature classifier
# ignores it for policy input/output (feature_utils.dataset_to_policy_features)
# while it stays queryable in the dataset — used to mask non-teleop frames
# (e.g. homing moves, where the action falls back to the measured state) at
# train time. 1.0 = teleoperation active, 0.0 = not.
TELEOP_ACTIVE_KEY = "teleop_active"

# End-effector pose channels per side: 3 position + 4 quaternion (w, x, y, z),
# expressed in that arm's OWN base frame. ``ee_pose`` is the measured pose;
# ``ee_target`` is the projected+constrained TARGET pose (the label an EE-space
# policy predicts). BOTH use NEUTRAL top-level keys (not ``observation.`` /
# ``action``) so LeRobot's feature classifier ignores them for the default
# joint-space policy — its state/action are unchanged by their presence. An
# EE-space experiment remaps ``ee_target`` → action (and optionally ``ee_pose``
# → an observation) explicitly, so both policies train from the SAME episodes
# with no re-collection. Recorded only in quest/IK mode, where EE targets exist.
_EE_COMPONENTS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
EE_NAMES: list[str] = [f"{side}_{c}" for side in SIDES for c in _EE_COMPONENTS]
EE_DOF = len(EE_NAMES)  # 14
OBS_EE_KEY = "ee_pose"
ACTION_EE_KEY = "ee_target"


def pose_to_vec7(pose_4x4: np.ndarray) -> np.ndarray:
    """4×4 homogeneous pose → (7,) float32 [x, y, z, qw, qx, qy, qz]."""
    p = np.asarray(pose_4x4, dtype=np.float64)
    quat = mat_to_quat(p[:3, :3])  # (w, x, y, z)
    return np.array([*p[:3, 3], *quat], dtype=np.float32)


def build_dataset_features(
    camera_specs: list[tuple[str, int, int]],
    include_phase: bool = False,
    include_ee: bool = False,
) -> dict[str, dict]:
    """Return the LeRobotDataset feature spec for the enabled streams.

    ``camera_specs`` is a list of ``(name, height, width)`` for each ENABLED
    camera; each becomes an ``observation.images.<name>`` video feature. When
    ``include_phase`` is set, a maskable ``teleop_active`` scalar is added (the
    real teleop recorder opts in; the sim oracle collector leaves it off so its
    schema is unchanged). When ``include_ee`` is set, the EE-space state
    (``observation.ee_pose``) and target (``action_ee``) features are added so
    an EE-space policy trains from the same episodes as the joint-space one.
    """
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
    }
    if include_ee:
        features[OBS_EE_KEY] = {
            "dtype": "float32",
            "shape": (EE_DOF,),
            "names": list(EE_NAMES),
        }
        features[ACTION_EE_KEY] = {
            "dtype": "float32",
            "shape": (EE_DOF,),
            "names": list(EE_NAMES),
        }
    for name, height, width in camera_specs:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    if include_phase:
        features[TELEOP_ACTIVE_KEY] = {
            "dtype": "float32",
            "shape": (1,),
            "names": [TELEOP_ACTIVE_KEY],
        }
    return features


def fresh_sides(
    last_commands: dict[str, tuple[np.ndarray | None, float | None, float | None]],
    now_mono: float,
    fresh_s: float = ACTION_FRESH_S,
) -> set[str]:
    """Return the sides whose last command is fresh enough to be the action.

    The recorder uses this to tally how often the action fell back to the
    measured state (a stale/missing command), which flags frames that teach a
    spurious "hold" and should be masked or excluded at train time.
    """
    out: set[str] = set()
    for side in SIDES:
        _urdf, _grip, t_mono = last_commands[side]
        if t_mono is not None and (now_mono - t_mono) < fresh_s:
            out.add(side)
    return out


def build_observation_state(
    measured_joints_10: np.ndarray,
    gripper_open: dict[str, float],
) -> np.ndarray:
    """Assemble the (12,) float32 observation.state vector.

    ``measured_joints_10`` is the 10-DOF URDF-degree joint array (left five,
    right five); ``gripper_open`` maps each side to its 0-1 open fraction.
    """
    joints = np.asarray(measured_joints_10, dtype=np.float64)
    if joints.shape != (BODY_DOF * len(SIDES),):
        raise ValueError(
            f"measured_joints_10 must have shape ({BODY_DOF * len(SIDES)},), "
            f"got {joints.shape}"
        )
    out = np.empty(len(STATE_NAMES), dtype=np.float32)
    for s, side in enumerate(SIDES):
        base = s * (BODY_DOF + 1)
        out[base : base + BODY_DOF] = joints[s * BODY_DOF : (s + 1) * BODY_DOF]
        out[base + BODY_DOF] = gripper_open[side]
    return out


def build_action(
    observation_state: np.ndarray,
    last_commands: dict[str, tuple[np.ndarray | None, float | None, float | None]],
    now_mono: float,
    fresh_s: float = ACTION_FRESH_S,
) -> np.ndarray:
    """Assemble the (12,) float32 action vector from the last sent commands.

    ``last_commands[side]`` is ``(urdf_deg_5, gripper_open, t_mono)`` as
    returned by ``DualDataManager.get_last_sent_command``. Each side whose
    command is missing or older than ``fresh_s`` falls back to that side's
    slice of ``observation_state`` (deterministic: covers HOMING and a
    released clutch, where no fresh command exists).
    """
    action = np.asarray(observation_state, dtype=np.float32).copy()
    for s, side in enumerate(SIDES):
        base = s * (BODY_DOF + 1)
        urdf_deg, gripper_open, t_mono = last_commands[side]
        if (
            urdf_deg is not None
            and gripper_open is not None
            and t_mono is not None
            and (now_mono - t_mono) < fresh_s
        ):
            action[base : base + BODY_DOF] = np.asarray(urdf_deg, dtype=np.float32)
            action[base + BODY_DOF] = np.float32(gripper_open)
    return action


def assemble_frame(
    observation_state: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
    task: str,
    teleop_active: bool | None = None,
    ee_pose: np.ndarray | None = None,
    ee_target: np.ndarray | None = None,
) -> dict:
    """Build the LeRobotDataset frame dict (features + task, no bookkeeping keys).

    ``images`` maps each enabled camera name to its RGB (H, W, 3) uint8 array.
    ``teleop_active`` records whether teleoperation drove this frame (for
    train-time masking); pass ``None`` (the default, used by the sim collector)
    to omit the key so the frame matches a feature spec built without
    ``include_phase``. ``ee_pose`` / ``ee_target`` are the (14,) measured and
    projected+constrained EE vectors; pass ``None`` to omit them (feature spec
    built without ``include_ee``). Never adds timestamp/frame_index — LeRobot
    derives those from fps.
    """
    frame: dict = {
        "observation.state": np.asarray(observation_state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if teleop_active is not None:
        frame[TELEOP_ACTIVE_KEY] = np.array(
            [1.0 if teleop_active else 0.0], dtype=np.float32
        )
    if ee_pose is not None:
        frame[OBS_EE_KEY] = np.asarray(ee_pose, dtype=np.float32)
    if ee_target is not None:
        frame[ACTION_EE_KEY] = np.asarray(ee_target, dtype=np.float32)
    for name, rgb in images.items():
        frame[f"observation.images.{name}"] = rgb
    return frame
