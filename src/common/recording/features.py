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


def build_dataset_features(
    camera_specs: list[tuple[str, int, int]],
) -> dict[str, dict]:
    """Return the LeRobotDataset feature spec for the enabled streams.

    ``camera_specs`` is a list of ``(name, height, width)`` for each ENABLED
    camera; each becomes an ``observation.images.<name>`` video feature.
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
    for name, height, width in camera_specs:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


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
) -> dict:
    """Build the LeRobotDataset frame dict (features + task, no bookkeeping keys).

    ``images`` maps each enabled camera name to its RGB (H, W, 3) uint8 array.
    Never adds timestamp/frame_index — LeRobot derives those from fps.
    """
    frame: dict = {
        "observation.state": np.asarray(observation_state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    for name, rgb in images.items():
        frame[f"observation.images.{name}"] = rgb
    return frame
