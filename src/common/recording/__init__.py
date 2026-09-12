"""LeRobot data-collection pipeline for the dual SO-101 rig.

Modules:
* ``features`` — pure feature-schema and frame builders (state/action layout);
* ``cameras`` — generic UVC capture threads publishing into DualDataManager;
* ``sidecar`` — ~100 Hz full-rate parquet sampler (per-episode, <root>/extra/);
* ``episode_recorder`` — the fps-paced state machine owning the dataset writer.
"""

from actoris_harena.recording.cameras import CameraCapture
from actoris_harena.recording.config import load_recording_config
from actoris_harena.recording.features import (
    ACTION_FRESH_S,
    assemble_frame,
    build_action,
    build_dataset_features,
    build_observation_state,
)

from common.recording.episode_recorder import EpisodeRecorder, RecorderState
from common.recording.sidecar import SidecarSampler, compute_world_base_transforms
from common.robot_schema import STATE_NAMES

__all__ = [
    "ACTION_FRESH_S",
    "CameraCapture",
    "EpisodeRecorder",
    "RecorderState",
    "STATE_NAMES",
    "SidecarSampler",
    "assemble_frame",
    "build_action",
    "build_dataset_features",
    "build_observation_state",
    "compute_world_base_transforms",
    "load_recording_config",
]
