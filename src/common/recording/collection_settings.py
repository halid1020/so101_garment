"""Read an existing dataset's stream settings and turn a selection into flags.

The friendly collection front-end (``tool/collect_dataset.py``) drives the same
recorder as ``tool/meta_quest_teleopration.py --record``. Two pure helpers back
it, kept dependency-light so they unit-test without a dataset or hardware:

* :func:`read_existing_streams` reads the LeRobotDataset metadata (``meta/
  info.json`` plus the sidecar ``meta/realsense.json``) to recover exactly which
  streams a dataset was collected with, so resuming it follows those settings
  instead of asking again.
* :func:`selection_to_teleop_flags` turns a stream selection into the precise
  ``--enable-camera`` / ``--disable-camera`` / ``--central-depth`` /
  ``--no-record-ee`` / ``--dataset-fps`` flag set the recorder needs, forcing
  the resolved streams to match regardless of ``recording.yaml`` defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

_IMAGE_PREFIX = "observation.images."


def read_existing_streams(root: "str | Path") -> dict:
    """Recover the stream settings of the dataset at ``root``.

    Returns a dict with:

    * ``cameras`` — the set of every ``observation.images.<name>`` feature
      (both UVC cameras and the central RealSense RGB);
    * ``depth`` — ``True`` iff the dataset carries a central RealSense
      (``meta/realsense.json`` present);
    * ``depth_rgb_name`` — the RealSense RGB feature name, or ``None``;
    * ``ee`` — ``True`` iff the EE-space features were recorded;
    * ``fps`` — the dataset frame rate (int).

    Raises ``FileNotFoundError`` if ``meta/info.json`` is missing (not a
    dataset). Pure read of the metadata JSONs — no dataset load.
    """
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features", {})
    cameras = {k[len(_IMAGE_PREFIX) :] for k in features if k.startswith(_IMAGE_PREFIX)}
    rs_path = root / "meta" / "realsense.json"
    depth = rs_path.is_file()
    depth_rgb_name = None
    if depth:
        depth_rgb_name = json.loads(rs_path.read_text()).get("rgb_name")
    ee = "ee_pose" in features or "ee_target" in features
    return {
        "cameras": cameras,
        "depth": depth,
        "depth_rgb_name": depth_rgb_name,
        "ee": ee,
        "fps": int(info["fps"]),
    }


def is_resumable_dataset(root: "str | Path") -> bool:
    """Whether the dataset at ``root`` can be appended to (has ≥ 1 episode).

    A dataset created but quit before any episode was saved has an
    ``info.json`` but no ``meta/tasks.parquet`` and ``total_episodes == 0``.
    LeRobot's ``resume`` cannot load such a stillborn dataset: its metadata
    read fails and it then wrongly falls back to the HuggingFace Hub (a
    misleading 401 for a local repo id). Gating resume on this predicate keeps
    that case a clear local error. Pure — unit-tested.
    """
    info = Path(root) / "meta" / "info.json"
    if not info.is_file():
        return False
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0)) > 0
    except (ValueError, json.JSONDecodeError):
        return False


def uvc_cameras(cameras: "set[str]", depth_rgb_name: "str | None") -> "set[str]":
    """The UVC camera names in ``cameras`` (excluding the RealSense RGB name).

    The central RealSense RGB is a video feature like any other, but the
    recorder opens it through ``--central-depth`` rather than as a UVC camera,
    so it must be split out before building the camera enable/disable flags.
    """
    return {c for c in cameras if c != depth_rgb_name}


def selection_to_teleop_flags(
    known_cameras: "set[str]",
    enabled_uvc: "set[str]",
    depth: bool,
    record_ee: bool,
    fps: "int | None" = None,
) -> "list[str]":
    """Flags that reproduce this stream selection for the teleop recorder.

    ``known_cameras`` is every UVC camera name in ``recording.yaml``; each is
    explicitly enabled or disabled so the resolved set equals ``enabled_uvc``
    whatever the yaml defaults are (essential when resuming, where the recorded
    camera set must be matched exactly). ``depth`` adds ``--central-depth``;
    ``record_ee=False`` adds ``--no-record-ee``; ``fps`` pins ``--dataset-fps``
    when given. Pure — unit-tested.
    """
    flags: list[str] = []
    for cam in sorted(known_cameras):
        if cam in enabled_uvc:
            flags += ["--enable-camera", cam]
        else:
            flags += ["--disable-camera", cam]
    if depth:
        flags.append("--central-depth")
    if not record_ee:
        flags.append("--no-record-ee")
    if fps is not None:
        flags += ["--dataset-fps", str(int(fps))]
    return flags
