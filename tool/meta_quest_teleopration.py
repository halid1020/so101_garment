#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest (LeRobot backend).

Left Meta Quest hand → left SO101 arm (arm 0, PORT_ID_0).
Right Meta Quest hand → right SO101 arm (arm 1, PORT_ID_1).
Single 10-DOF IK solver on the dual-arm URDF.

--input leader instead drives the followers from two SO-101 LEADER arms
in direct joint-to-joint control (no IK, no clutch — the followers mirror
the leaders whenever enabled). Leader ports/ids come from
src/conf/sensor_map.yaml (tool/test_sensor_rates.py --assign). Control is
by keyboard: Y enable + start follow, B home, X park, A episode (with
--record), Q quit — typed into the --sensor-view window when it is open,
otherwise into the terminal.

Optionally records LeRobot-format episodes (--record): a training-ready
dataset at the configured fps plus a ~100 Hz full-rate sidecar parquet per
episode, controlled from the Quest handles (or the A key in leader mode).

--sensor-view opens a live window (the same layout as
tool/test_sensor_rates.py --view): a row of tactile cameras above one
cell per side showing that side's follower and leader joints (Quest mode:
follower measured vs commanded). Cameras come from src/conf/sensor_map.yaml
or ad-hoc --view-camera NAME=DEV. In Quest mode q/Esc closes just the
window; in leader mode the window is the control surface (Q quits).

Controls (the Y/X/A/B semantics apply even WITHOUT --record):
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Hold triggers           - close grippers
  Thumbstick (mymethod)   - deflect to trim that arm's wrist (x = roll,
                            y = flex); the arm's other joints freeze while
                            the stick is deflected and the handle is ignored
                            for that arm, then resumes from the new pose on
                            release
  Button Y                - ENABLE: torque on + move both arms to ready pose
  Button X                - PARK: move both arms to rest pose + torque off
                            (refused while an episode is being recorded)
  Button A                - EPISODE toggle (--record): move to ready then
                            START recording; press again to move to ready
                            (still recorded) then STOP and SAVE
  Button B                - move both arms to the ready pose (recording, if
                            any, keeps running — the motion stays in the
                            episode)
  Joystick clicks (LJ/RJ) - glide that gripper's roll back to neutral
                            at the next grip
  Ctrl+C                  - exit (discards any in-flight episode)
"""

import argparse
import os
import shutil
import signal
import sys
import termios
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import yaml
from actoris_harena.recording.camera_controls import CONTROL_NAMES
from actoris_harena.recording.device_faults import bus_gone
from meta_quest_teleop.reader import MetaQuestReader

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    GRIPPER_OPEN_MAX_FRAC,
    IK_SOLVER_RATE,
    MAX_JOINT_VEL_HW_RAD_S,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.keyboard_buttons import KeyboardButtons
from common.recording import (
    CameraCapture,
    EpisodeRecorder,
    RecorderState,
    SidecarSampler,
    build_dataset_features,
    load_recording_config,
)
from common.recording.controls import control_steps
from common.recording.depth import DepthWriter
from common.recording.monitor_server import MonitorServer, allowed_keys_for
from common.sensor_view import CollectionStatus, run_sensor_view_loop
from common.teleop_setup import add_teleop_cli_args, create_teleop_stack
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from common.threads.leader_arm import leader_arm_thread
from src.so101_dual_arm import SO101DualArm

# Disk-space thresholds for --record (GB free on the dataset volume).
_DISK_WARN_GB = 10.0
_DISK_REFUSE_GB = 2.0

# Streaming video-encoder backlog PER camera (frames) when streaming encoding is
# on. Each camera frame is encoded in a background thread during recording so the
# A-to-save pause stays short; this bounds how far the encoder may fall behind
# before LeRobot drops a backlogged frame (with a warning). ~90 ≈ 3 s at 30 fps.
_ENCODER_QUEUE_MAXSIZE = 90


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def add_recording_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the data-collection flags (all inert unless --record)."""
    group = parser.add_argument_group("recording (LeRobot data collection)")
    group.add_argument(
        "--record",
        action="store_true",
        help="Record LeRobot-format episodes controlled by the A button",
    )
    group.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset repo id, e.g. halid/so101_towel (required with --record; "
        "local only, never pushed to the hub)",
    )
    group.add_argument(
        "--task",
        type=str,
        default=None,
        help="Language task string stored with every frame " "(required with --record)",
    )
    group.add_argument(
        "--dataset-fps",
        type=int,
        default=None,
        help="Dataset frame rate; default comes from src/conf/recording.yaml",
    )
    group.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset directory; default $HF_LEROBOT_HOME/<repo-id>",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to an existing dataset instead of creating one",
    )
    group.add_argument(
        "--enable-camera",
        action="append",
        default=[],
        metavar="NAME",
        help="Enable a camera stream by name, overriding recording.yaml "
        "(repeatable)",
    )
    group.add_argument(
        "--disable-camera",
        action="append",
        default=[],
        metavar="NAME",
        help="Disable a camera stream by name, overriding recording.yaml "
        "(repeatable)",
    )
    group.add_argument(
        "--tactile",
        action="store_true",
        help="Enable all four tactile gripper camera streams at once",
    )
    group.add_argument(
        "--central-depth",
        action="store_true",
        help="Enable the central RealSense RGB-D camera (RGB video feature + "
        "aligned 16-bit depth stream); overrides realsense.enabled in "
        "src/conf/recording.yaml",
    )
    group.add_argument(
        "--no-record-ee",
        action="store_true",
        help="Do not record the EE-space features (ee_pose measured + "
        "ee_target projected/constrained); EE features are on by default in "
        "quest mode and always off in --input leader (no IK targets)",
    )
    group.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Disable the ~100 Hz full-rate sidecar parquet",
    )
    group.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable the audible record start/stop cue (overrides "
        "audio.enabled in src/conf/recording.yaml)",
    )
    group.add_argument(
        "--no-streaming-encode",
        action="store_true",
        help="Encode each episode's videos in one blocking pass at save time "
        "instead of streaming them during recording. Streaming (the default) "
        "keeps the A-to-save pause short so collection stays continuous; opt "
        "out if a CPU-starved rig drops encoder frames",
    )
    group.add_argument(
        "--episode-goal",
        type=int,
        default=0,
        metavar="N",
        help="Target episode count; shows an episodes n/N progress bar on the "
        "--sensor-view footer (0 = no goal, bar hidden)",
    )


def resolve_camera_streams(rec_cfg: dict, args: argparse.Namespace) -> dict:
    """Return {name: camera-config} for the streams enabled after overrides."""
    from tool.test_sensor_rates import TACTILE_CAMERA_NAMES

    cameras = rec_cfg["cameras"]
    known = set(cameras)
    for name in [*args.enable_camera, *args.disable_camera]:
        if name not in known:
            raise SystemExit(
                f"Unknown camera '{name}' (known: {sorted(known)}) — "
                "check src/conf/recording.yaml"
            )
    enabled = {}
    for name, cfg in cameras.items():
        on = bool(cfg["enabled"])
        if args.tactile and name in TACTILE_CAMERA_NAMES:
            on = True
        if name in args.enable_camera:
            on = True
        if name in args.disable_camera:
            on = False
        if on:
            enabled[name] = cfg
    return enabled


def overlay_sensor_map_devices(streams: dict, sensor_map: dict) -> dict:
    """Prefer the stable by-path node from ``sensor_map`` over the index.

    For every enabled stream whose name is also assigned in
    ``sensor_map["cameras"]`` (via ``tool/test_sensor_rates.py --assign``),
    replace its ``device`` with that stable ``/dev/v4l/by-path`` node so a
    replug cannot reorder it. Streams absent from the map keep their
    ``recording.yaml`` index. Returns a NEW dict and does not mutate the input.
    """
    cams = (sensor_map or {}).get("cameras") or {}
    out: dict = {}
    for name, cfg in streams.items():
        node = cams.get(name)
        out[name] = {**cfg, "device": node} if node else cfg
    return out


def resolve_realsense_serial(rs_cfg: dict, sensor_map: dict) -> str:
    """Serial for the central RealSense: sensor_map assignment wins.

    Prefers the serial stored by ``tool/test_sensor_rates.py --assign``
    (``sensor_map["realsense"]["serial"]``), falling back to the
    ``recording.yaml`` serial, then ``""`` (= the first RealSense found).
    Pure — unit-tested.
    """
    assigned = (sensor_map or {}).get("realsense") or {}
    return assigned.get("serial") or (rs_cfg or {}).get("serial") or ""


def build_realsense_capture(
    rec_cfg: dict, args: argparse.Namespace, sensor_map: dict, *, for_view: bool
):
    """Construct (not open) the central RealSenseCapture, or None if unwanted.

    Recording wants it when it is ``enabled`` or ``--central-depth``; the live
    view additionally wants it when a serial is assigned in ``sensor_map`` (so
    an assigned central camera shows in ``--sensor-view`` with no extra flag,
    mirroring the wrist cameras). Returns None WITHOUT importing pyrealsense2
    when the camera is not wanted, so the import cost is paid only on the
    hardware path.
    """
    rs_cfg = rec_cfg.get("realsense")
    if rs_cfg is None:
        return None
    wanted = rs_cfg["enabled"] or args.central_depth
    if for_view:
        assigned = bool((sensor_map or {}).get("realsense", {}).get("serial"))
        wanted = wanted or assigned
    if not wanted:
        return None
    from common.recording.realsense_camera import RealSenseCapture

    return RealSenseCapture(
        rgb_name=rs_cfg["rgb_name"],
        depth_name=rs_cfg["depth_name"],
        width=rs_cfg["width"],
        height=rs_cfg["height"],
        fps=rs_cfg["fps"],
        serial=resolve_realsense_serial(rs_cfg, sensor_map),
        align_to_color=rs_cfg["align_to_color"],
        lock_auto_exposure=rs_cfg["lock_auto_exposure"],
    )


def check_disk_space(root: Path) -> None:
    """Warn below 10 GB free, refuse to record below 2 GB."""
    probe = root
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < _DISK_REFUSE_GB:
        raise SystemExit(
            f"❌ Only {free_gb:.1f} GB free on {probe} — refusing to record "
            f"(needs ≥ {_DISK_REFUSE_GB:.0f} GB)"
        )
    if free_gb < _DISK_WARN_GB:
        print(f"⚠️  Low disk space: {free_gb:.1f} GB free on {probe}")


def write_realsense_meta(root: Path, rs_capture, rs_cfg: dict) -> None:
    """Write the central camera's intrinsics/scale/config for replicability.

    Saved once per dataset to ``<root>/meta/realsense.json``. The camera's
    extrinsic pose in the rig frame is a physical measurement made separately,
    so it is emitted as a null placeholder for the operator to fill in.
    """
    import json

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rgb_name": rs_capture.name,
        "depth_name": rs_capture.depth_name,
        "width": rs_cfg["width"],
        "height": rs_cfg["height"],
        "fps": rs_cfg["fps"],
        "aligned_to_color": rs_cfg["align_to_color"],
        "depth_scale_m_per_unit": rs_capture.depth_scale,
        "color_intrinsics": rs_capture.intrinsics,
        "extrinsics_camera_to_rig": None,  # measure + fill in for replicability
    }
    (meta_dir / "realsense.json").write_text(json.dumps(payload, indent=2))
    print(f"  🗂  wrote {meta_dir / 'realsense.json'} (intrinsics + depth scale)")


def write_action_space_meta(root: Path, fps: int, record_ee: bool) -> None:
    """Record the action-definition constants so inference reproduces them.

    A policy's output must pass back through the SAME command path that
    produced the labels (joint clamp/gripper-cap, or the IK + workspace
    envelope for the EE target). Persisting these constants next to the
    dataset lets a training/inference script reproduce them exactly.
    """
    import json

    from common.configs import GRIPPER_OPEN_MAX_FRAC, ROTATION_SCALE, TRANSLATION_SCALE
    from common.recording.features import EE_NAMES, STATE_NAMES

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": fps,
        "joint_state_names": list(STATE_NAMES),
        "joint_units": "urdf_degrees",
        "gripper_open_fraction": "0=closed, 1=capped-open",
        "gripper_open_max_frac": GRIPPER_OPEN_MAX_FRAC,
        "translation_scale": TRANSLATION_SCALE,
        "rotation_scale": ROTATION_SCALE,
        "ee_features": {
            "recorded": record_ee,
            "names": list(EE_NAMES),
            "frame": "each arm's own base frame",
            "quaternion_order": "wxyz",
            "state_key": "ee_pose",
            "target_key": "ee_target",
            "note": "ee_target is the projected+constrained IK target; an "
            "EE-space policy remaps ee_target -> action and replays it "
            "through the same IK + workspace envelope at inference.",
        },
    }
    (meta_dir / "action_space.json").write_text(json.dumps(payload, indent=2))
    print(f"  🗂  wrote {meta_dir / 'action_space.json'} (action-definition constants)")


def build_recording_stack(
    args: argparse.Namespace,
    data_manager: DualDataManager,
    quest_reader,
    park_arms,
    record_ee: bool = False,
):
    """Open cameras (fail fast), create/resume the dataset, build the recorder.

    Returns the EpisodeRecorder (not yet started). Exits the process if a
    required flag is missing, an enabled camera cannot open, or disk is full.
    """
    if not args.repo_id:
        raise SystemExit("❌ --record requires --repo-id")
    if not args.task:
        raise SystemExit("❌ --record requires --task")

    rec_cfg = load_recording_config()
    fps = args.dataset_fps or rec_cfg["dataset"]["fps"]
    sidecar_cfg = rec_cfg["sidecar"]
    streams = resolve_camera_streams(rec_cfg, args)
    if not streams:
        raise SystemExit("❌ --record with every camera disabled is not supported")

    # Prefer the stable by-path node for any stream assigned in sensor_map.yaml
    # (the wrist cameras especially), so the recorded device matches the live
    # view and survives a replug. Unassigned streams keep their yaml index. The
    # same map also supplies the central RealSense serial below.
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    sensor_map = load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
    if sensor_map:
        streams = overlay_sensor_map_devices(streams, sensor_map)

    # Open every enabled camera BEFORE creating the dataset: an unopenable
    # device at startup is a wiring problem, not a mid-session dropout.
    captures: list[CameraCapture] = []
    for name, cfg in streams.items():
        cam = CameraCapture(
            name=name,
            device=cfg["device"],
            width=cfg["width"],
            height=cfg["height"],
            fps=cfg["fps"],
            rotate180=cfg["rotate180"],
            fourcc=cfg["fourcc"],
            controls={k: cfg.get(k) for k in CONTROL_NAMES},
        )
        if not cam.open():
            for opened in captures:
                opened.stop()
            raise SystemExit(
                f"❌ Camera '{name}' failed to open on device {cfg['device']} "
                "— it is unplugged, in another socket, or was refused its share "
                "of the USB bandwidth.\n"
                f"   Recording {len(streams)} cameras at once: about three fit "
                "one USB 2.0 bus, so try --disable-camera on one sharing its "
                "controller (tool/collect_preflight.py names them), fix the "
                "device index in src/conf/recording.yaml, or "
                f"pass --disable-camera {name}"
            )
        captures.append(cam)

    # Optional central RealSense RGB-D: its colour stream joins ``captures`` as
    # a normal video feature; its aligned 16-bit depth is written separately.
    rs_cfg = rec_cfg.get("realsense")
    if args.central_depth and rs_cfg is None:
        for opened in captures:
            opened.stop()
        raise SystemExit(
            "❌ --central-depth but src/conf/recording.yaml has no 'realsense' "
            "section — add one (see the example) or drop the flag"
        )
    all_captures: list = list(captures)
    rs_capture = build_realsense_capture(rec_cfg, args, sensor_map, for_view=False)
    if rs_capture is not None:
        if not rs_capture.open():
            for opened in captures:
                opened.stop()
            raise SystemExit(
                "❌ RealSense failed to open — check the camera is connected "
                "and the assigned serial (tool/test_sensor_rates.py --assign) "
                "or drop --central-depth / set realsense.enabled: false"
            )
        all_captures.append(rs_capture)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(
        args.dataset_root if args.dataset_root else HF_LEROBOT_HOME / args.repo_id
    )
    check_disk_space(root)

    threads_total = rec_cfg["dataset"]["image_writer_threads_per_camera"] * len(
        all_captures
    )
    # Real-time (streaming) video encoding: each camera frame is encoded in a
    # background thread as it is recorded, so the A-to-save step only has to
    # flush the last queued frames instead of encoding the whole episode. This
    # keeps the pause between episodes short so collection stays continuous. On
    # a CPU-starved rig the encoder can fall behind — LeRobot then DROPS the
    # backlogged video frame with a warning (never crashes), so opt out with
    # --no-streaming-encode if that trade-off is not wanted.
    streaming = not args.no_streaming_encode
    if args.resume:
        print(f"📂 Resuming dataset {args.repo_id} at {root}")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=root,
            image_writer_threads=threads_total,
            streaming_encoding=streaming,
            encoder_queue_maxsize=_ENCODER_QUEUE_MAXSIZE,
        )
    else:
        if root.exists():
            for opened in all_captures:
                opened.stop()
            raise SystemExit(
                f"❌ {root} already exists — pass --resume to append or "
                "choose another --repo-id/--dataset-root"
            )
        features = build_dataset_features(
            [(c.name, c.height, c.width) for c in all_captures],
            include_phase=True,
            include_ee=record_ee,
        )
        print(f"📂 Creating dataset {args.repo_id} at {root} ({fps} fps)")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=rec_cfg["dataset"]["robot_type"],
            image_writer_threads=threads_total,
            streaming_encoding=streaming,
            encoder_queue_maxsize=_ENCODER_QUEUE_MAXSIZE,
        )

    # Persist the RealSense intrinsics + depth scale once per dataset so the
    # depth stream is interpretable and the setup is replicable. The camera's
    # extrinsic pose in the rig frame is a physical measurement recorded
    # separately (left as a null placeholder for the operator to fill).
    depth_streams: list[str] = []
    depth_writer = None
    if rs_capture is not None:
        write_realsense_meta(dataset.root, rs_capture, rs_cfg)
        depth_streams = [rs_capture.depth_name]
        depth_writer = DepthWriter(dataset.root, depth_streams)
    if not args.resume:
        write_action_space_meta(dataset.root, fps, record_ee)

    sidecar = None
    if sidecar_cfg["enabled"] and not args.no_sidecar:
        sidecar = SidecarSampler(
            data_manager=data_manager,
            quest_reader=quest_reader,
            root=dataset.root,
            rate_hz=sidecar_cfg["rate_hz"],
            include_hw_frame_goal=sidecar_cfg["include_hw_frame_goal"],
        )

    # Audible start/stop cue (optional 'audio' section; --no-sound disables).
    audio_cfg = rec_cfg.get("audio") or {}
    audio_cue = None
    if audio_cfg.get("enabled", False) and not args.no_sound:
        from common.recording.audio_cue import AudioCue

        audio_cue = AudioCue(
            enabled=True,
            start_sound=audio_cfg.get("start_sound", ""),
            stop_sound=audio_cfg.get("stop_sound", ""),
        )

    return EpisodeRecorder(
        dataset=dataset,
        data_manager=data_manager,
        task=args.task,
        fps=fps,
        camera_names=[c.name for c in all_captures],
        cameras=all_captures,
        sidecar=sidecar,
        park_arms=park_arms,
        depth_streams=depth_streams,
        depth_writer=depth_writer,
        record_ee=record_ee,
        audio_cue=audio_cue,
    )


def add_sensor_view_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("live sensor view")
    group.add_argument(
        "--sensor-view",
        action="store_true",
        help="Live window with camera feeds + both arms' joint state while "
        "teleoperating; with --record it mirrors the recorded streams (scene, "
        "wrist cameras, tactile, central RGB) with per-stream drift, otherwise "
        "the sensor_map cameras (q/Esc closes just the window)",
    )
    group.add_argument(
        "--view-camera",
        action="append",
        default=[],
        metavar="NAME=DEV",
        help="Ad-hoc camera for --sensor-view, e.g. "
        "left_arm_left_gripper=/dev/video4 (repeatable; default: the "
        "cameras assigned in src/conf/sensor_map.yaml)",
    )
    group.add_argument(
        "--monitor-port",
        type=int,
        default=0,
        metavar="PORT",
        help="Serve the live view on this loopback port so the rig console "
        "(tool/rig_web.py) can show the cameras and the recorder state in a "
        "browser, and press A/Q remotely (0 = off, the default). No device is "
        "opened for it: the frames are the ones this session already has",
    )


def build_sensor_view_captures(
    args: argparse.Namespace, data_manager: DualDataManager
) -> list:
    """Open + start the viewer's camera capture threads (UVC + central RGB-D).

    The view is a convenience: a camera that fails to open is warned about and
    skipped (joints-only view if none open) — it never kills teleoperation. The
    central RealSense colour stream is added when it is enabled, requested with
    --central-depth, or assigned in sensor_map, so an assigned central camera
    shows here without --record. Only NO camera being configured at all is an
    error.
    """
    from common.config_parser import load_recording_config
    from tool.test_sensor_rates import (
        SENSOR_MAP_PATH,
        load_sensor_map,
        parse_camera_spec,
    )

    sensor_map = load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
    if args.view_camera:
        specs = [parse_camera_spec(spec) for spec in args.view_camera]
    else:
        specs = sorted((sensor_map.get("cameras") or {}).items())

    captures: list = []
    for name, device in specs:
        cam = CameraCapture(
            name=name,
            device=device,
            width=640,
            height=480,
            fps=30,  # the tactile cameras' true ceiling (measured)
            rotate180=False,
            fourcc="MJPG",  # compressed: 4 simultaneous streams share USB
        )
        if not cam.open():
            print(
                f"⚠️  sensor-view camera '{name}' ({device}) failed to open — skipped"
            )
            continue
        cam.start(data_manager)
        captures.append(cam)

    # Central RealSense colour stream (opens its own pipeline; only reached
    # WITHOUT --record — the record path reuses the recorder's capture instead,
    # so the single pipeline is never opened twice).
    rs_capture = build_realsense_capture(
        load_recording_config(), args, sensor_map, for_view=True
    )
    if rs_capture is not None:
        if rs_capture.open():
            rs_capture.start(data_manager)
            captures.append(rs_capture)
        else:
            print("⚠️  sensor-view: central RealSense failed to open — skipped")

    if not captures and not args.view_camera and not sensor_map:
        raise SystemExit(
            "❌ --sensor-view has no cameras: run "
            "tool/test_sensor_rates.py --assign once, or pass "
            "--view-camera NAME=DEV"
        )
    if not captures:
        print("⚠️  no sensor-view camera opened — showing joint panels only")
    return captures


def connect_leader_arms() -> dict:
    """Connect the two SO-101 leader arms for ``--input leader``.

    Each leader's PORT comes from ``src/conf/sensor_map.yaml`` (assigned
    with ``tool/test_sensor_rates.py --assign``), which is the single
    source of truth for which physical leader is which. Its calibration
    ID is fixed by side in ``robot.yaml`` (``LEADER_ID_LEFT/RIGHT``), so
    assignment no longer stores it. ``connect(calibrate=False)`` loads
    each leader's existing calibration without the interactive stdin
    prompt. Torque is left off (the leaders stay back-drivable).

    Returns ``{"left": SOLeader, "right": SOLeader}``.
    """
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    from common.follower_bus import leader_calib_id_for_side
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    hint = "run tool/test_sensor_rates.py --assign and assign leader right/left"
    if not SENSOR_MAP_PATH.exists():
        raise SystemExit(f"❌ --input leader needs leader arms assigned — {hint}")
    leaders_map = load_sensor_map(SENSOR_MAP_PATH)["leaders"]
    if not leaders_map:
        raise SystemExit(f"❌ no leader arms in sensor_map.yaml — {hint}")

    leaders: dict = {}
    for side in ("left", "right"):
        entry = leaders_map.get(side) or {}
        port = entry.get("port")
        if not port:
            raise SystemExit(
                f"❌ leader {side} not assigned in sensor_map.yaml — {hint}"
            )
        calib_id = leader_calib_id_for_side(side)
        cfg = SOLeaderTeleopConfig(port=port, id=calib_id, use_degrees=True)
        leader = SOLeader(cfg)
        print(f"🕹️  connecting {side} leader ({calib_id}) on {port} ...")
        leader.connect(calibrate=False)
        print(f"  ✓ {side} leader connected (torque off, back-drivable)")
        leaders[side] = leader
    return leaders


def apply_follower_ports_from_sensor_map(robot_conf: dict, required: bool) -> None:
    """Bind each follower bus to the physical arm the wiggle test assigned.

    robot.yaml's PORT_ID_0/1 are ``/dev/ttyACM*`` names, which reorder
    across replugs — and with the two leaders also plugged in they can
    resolve to a LEADER's device, so the follower bus and the leader bus
    fight over one port ("multiple access on port").

    Two conventions must be reconciled:
      * this tool wires bus_0 = PORT_ID_0/ROBOT_NAME_0 to side "left" and
        bus_1 = PORT_ID_1/ROBOT_NAME_1 to side "right";
      * the sensor map (and ``connect_follower_bus``) pairs side "right"
        with follower_0's calibration and side "left" with follower_1's.

    So for teleop side "left" to be the PHYSICAL left arm with the right
    calibration, bus_0 must take the map's ``left`` port together with
    follower_1's calibration — i.e. the ROBOT_NAME_* pair is swapped
    alongside the ports. This makes the Quest/leader left↔right and the
    sensor-view labels all agree with the physical arms, and keeps each
    bus paired with its own calibration.

    ``required`` (leader mode) makes missing follower assignments fatal;
    otherwise the robot.yaml default is left in place.
    """
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    hint = "run tool/test_sensor_rates.py --assign and assign follower right/left"
    arms = load_sensor_map(SENSOR_MAP_PATH)["arms"] if SENSOR_MAP_PATH.exists() else {}
    # Calibration that the sensor convention pairs with each side.
    calib_for = {
        "right": robot_conf.get("ROBOT_NAME_0"),  # follower_0
        "left": robot_conf.get("ROBOT_NAME_1"),  # follower_1
    }
    plan = {
        "left": ("PORT_ID_0", "ROBOT_NAME_0"),
        "right": ("PORT_ID_1", "ROBOT_NAME_1"),
    }
    for side, (port_key, name_key) in plan.items():
        port = arms.get(side)
        if port:
            robot_conf[port_key] = port
            robot_conf[name_key] = calib_for[side]
            print(
                f"🔌 follower {side} → {port} "
                f"(calibration {calib_for[side]}) from sensor_map.yaml"
            )
        elif required:
            raise SystemExit(
                f"❌ follower {side} not assigned in sensor_map.yaml — {hint}"
            )


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--input",
        type=str,
        default="quest",
        choices=["quest", "leader"],
        help="Operator interface: 'quest' (Meta Quest, default) or 'leader' "
        "(two SO-101 leader arms, direct joint-to-joint, no clutch; "
        "keyboard Y=enable+follow, B=home, X=park, A=episode, Q=quit; "
        "ports/ids from src/conf/sensor_map.yaml via test_sensor_rates.py "
        "--assign)",
    )
    parser.add_argument(
        "--assign",
        action="store_true",
        help="Run the sensor-assignment GUI first (reassign follower/leader "
        "arms and cameras to their names in src/conf/sensor_map.yaml), then "
        "continue into teleop/collection with the fresh assignments",
    )
    add_teleop_cli_args(
        parser, default_max_joint_vel=MAX_JOINT_VEL_HW_RAD_S, default_method="armplane"
    )
    add_recording_cli_args(parser)
    add_sensor_view_cli_args(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOPERATION (LeRobot Backend)")
    print("=" * 60)

    # Optional: reassign robot + sensor streams to their names BEFORE any
    # hardware is opened, so the connection below reads the fresh sensor_map.
    if args.assign:
        from tool.test_sensor_rates import (
            SENSOR_MAP_PATH,
            load_sensor_map,
            run_assignment,
            save_sensor_map,
        )

        current = (
            load_sensor_map(SENSOR_MAP_PATH)
            if SENSOR_MAP_PATH.exists()
            else {
                "cameras": {},
                "arms": {},
                "leaders": {},
                "realsense": {},
            }
        )
        updated = run_assignment(current)
        save_sensor_map(SENSOR_MAP_PATH, updated)
        print(f"✓ sensor assignments written to {SENSOR_MAP_PATH}")

    # 1. Shared state
    data_manager = DualDataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    # 2. LeRobot dual arm hardware (arm 0 = left, arm 1 = right)
    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
        "ready_pos": load_yaml(_root / "src/conf/ready_pos.yaml"),
    }
    use_leader = args.input == "leader"
    # In leader mode all four arms are plugged in, so the followers MUST use
    # their stable sensor_map ports (robot.yaml's ttyACM names can collide
    # with a leader's device). Quest mode is left on robot.yaml unchanged.
    if use_leader:
        apply_follower_ports_from_sensor_map(config["robot"], required=True)
    dual_arm = SO101DualArm(config)
    ready_pos = config["ready_pos"]
    rest_pos = config["rest_pos"]

    # 3. Input layer.
    # Quest: IK stack (10 body DOF, grippers locked) built by the shared
    # helper so this tool and the sim rehearsal (tool/quest_sim_teleop.py)
    # cannot drift, plus the Quest reader (the IK thread reads it directly).
    # Leader: two passive SO-101 leader arms — connected NOW, while the
    # terminal is still cooked (first-connect calibration prompts on stdin);
    # no IK solver, the leader thread publishes joint targets directly.
    quest_reader = None
    ik_solver = thread_kwargs = None
    leaders = None
    if use_leader:
        leaders = connect_leader_arms()
    else:
        # 'mymethod' reuses the pink_relaxed solver plus the thumbstick wrist
        # trims (--wrist-mode); armplane keeps the tuned Pink solver +
        # armplane mapping.
        ik_solver, thread_kwargs = create_teleop_stack(args, dt=1.0 / IK_SOLVER_RATE)
        print("\n🎮 Initializing Meta Quest reader...")
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    # 5. Threads: per-arm joint state I/O + the input thread (dual IK solver
    # for Quest, leader poller for leader mode — each leader owns its own
    # serial port, so no lock is shared with the follower buses).
    # The Feetech serial port handler is not thread-safe: every bus access
    # (joint threads AND button callbacks) must hold that bus's lock.
    left_bus_lock = threading.Lock()
    right_bus_lock = threading.Lock()
    # Gripper range: the Quest cap for the headset, the FULL jaw range for
    # leader teleoperation (the leader jaw itself is the operator's control).
    gripper_cap = 1.0 if use_leader else GRIPPER_OPEN_MAX_FRAC
    left_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_0, "left", left_bus_lock),
        kwargs={"gripper_open_max_frac": gripper_cap},
        daemon=True,
    )
    right_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_1, "right", right_bus_lock),
        kwargs={"gripper_open_max_frac": gripper_cap},
        daemon=True,
    )
    if use_leader:
        input_thread = threading.Thread(
            target=leader_arm_thread,
            args=(data_manager, leaders),
            kwargs={"gripper_open_max_frac": gripper_cap},
            daemon=True,
        )
    else:
        input_thread = threading.Thread(
            target=dual_ik_solver_thread,
            args=(data_manager, ik_solver, quest_reader),
            kwargs=thread_kwargs,
            daemon=True,
        )
    left_joint_thread.start()
    right_joint_thread.start()
    input_thread.start()

    # park_arms: the recorder invokes this ONLY for shutdown/thread-error
    # discards (never for DISABLED- or camera-staleness-triggered ones). Bare
    # torque-off fallback if the interpolated move itself fails.
    def park_arms() -> None:
        print("🅿️  Parking arms (rest pose, torque off)...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False)
        try:
            with left_bus_lock, right_bus_lock:
                dual_arm.move_to_joint_pose(rest_pos, rest_pos, 2.0)
                dual_arm.bus_0.disable_torque()
                dual_arm.bus_1.disable_torque()
        except Exception as e:
            if bus_gone(e) or data_manager.is_bus_lost():
                # Nothing can be commanded over a bus that is not there, and the
                # retry below would fail the same way. What the operator needs is
                # the consequence, not the stack: the arms are still holding.
                print(
                    "⚠️  the arms' serial bus went away — they could not be "
                    "parked and their torque is still ON. Power-cycle them (or "
                    "re-run test/send_middle_and_rest.py once the bus is back)."
                )
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
                return
            traceback.print_exc()
            try:
                with left_bus_lock, right_bus_lock:
                    dual_arm.disable_torque()
            except Exception as e2:
                if bus_gone(e2) or data_manager.is_bus_lost():
                    print(
                        "⚠️  the arms' serial bus went away — torque could not "
                        "be disabled; power-cycle the arms."
                    )
                else:
                    traceback.print_exc()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        print("✓ 🅿️  Both arms parked and disabled (torque off)")

    # 5b. Recording stack (only with --record). Cameras are opened (fail-fast)
    # BEFORE the dataset is created; the recorder thread owns the writer.
    recorder: EpisodeRecorder | None = None
    if args.record:
        # EE-space features need the IK thread's targets/poses, which only run
        # in quest mode — auto-disable them under --input leader.
        record_ee = (not use_leader) and (not args.no_record_ee)
        if use_leader and not args.no_record_ee:
            print(
                "ℹ️  leader mode: recording joint-space only (EE targets need "
                "the IK thread, which does not run for --input leader)"
            )
        recorder = build_recording_stack(
            args, data_manager, quest_reader, park_arms, record_ee=record_ee
        )
        recorder.start()

    def _move_to_ready() -> None:
        """HOMING → interpolate both arms to the ready pose → ENABLED (teleop off).

        Leader mode has no separate ready pose — the rest pose IS the ready
        pose — so both arms interpolate to rest_pos there instead of ready_pos.
        Its wrist_roll is dropped, though: ``move_to_joint_pose`` writes the pose
        straight to the motors, so rest_pos/ready_pos are MOTOR-space degrees,
        whereas leader-follow commands in URDF space (urdf = hw + 90 for
        wrist_roll). A neutrally-held leader parks the motor near hw −90, so
        homing to rest_pos's wrist_roll (motor ≈ 0) would roll the wrist ~90°
        and it would snap back the instant tracking resumes (ready_pos's 90 made
        it ~180°). Omitting wrist_roll from the home pose leaves the motor
        un-commanded — it holds its present position — and leader-follow then
        continues it, so pressing A/B causes no wrist roll.
        """
        home = rest_pos if use_leader else ready_pos
        if use_leader:
            home = {k: v for k, v in home.items() if k != "wrist_roll"}
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False)
        with left_bus_lock, right_bus_lock:
            dual_arm.move_to_joint_pose(home, home, 2.0)
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)

    def _start_leader_tracking() -> None:
        """Leader mode: begin direct joint-to-joint follow (no clutch).

        Seed the target from the followers' measured pose BEFORE enabling
        teleop so the joint threads can't replay a stale target; the
        leader thread then slews from there toward the leader pose.
        """
        measured = data_manager.get_current_joint_angles()
        if measured is not None:
            data_manager.set_target_joint_angles(measured)
        data_manager.set_teleop_state(True)
        print("🕹️  Leader follow active — followers mirror the leaders")

    # 6. Quest button callbacks.
    # MUST be crash-proof: the quest reader dispatches callbacks without an
    # except clause, so a raised exception kills its thread (no more buttons
    # OR hand tracking), and a crash mid-move would leave the state stuck in
    # HOMING, making the buttons silently dead.
    def _safe_button(name, fn):
        def wrapped() -> None:
            print(
                f"[{name}] pressed "
                f"(state={data_manager.get_robot_activity_state().value})"
            )
            try:
                fn()
            except Exception:
                traceback.print_exc()
                print(
                    f"❌ [{name}] handler failed (see traceback above). "
                    "Torque off, state reset to DISABLED — press Y to retry."
                )
                try:
                    with left_bus_lock, right_bus_lock:
                        dual_arm.disable_torque()
                except Exception:
                    traceback.print_exc()
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

        return wrapped

    def on_enable() -> None:
        """Y: enable torque. Only from DISABLED.

        Quest mode moves the followers to the ready pose first — the defined
        start pose behind the clutch. Leader mode has no separate ready pose
        (the rest pose IS the ready pose), so Y just turns torque on and hands
        control straight to the leaders: the leader thread then slews the
        followers from their current pose toward the leader pose at a bounded
        velocity, so no homing sweep is needed.
        """
        if data_manager.get_robot_activity_state() != RobotActivityState.DISABLED:
            print("⚠️  Y ignored: arms are not DISABLED")
            return
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        if use_leader:
            print("🟢 Enabling: torque on, handing control to the leaders...")
            with left_bus_lock, right_bus_lock:
                dual_arm.bus_0.enable_torque()
                dual_arm.bus_1.enable_torque()
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            _start_leader_tracking()
            return
        print("🟢 Enabling: moving both arms to ready pose...")
        with left_bus_lock, right_bus_lock:
            dual_arm.bus_0.enable_torque()
            dual_arm.bus_1.enable_torque()
            dual_arm.move_to_joint_pose(ready_pos, ready_pos, 2.0)
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        print("✓ 🟢 Both arms at ready pose and enabled")

    def on_park() -> None:
        """X: move to rest + torque off. Only when ENABLED and not recording."""
        if data_manager.get_robot_activity_state() != RobotActivityState.ENABLED:
            print("⚠️  X ignored: arms are not ENABLED")
            return
        if recorder is not None and recorder.get_state() != RecorderState.IDLE:
            print(
                f"⚠️  X ignored: recorder is {recorder.get_state().value} — "
                "stop the episode first (A)"
            )
            return
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False)
        print("🔴 Parking: moving both arms to rest pose...")
        with left_bus_lock, right_bus_lock:
            dual_arm.move_to_joint_pose(rest_pos, rest_pos, 2.0)
            dual_arm.bus_0.disable_torque()
            dual_arm.bus_1.disable_torque()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        print("✓ 🔴 Both arms at rest and disabled (torque off)")

    def on_episode_toggle() -> None:
        """A: start/stop-save an episode (requires ENABLED and --record).

        Leader mode records in place: enabling (Y) already hands control to the
        leaders, and the followers keep tracking them for the whole session, so
        A only toggles the recorder — it never homes the arms. Quest mode still
        bookends each episode with a move to the ready pose (recorded), the
        defined start/end frame behind the clutch.
        """
        if recorder is None:
            print("⚠️  A ignored: started without --record")
            return
        if data_manager.get_robot_activity_state() != RobotActivityState.ENABLED:
            print("⚠️  A ignored: arms are not ENABLED (press Y first)")
            return
        state = recorder.get_state()
        if state == RecorderState.IDLE:
            if use_leader:
                print("🔴 Recording — the leaders keep driving the followers")
            else:
                print("🏁 Moving to ready, then starting the episode...")
                _move_to_ready()
            recorder.request_start_episode()
        elif state in (RecorderState.RECORDING, RecorderState.PAUSED):
            # PAUSED still holds an open episode (a camera is briefly away), so
            # stop means stop: keep the frames captured up to here.
            if use_leader:
                print("💾 Stopping and saving the episode...")
            else:
                # The ready-move stays inside the episode (still recording).
                print("🏁 Moving to ready (recorded), then stopping and saving...")
                _move_to_ready()
            recorder.request_stop_save()
        else:
            print(f"⚠️  A ignored: recorder is busy ({state.value})")

    def on_go_home() -> None:
        """B: move to the ready pose; the recorder is never touched."""
        state = data_manager.get_robot_activity_state()
        if state in (RobotActivityState.ENABLED, RobotActivityState.HOMING):
            print("🏠 Moving both arms to ready pose...")
            _move_to_ready()
            if use_leader:
                _start_leader_tracking()
            print("✓ Both arms at ready pose and re-enabled")
        else:
            print("⚠️  Cannot home: arms not enabled")

    # Control surface, one table for every way a button can be pressed. Quest:
    # the headset buttons bind to these closures. Leader: the same callbacks
    # keyed to characters, dispatched either by the sensor-view window (if
    # --sensor-view) or a terminal keyboard reader; leader-follower has NO
    # clutch — enabling (Y) starts the direct joint-to-joint follow, and there
    # is no engage/pause key. With --monitor-port the live monitor presses the
    # two entries it is allowed to, which is why the table is now built in both
    # modes rather than only for the leader.
    button_actions: dict = {
        "y": _safe_button("Y (enable)", on_enable),
        "x": _safe_button("X (park)", on_park),
        "a": _safe_button("A (episode)", on_episode_toggle),
        "b": _safe_button("B (home)", on_go_home),
        "q": _safe_button("Q (quit)", data_manager.request_shutdown),
    }
    leader_keys: dict = button_actions if use_leader else {}
    if not use_leader:
        quest_reader.on("button_y_pressed", button_actions["y"])
        quest_reader.on("button_x_pressed", button_actions["x"])
        quest_reader.on("button_a_pressed", button_actions["a"])
        quest_reader.on("button_b_pressed", button_actions["b"])
        quest_reader.on(
            "button_lj_pressed",
            _safe_button(
                "Left joystick click",
                lambda: data_manager.request_roll_reset("left"),
            ),
        )
        quest_reader.on(
            "button_rj_pressed",
            _safe_button(
                "Right joystick click",
                lambda: data_manager.request_roll_reset("right"),
            ),
        )

    print()
    print("🚀 Dual-arm teleoperation ready.")
    # The same list the rig console shows beside its live view
    # (common.recording.controls), so the terminal and the browser cannot drift
    # apart on how this session is driven.
    if use_leader:
        surface = "the sensor-view window" if args.sensor_view else "this terminal"
        print(f"   Leader-follower mode — type the keys into {surface}:")
    for step in control_steps(
        "leader" if use_leader else "quest", record=args.record, method=args.method
    ):
        lead = f"{step['key']} = " if step["key"] else ""
        print(f"   {lead}{step['what']}")
    if args.sensor_view and not use_leader:
        print("   👁 --sensor-view window: q/Esc closes it (teleop keeps running)")
    print("⚠️  Press Ctrl+C to exit")
    print()

    view_captures: list = []
    owned_view_captures: list = []
    if args.sensor_view:
        if recorder is not None and recorder.cameras:
            # Reuse the recorder's already-open captures: they publish RGB into
            # the DataManager, so the view shows scene + both wrist cameras +
            # tactile + central RGB with drift WITHOUT opening the same by-path
            # device a second time. The recorder owns their lifecycle.
            view_captures = list(recorder.cameras)
        else:
            owned_view_captures = build_sensor_view_captures(args, data_manager)
            view_captures = owned_view_captures

    # Footer for the view: the control-key hints plus, when recording, a live
    # RECORDING/IDLE + episodes-to-goal readout, so the operator drives the
    # session from the window without watching the terminal.
    view_key_help: list[str] | None = None
    collection_status: "Callable[[], CollectionStatus] | None" = None
    if recorder is not None:
        _rec = recorder
        _goal = args.episode_goal

        def _collection_status() -> CollectionStatus:
            st = _rec.get_state()
            return CollectionStatus(
                state_label=st.value,
                episodes_done=_rec.get_episode_count(),
                episodes_goal=_goal,
                current_frames=_rec.get_current_frame_count(),
                # PAUSED keeps the badge lit: the episode is still open, and
                # the label itself tells the operator a stream is missing.
                recording=st in (RecorderState.RECORDING, RecorderState.PAUSED),
            )

        collection_status = _collection_status
    if args.sensor_view:
        rec_hint = "A record" if args.record else "A —"
        if use_leader:
            view_key_help = ["Y enable", rec_hint, "B home", "X park", "Q quit"]
        else:
            view_key_help = [
                "btn Y enable",
                f"btn {rec_hint}",
                "btn B home",
                "btn X park",
            ]
    view_status_provider = collection_status if args.sensor_view else None

    # The browser live view. It opens NO device: it serves the frames this
    # session already publishes, so it costs a JPEG encode per viewer and can
    # run beside the window, or instead of it on a rig with no display.
    monitor: MonitorServer | None = None
    if args.monitor_port:
        monitor_captures = view_captures or (
            list(recorder.cameras) if recorder is not None else []
        )
        monitor = MonitorServer(
            data_manager,
            captures=monitor_captures,
            status_provider=collection_status,
            key_callbacks=button_actions,
            # What a watcher may press depends on where the operator is: with a
            # headset on, every button is to hand and only the two that move
            # nothing are remote. Leader arms leave the page as the only surface
            # a console-started session has, so enabling is remote there too.
            allowed_keys=allowed_keys_for("leader" if use_leader else "quest"),
            port=args.monitor_port,
        )
        monitor.start()
    keyboard: KeyboardButtons | None = None

    try:
        if args.sensor_view:
            # The view loop runs on the main thread (sole owner of the cv2
            # GUI). In leader mode the window is also the control surface:
            # it dispatches the Y/X/A/B/Q keys and stays open until quit.
            run_sensor_view_loop(
                data_manager,
                view_captures,
                leader_mode=use_leader,
                key_callbacks=leader_keys if use_leader else None,
                key_help=view_key_help,
                status_provider=view_status_provider,
            )
        elif use_leader:
            # No window: read the control keys from the terminal, if there is
            # one. Started by the rig console there is not -- stdin is whatever
            # the console inherited -- and that is not a reason to end a session
            # the console can drive itself; it is a reason to say where the
            # keys have to come from instead.
            keyboard = KeyboardButtons()
            for key, cb in leader_keys.items():
                keyboard.on(key, cb)
            try:
                keyboard.start()
            except (termios.error, ValueError, OSError) as exc:
                keyboard = None
                print(
                    f"⌨️  no terminal to read the control keys from ({exc}). "
                    "Drive this session from the rig console instead: it can "
                    "enable the arms, record an episode and end the session."
                )
        while not data_manager.is_shutdown_requested():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received — shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        # A second Ctrl+C here must NOT abort shutdown. The teardown below (park
        # torque off, stop cameras/RealSense, let an in-flight episode finish
        # saving) ends in the os._exit(0) at the bottom, which deliberately
        # dodges the librealsense std::terminate -> SIGABRT. If an interrupt is
        # allowed to escape mid-teardown it skips that exit, so CPython
        # force-unwinds the RealSense C++ thread (core dump) AND the in-flight
        # save is lost. Ignore SIGINT/SIGTERM so shutdown always runs to _exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("\n🧹 Cleaning up... (interrupt ignored until shutdown completes)")
        data_manager.request_shutdown()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        # Recorder first (discards any in-flight episode, finalizes the
        # dataset) while the input reader is still alive.
        if recorder is not None:
            recorder.shutdown()
        if quest_reader is not None:
            quest_reader.stop()
        if monitor is not None:
            monitor.stop()
        if keyboard is not None:
            keyboard.stop()  # restores the terminal
        input_thread.join(timeout=3.0)
        left_joint_thread.join(timeout=3.0)
        right_joint_thread.join(timeout=3.0)
        if leaders is not None:
            for leader in leaders.values():
                try:
                    leader.disconnect()
                except Exception as e:
                    if bus_gone(e) or data_manager.is_bus_lost():
                        print("   leader bus already gone; nothing to close")
                    else:
                        traceback.print_exc()
        for cam in owned_view_captures:
            cam.stop()
        if data_manager.is_bus_lost():
            print(
                "⚠️  a motor bus went away during this session — torque could "
                "not be disabled on it; power-cycle the arms."
            )
        else:
            with left_bus_lock, right_bus_lock:
                dual_arm.disable_torque()
        print("👋 Done.")
        # Hard-exit to skip interpreter finalisation. pyrealsense2 keeps an
        # internal C++ context thread that pipeline.stop() does not fully join;
        # at normal shutdown CPython force-unwinds it via pthread_exit, which
        # raises a forced-unwind exception across librealsense's non-unwindable
        # callback boundary -> std::terminate -> SIGABRT. On Ubuntu apport then
        # dumps a ~500 MB core to /var/crash on EVERY run. All our own teardown
        # (torque off, cameras/pipeline stopped, leaders disconnected) has
        # already run above, so exiting now is clean for us and dodges the abort.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
