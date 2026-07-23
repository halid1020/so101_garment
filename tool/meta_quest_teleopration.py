#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest (LeRobot backend).

Left Meta Quest hand → left SO101 arm (arm 0, PORT_ID_0).
Right Meta Quest hand → right SO101 arm (arm 1, PORT_ID_1).
Single 10-DOF IK solver on the dual-arm URDF.

Optionally records LeRobot-format episodes (--record): a training-ready
dataset at the configured fps plus a ~100 Hz full-rate sidecar parquet per
episode, all controlled from the Quest handles.

--sensor-view opens a live window with the tactile-camera feeds and both
arms' measured/commanded joints while you teleoperate (cameras come from
src/conf/sensor_map.yaml — run tool/test_sensor_rates.py --assign once —
or ad-hoc --view-camera NAME=DEV). q/Esc closes just the window.

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
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import yaml
from meta_quest_teleop.reader import MetaQuestReader

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    IK_SOLVER_RATE,
    MAX_JOINT_VEL_HW_RAD_S,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.recording import (
    CameraCapture,
    EpisodeRecorder,
    RecorderState,
    SidecarSampler,
    build_dataset_features,
    load_recording_config,
)
from common.sensor_view import run_sensor_view_loop
from common.teleop_setup import add_teleop_cli_args, create_teleop_stack
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from src.so101_dual_arm import SO101DualArm

# Disk-space thresholds for --record (GB free on the dataset volume).
_DISK_WARN_GB = 10.0
_DISK_REFUSE_GB = 2.0


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
        help="Enable the tactile_0..3 camera streams (hardware required)",
    )
    group.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Disable the ~100 Hz full-rate sidecar parquet",
    )


def resolve_camera_streams(rec_cfg: dict, args: argparse.Namespace) -> dict:
    """Return {name: camera-config} for the streams enabled after overrides."""
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
        if args.tactile and name.startswith("tactile_"):
            on = True
        if name in args.enable_camera:
            on = True
        if name in args.disable_camera:
            on = False
        if on:
            enabled[name] = cfg
    return enabled


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


def build_recording_stack(
    args: argparse.Namespace,
    data_manager: DualDataManager,
    quest_reader,
    park_arms,
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
        )
        if not cam.open():
            for opened in captures:
                opened.stop()
            raise SystemExit(
                f"❌ Camera '{name}' failed to open on device {cfg['device']} "
                "— fix the device index in src/conf/recording.yaml or pass "
                f"--disable-camera {name}"
            )
        captures.append(cam)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(
        args.dataset_root if args.dataset_root else HF_LEROBOT_HOME / args.repo_id
    )
    check_disk_space(root)

    threads_total = rec_cfg["dataset"]["image_writer_threads_per_camera"] * len(
        captures
    )
    if args.resume:
        print(f"📂 Resuming dataset {args.repo_id} at {root}")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=root,
            image_writer_threads=threads_total,
        )
    else:
        if root.exists():
            for opened in captures:
                opened.stop()
            raise SystemExit(
                f"❌ {root} already exists — pass --resume to append or "
                "choose another --repo-id/--dataset-root"
            )
        features = build_dataset_features(
            [(c.name, c.height, c.width) for c in captures]
        )
        print(f"📂 Creating dataset {args.repo_id} at {root} ({fps} fps)")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=rec_cfg["dataset"]["robot_type"],
            image_writer_threads=threads_total,
        )

    sidecar = None
    if sidecar_cfg["enabled"] and not args.no_sidecar:
        sidecar = SidecarSampler(
            data_manager=data_manager,
            quest_reader=quest_reader,
            root=dataset.root,
            rate_hz=sidecar_cfg["rate_hz"],
            include_hw_frame_goal=sidecar_cfg["include_hw_frame_goal"],
        )

    return EpisodeRecorder(
        dataset=dataset,
        data_manager=data_manager,
        task=args.task,
        fps=fps,
        camera_names=[c.name for c in captures],
        cameras=captures,
        sidecar=sidecar,
        park_arms=park_arms,
    )


def add_sensor_view_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("live sensor view")
    group.add_argument(
        "--sensor-view",
        action="store_true",
        help="Live window with tactile-camera feeds + both arms' joint "
        "state while teleoperating (q/Esc closes just the window)",
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


def build_sensor_view_captures(
    args: argparse.Namespace, data_manager: DualDataManager
) -> list[CameraCapture]:
    """Open + start the viewer's tactile-camera capture threads.

    The view is a convenience: a camera that fails to open is warned
    about and skipped (joints-only view if none open) — it never kills
    teleoperation. Only NO cameras being configured at all is an error.
    """
    from tool.test_sensor_rates import (
        SENSOR_MAP_PATH,
        load_sensor_map,
        parse_camera_spec,
    )

    if args.view_camera:
        specs = [parse_camera_spec(spec) for spec in args.view_camera]
    elif SENSOR_MAP_PATH.exists():
        specs = sorted(load_sensor_map(SENSOR_MAP_PATH)["cameras"].items())
    else:
        raise SystemExit(
            "❌ --sensor-view has no cameras: run "
            "tool/test_sensor_rates.py --assign once, or pass "
            "--view-camera NAME=DEV"
        )

    captures: list[CameraCapture] = []
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
    if not captures:
        print("⚠️  no sensor-view camera opened — showing joint panels only")
    return captures


def main():
    parser = argparse.ArgumentParser(description="Dual-arm SO101 teleoperation")
    parser.add_argument("--ip-address", type=str, default=None)
    add_teleop_cli_args(
        parser, default_max_joint_vel=MAX_JOINT_VEL_HW_RAD_S, default_method="armplane"
    )
    add_recording_cli_args(parser)
    add_sensor_view_cli_args(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOPERATION (LeRobot Backend)")
    print("=" * 60)

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
    dual_arm = SO101DualArm(config)
    ready_pos = config["ready_pos"]
    rest_pos = config["rest_pos"]

    # 3. IK layer (10 body DOF, grippers locked): built by the shared helper so
    # this tool and the sim rehearsal (tool/quest_sim_teleop.py) cannot drift.
    # 'mymethod' reuses the pink_relaxed solver plus the thumbstick wrist trims
    # (--wrist-mode); armplane keeps the tuned Pink solver + armplane mapping.
    ik_solver, thread_kwargs = create_teleop_stack(args, dt=1.0 / IK_SOLVER_RATE)

    # 4. Quest reader (IK thread reads it directly)
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    # 5. Threads: per-arm joint state I/O + dual IK solver.
    # The Feetech serial port handler is not thread-safe: every bus access
    # (joint threads AND quest button callbacks) must hold that bus's lock.
    left_bus_lock = threading.Lock()
    right_bus_lock = threading.Lock()
    left_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_0, "left", left_bus_lock),
        daemon=True,
    )
    right_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_arm.bus_1, "right", right_bus_lock),
        daemon=True,
    )
    ik_thread = threading.Thread(
        target=dual_ik_solver_thread,
        args=(data_manager, ik_solver, quest_reader),
        kwargs=thread_kwargs,
        daemon=True,
    )
    left_joint_thread.start()
    right_joint_thread.start()
    ik_thread.start()

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
        except Exception:
            traceback.print_exc()
            try:
                with left_bus_lock, right_bus_lock:
                    dual_arm.disable_torque()
            except Exception:
                traceback.print_exc()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        print("✓ 🅿️  Both arms parked and disabled (torque off)")

    # 5b. Recording stack (only with --record). Cameras are opened (fail-fast)
    # BEFORE the dataset is created; the recorder thread owns the writer.
    recorder: EpisodeRecorder | None = None
    if args.record:
        recorder = build_recording_stack(args, data_manager, quest_reader, park_arms)
        recorder.start()

    def _move_to_ready() -> None:
        """HOMING → interpolate both arms to ready → ENABLED (teleop off)."""
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False)
        with left_bus_lock, right_bus_lock:
            dual_arm.move_to_joint_pose(ready_pos, ready_pos, 2.0)
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)

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
        """Y: torque on + move to ready. Only from DISABLED."""
        if data_manager.get_robot_activity_state() != RobotActivityState.DISABLED:
            print("⚠️  Y ignored: arms are not DISABLED")
            return
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
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
        """A: start/stop-save an episode (requires ENABLED and --record)."""
        if recorder is None:
            print("⚠️  A ignored: started without --record")
            return
        if data_manager.get_robot_activity_state() != RobotActivityState.ENABLED:
            print("⚠️  A ignored: arms are not ENABLED (press Y first)")
            return
        state = recorder.get_state()
        if state == RecorderState.IDLE:
            print("🏁 Moving to ready, then starting the episode...")
            _move_to_ready()
            recorder.request_start_episode()
        elif state == RecorderState.RECORDING:
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
            print("✓ Both arms at ready pose and re-enabled")
        else:
            print("⚠️  Cannot home: arms not enabled")

    quest_reader.on("button_y_pressed", _safe_button("Button Y", on_enable))
    quest_reader.on("button_x_pressed", _safe_button("Button X", on_park))
    quest_reader.on("button_a_pressed", _safe_button("Button A", on_episode_toggle))
    quest_reader.on("button_b_pressed", _safe_button("Button B", on_go_home))
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
    print("   1. Press BUTTON Y to enable both arms (ready pose, torque on)")
    print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
    print("   3. Move controllers — arms follow!")
    print("   4. Hold triggers to close grippers")
    if args.record:
        print("   5. Press BUTTON A to start an episode; press again to save")
    else:
        print("   5. BUTTON A records episodes (needs --record; warns otherwise)")
    print("   6. Press BUTTON B to move both arms to the ready pose")
    print("   7. Press BUTTON X to park (rest pose, torque off)")
    if args.method == "mymethod":
        print(
            "   8. Deflect a THUMBSTICK to trim that arm's wrist (x = roll, "
            "y = flex); its other joints freeze while deflected, then the "
            "handle resumes from the new pose on release"
        )
    if args.sensor_view:
        print("   👁 --sensor-view window: q/Esc closes it (teleop keeps running)")
    print("⚠️  Press Ctrl+C to exit")
    print()

    view_captures: list[CameraCapture] = []
    if args.sensor_view:
        view_captures = build_sensor_view_captures(args, data_manager)

    try:
        if args.sensor_view:
            # Runs on the main thread (sole owner of the cv2 GUI);
            # returns when the window is closed or shutdown is requested.
            run_sensor_view_loop(data_manager, view_captures)
        while not data_manager.is_shutdown_requested():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received — shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        data_manager.request_shutdown()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        # Recorder first (discards any in-flight episode, finalizes the
        # dataset) while the quest reader is still alive.
        if recorder is not None:
            recorder.shutdown()
        quest_reader.stop()
        ik_thread.join(timeout=3.0)
        left_joint_thread.join(timeout=3.0)
        right_joint_thread.join(timeout=3.0)
        for cam in view_captures:
            cam.stop()
        with left_bus_lock, right_bus_lock:
            dual_arm.disable_torque()
        print("👋 Done.")


if __name__ == "__main__":
    main()
