#!/usr/bin/env python3
"""Meta-Quest teleoperation of the dual SO-101 arms in MuJoCo simulation.

Runs the *armplane* Quest pipeline (the tuned production stack:
One-Euro filtering, grip clutch, handle-axis calibration, armplane
orientation mapping — the exact `dual_ik_solver_thread` used on the real
robot) but streams the joint
commands into the MuJoCo scene instead of the motor buses. Any of the
registered benchmark IK methods can be selected, so every method can be
rehearsed with the real headset in simulation before it ever moves the
real arms.

Usage:
    # real headset (Quest on the same network / USB, like the real tool)
    python tool/quest_sim_teleop.py --method pink_relaxed
    python tool/quest_sim_teleop.py --method scipy_ls --ip-address 192.168.0.42

    # no headset: scripted mock device draws circles (pipeline smoke test)
    python tool/quest_sim_teleop.py --method dls --mock --duration 15

Controls (real headset): hold BOTH grips to activate teleop (EVERY grip
recalibrates the handle axes and the operator control frame — point both
handles straight down when gripping), triggers close the grippers,
release grips to pause. Ctrl+C exits.
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.configs import WORKSPACE_OOB_MODE  # noqa: E402
from common.configs import (  # noqa: E402
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    IK_SOLVER_RATE,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState  # noqa: E402
from common.threads.dual_ik_solver import dual_ik_solver_thread  # noqa: E402
from common.workspace_envelope import OOE_POLICIES  # noqa: E402
from sim_benchmark.constants import CONTROL_RATE_HZ, SIDES  # noqa: E402
from sim_benchmark.method_adapter import MethodIKAdapter  # noqa: E402
from sim_benchmark.methods import METHODS  # noqa: E402
from sim_benchmark.mock_quest_device import MOCK_PATTERNS  # noqa: E402
from sim_benchmark.scene import DualArmSim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", type=str, default="pink_relaxed", choices=sorted(METHODS)
    )
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a scripted mock Quest device instead of real hardware",
    )
    parser.add_argument(
        "--headless", action="store_true", help="No viewer (CI / smoke tests)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Exit after this many seconds (0 = run until Ctrl+C)",
    )
    parser.add_argument("--max-joint-vel", type=float, default=3.0)
    parser.add_argument(
        "--oob-mode",
        type=str,
        default=WORKSPACE_OOB_MODE,
        choices=sorted(OOE_POLICIES),
        help="Out-of-envelope target policy (see common/workspace_envelope.py)",
    )
    parser.add_argument(
        "--mock-pattern",
        type=str,
        default="circle",
        choices=sorted(MOCK_PATTERNS),
        help="Motion pattern of the --mock device "
        "(circle = table circles, wrist = wrist oscillation, "
        "excursion = deliberately out-of-envelope strokes)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="twin",
        choices=["twin", "plain"],
        help="'twin' = full digital-twin rig (board, tower, cameras, ALL "
        "collisions enabled); 'plain' = bare benchmark scene (no contacts)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip the OpenCV camera-view windows (twin scene only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"QUEST → MUJOCO SIM TELEOPERATION  (method: {args.method})")
    print("=" * 60)

    data_manager = DualDataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    ik_solver = MethodIKAdapter(
        args.method,
        dt=1.0 / IK_SOLVER_RATE,
        max_joint_vel=args.max_joint_vel,
        initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
    )

    if args.mock:
        from sim_benchmark.mock_quest_device import MockQuestReader

        quest_reader = MockQuestReader(pattern=args.mock_pattern)
    else:
        from meta_quest_teleop.reader import MetaQuestReader

        print("🎮 Initializing Meta Quest reader...")
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    sim: Any
    if args.scene == "twin":
        try:
            from sim_twin.scene import TwinSim

            sim = TwinSim(all_collisions=True)
            print("🏗️  Twin rig scene (board + tower + cameras, ALL collisions)")
        except FileNotFoundError as e:
            print(f"⚠️  Twin assets missing ({e}) — falling back to plain scene")
            args.scene = "plain"
            sim = DualArmSim()
    else:
        sim = DualArmSim()
    q0 = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
    sim.reset(q0)
    data_manager.set_current_joint_angles(np.degrees(q0))
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)

    # The twin mounts the arms on the printed board (bases above/off the
    # IK model's z=0 plane). Joint streaming is frame-agnostic, but markers
    # and tracking errors compare IK-frame targets with twin-world
    # measurements — bridge with the per-side FK offset at neutral.
    ik_poses0 = ik_solver.get_current_end_effector_poses()
    world_offset = {}
    for side, frame in (("left", "left_eef_link"), ("right", "right_eef_link")):
        meas0, _ = sim.eef_pose(side)
        world_offset[side] = meas0 - ik_poses0[frame][:3, 3]
    if np.linalg.norm(world_offset["left"]) > 1e-6:
        print(
            f"📐 IK→twin world offset: L={np.round(world_offset['left'], 4)} "
            f"R={np.round(world_offset['right'], 4)}"
        )

    ik_thread = threading.Thread(
        target=dual_ik_solver_thread,
        args=(data_manager, ik_solver, quest_reader),
        kwargs={"oob_mode": args.oob_mode},
        daemon=True,
    )
    ik_thread.start()

    viewer_ctx = None
    if not args.headless:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)

    gripper_range = {side: sim.model.joint(f"{side}_gripper").range for side in SIDES}
    dt = 1.0 / CONTROL_RATE_HZ
    n_substeps = max(1, round(dt / sim.model.opt.timestep))
    start = time.time()
    track_errs: list[float] = []
    held_target = q0.copy()  # servo target held while teleop is inactive

    # Camera-view windows (twin scene renders its C310s offscreen).
    show_cameras = args.scene == "twin" and not args.headless and not args.no_cameras
    cv2 = None
    if show_cameras:
        try:
            import cv2  # type: ignore[no-redef]

            print("📷 Camera views: rgb_scene | rgb_wrist_left | rgb_wrist_right")
        except ImportError:
            print("⚠️  OpenCV not installed — camera windows disabled")
            show_cameras = False
    camera_names = ["rgb_scene", "rgb_wrist_left", "rgb_wrist_right"]
    camera_period_ticks = max(1, round(CONTROL_RATE_HZ / 15.0))  # ~15 fps
    tick_count = 0

    print("🤖 Sim streaming started (Ctrl+C to exit)")
    try:
        while True:
            tick_start = time.time()
            if args.duration and tick_start - start > args.duration:
                break
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break

            # Only stream IK targets while teleop is ACTIVE. When idle the
            # thread syncs its targets from *measured* joints; feeding those
            # back to the position servos lets gravity sag ratchet the arms
            # downward (the servos chase their own droop — observed ~13 cm
            # EE sink in 2 s). Holding the last active command instead keeps
            # the arms up and the teleop activation anchor honest.
            if data_manager.get_teleop_active():
                target_deg = data_manager.get_target_joint_angles()
                if target_deg is not None:
                    held_target = np.radians(target_deg)
            sim.set_arm_targets(held_target)

            # Triggers drive the gripper joints (contact-active in the twin).
            markers = {}
            for side in SIDES:
                _, _, trigger = data_manager.get_controller_state(side)
                lo, hi = gripper_range[side]
                sim.data.ctrl[sim.gripper_ctrl_idx[SIDES.index(side)]] = (
                    hi + (lo - hi) * trigger
                )
                pose = data_manager.get_target_pose(side)
                if pose is not None:
                    markers[side] = (pose[:3, 3] + world_offset[side], pose[:3, :3])
            if markers:
                sim.set_target_markers(markers)

            # Headset-center marker (published by the IK thread at grip).
            if hasattr(sim, "set_headset_marker"):
                headset = data_manager.get_frame_markers().get("headset_center")
                if headset is not None:
                    mean_offset = 0.5 * (world_offset["left"] + world_offset["right"])
                    sim.set_headset_marker(headset + mean_offset)

            sim.step(n_substeps)
            data_manager.set_current_joint_angles(np.degrees(sim.arm_q()))

            # Tracking diagnostics against the commanded EE targets.
            for side, (pos, _rot) in markers.items():
                meas, _ = sim.eef_pose(side)
                track_errs.append(float(np.linalg.norm(pos - meas)))

            tick_count += 1
            if (
                show_cameras
                and cv2 is not None
                and tick_count % camera_period_ticks == 0
            ):
                frames = [sim.render_camera(name) for name in camera_names]
                strip = np.hstack(frames)
                cv2.imshow("so101 twin cameras", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if viewer_ctx is not None:
                viewer_ctx.sync()
            sleep = dt - (time.time() - tick_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n⏹  Interrupted")
    finally:
        data_manager.request_shutdown()
        if viewer_ctx is not None:
            viewer_ctx.close()
        if show_cameras and cv2 is not None:
            cv2.destroyAllWindows()
        quest_reader.stop()
        ik_thread.join(timeout=2.0)
        if track_errs:
            errs = np.asarray(track_errs)
            print(
                f"📊 EE tracking error while teleop active: "
                f"mean {errs.mean()*1e3:.1f} mm, "
                f"p95 {np.percentile(errs, 95)*1e3:.1f} mm "
                f"({len(errs)} samples)"
            )
        print("✅ Sim teleop stopped")


if __name__ == "__main__":
    main()
