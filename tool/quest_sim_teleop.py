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
release grips to pause. With --method mymethod, deflecting a thumbstick
trims that arm's wrist (x = roll, y = flex) while freezing its other joints
and ignoring the handle for that arm; releasing resumes handle control from
the new pose. The viewer draws RGB orientation triads for each arm's
measured EE pose (opaque) and its commanded target (semi-transparent).
Ctrl+C exits.
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

from common.configs import (  # noqa: E402
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    IK_SOLVER_RATE,
    MAX_JOINT_VEL_SIM_RAD_S,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState  # noqa: E402
from common.teleop_setup import add_teleop_cli_args, create_teleop_stack  # noqa: E402
from common.threads.dual_ik_solver import dual_ik_solver_thread  # noqa: E402
from sim_benchmark.constants import CONTROL_RATE_HZ, SIDES  # noqa: E402
from sim_benchmark.mock_quest_device import MOCK_PATTERNS  # noqa: E402
from sim_benchmark.scene import DualArmSim  # noqa: E402

# Orientation-triad drawing (viewer only): x=red, y=green, z=blue.
_TRIAD_AXIS_RGB = (
    (1.0, 0.25, 0.25),  # x
    (0.25, 1.0, 0.25),  # y
    (0.35, 0.5, 1.0),  # z
)
_TRIAD_LEN = 0.06  # m, axis arrow length
_TRIAD_WIDTH = 0.004  # m, arrow shaft radius


def _add_triad(scn: Any, pos: np.ndarray, rot: np.ndarray, alpha: float) -> None:
    """Append 3 RGB axis arrows for a world pose to a viewer user_scn.

    ``alpha`` < 1 marks the semi-transparent target triad apart from the
    opaque measured one. No-op once the scene's geom buffer is full.
    """
    import mujoco

    for axis in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        rgba = np.array([*_TRIAD_AXIS_RGB[axis], alpha], dtype=np.float32)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba,
        )
        tip = pos + rot[:, axis] * _TRIAD_LEN
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, _TRIAD_WIDTH, pos, tip)
        scn.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Shared teleop args (method/wrist-mode/max-joint-vel/oob/orientation-cost/
    # envelope) — identical to the real tool so the rehearsal cannot drift. The
    # sim default method stays pink_relaxed.
    add_teleop_cli_args(
        parser,
        default_max_joint_vel=MAX_JOINT_VEL_SIM_RAD_S,
        default_method="pink_relaxed",
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
    parser.add_argument(
        "--mock-pattern",
        type=str,
        default="circle",
        choices=sorted(MOCK_PATTERNS),
        help="Motion pattern of the --mock device "
        "(circle = table circles, wrist = wrist oscillation, "
        "excursion = deliberately out-of-envelope strokes, "
        "roll_ratchet = repeated grip-twist/release-untwist cycles, "
        "joystick = scripted thumbstick roll/flex trims for --method mymethod)",
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

    # IK layer + thread kwargs from the shared helper — identical wiring to
    # tool/meta_quest_teleopration.py (mymethod = pink_relaxed + thumbstick
    # wrist trims; armplane = the tuned Pink solver). The sim thereby also
    # exercises the armplane/PinkIKSolver branch, not just MethodIKAdapter.
    ik_solver, thread_kwargs = create_teleop_stack(args, dt=1.0 / IK_SOLVER_RATE)

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
        kwargs=thread_kwargs,
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
    orient_errs: list[float] = []  # geodesic target-vs-measured EE angle (deg)
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

            # Tracking diagnostics against the commanded EE targets: position
            # distance plus the geodesic angle between the commanded and
            # measured EE rotations (sim-side diagnostics only).
            for side, (pos, rot) in markers.items():
                meas_pos, meas_rot = sim.eef_pose(side)
                track_errs.append(float(np.linalg.norm(pos - meas_pos)))
                cos_ang = np.clip((np.trace(rot @ meas_rot.T) - 1.0) / 2.0, -1.0, 1.0)
                orient_errs.append(float(np.degrees(np.arccos(cos_ang))))

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
                # Orientation triads (RGB axes) for each arm's MEASURED EE pose
                # (opaque) and its commanded TARGET pose (semi-transparent), on
                # top of the mocap position markers the scene already draws.
                scn = viewer_ctx.user_scn
                scn.ngeom = 0
                for side in SIDES:
                    meas_pos, meas_rot = sim.eef_pose(side)
                    _add_triad(scn, meas_pos, meas_rot, 1.0)
                    tpose = data_manager.get_target_pose(side)
                    if tpose is not None:
                        _add_triad(
                            scn, tpose[:3, 3] + world_offset[side], tpose[:3, :3], 0.45
                        )
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
        if orient_errs:
            oerrs = np.asarray(orient_errs)
            print(
                f"📊 EE orientation error while teleop active: "
                f"mean {oerrs.mean():.1f} deg, "
                f"p95 {np.percentile(oerrs, 95):.1f} deg "
                f"({len(oerrs)} samples)"
            )
        print("✅ Sim teleop stopped")


if __name__ == "__main__":
    main()
