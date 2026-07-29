#!/usr/bin/env python3
"""Fit hardware->URDF joint zero offsets from a table-drag test.

Why: pan-driven arcs are easy but radial straight lines bend. That is the
signature of joint-zero offsets on shoulder_lift / elbow_flex / wrist_flex:
the IK model then believes it is drawing a straight line while the real arm
draws a vertical arc. (Pure pan motion is immune, which is why circles
work.)

The idea of the test: with motors relaxed, you slide the gripper's tip on
the table by hand. The real tip height is constant (it's on the table!),
so if the model disagrees with itself about that height across poses, the
disagreement measures the model error — and the script solves for the
joint-zero corrections that remove it.

Run:  python tool/fit_joint_offsets.py --arm left
then again with --arm right. The script guides you through the rest.
"""

# TODO: do we really need this script?

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import pinocchio as pin  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from common.configs import (  # noqa: E402
    DUAL_URDF_PATH,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from src.so101_dual_arm import SO101DualArm  # noqa: E402

_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# Offsets are fitted on these joints (pan and roll do not affect tip height)
_FIT_JOINTS = [1, 2, 3]  # shoulder_lift, elbow_flex, wrist_flex


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def say(text: str = "") -> None:
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit joint zero offsets")
    parser.add_argument("--arm", choices=["left", "right"], required=True)
    parser.add_argument(
        "--seconds",
        type=float,
        default=90.0,
        help="total recording time across the 3 steps (default 90 = 30s each)",
    )
    parser.add_argument(
        "--read-time",
        type=int,
        default=10,
        dest="read_time",
        help="pause before each step to read its instruction (default 10s)",
    )
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()

    arm = args.arm.upper()

    say()
    say("=" * 66)
    say(f"  JOINT ZERO CALIBRATION — {arm} ARM")
    say("=" * 66)
    say()
    say("What this does: you will slide the gripper tip around ON the")
    say("table while the arm is limp. Because the tip stays on the table,")
    say("its real height never changes — any 'change' the software computes")
    say("is model error, and this script measures and corrects it.")
    say()
    say("Before we start, check:")
    say("  [] The table in front of the arm is clear.")
    say(f"  [] You can comfortably reach the {arm} arm with one hand.")
    say("  [] Look at the gripper: one finger is part of the arm's body")
    say("      and cannot move; the other finger opens and closes. During")
    say("      the test, touch the table with the tip of the finger that")
    say("      CANNOT move. (The moving finger swings freely once the")
    say("      motors are off, so it would ruin the measurement.)")
    say()
    say("During the test you will simply drag that fingertip around on")
    say("the table, like a pencil drawing on paper. The only rules:")
    say("  - the tip must stay touching the table the whole time,")
    say("  - press only lightly (don't bend the plastic),")
    say("  - move slowly and smoothly.")
    say()
    input("Press Enter when ready to connect to the arms... ")

    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
    }
    say("\nConnecting to the arms...")
    dual_arm = SO101DualArm(config)
    bus = dual_arm.bus_0 if args.arm == "left" else dual_arm.bus_1

    say()
    say("!" * 66)
    say(f"  NEXT STEP: the {arm} arm's motors will TURN OFF and the arm")
    say("  will fall if nobody holds it.")
    say()
    say(f"--> Grab the {arm} arm near the gripper NOW, then press Enter.")
    say("!" * 66)
    input("Holding it? Press Enter to release the motors... ")
    dual_arm.disable_torque()
    say(f"\n✓ Motors off. The {arm} arm is limp — you are in control.")
    say()
    step_len = args.seconds / 3.0
    say("Now place the FIXED fingertip on the table, like a pencil you're")
    say("about to draw with. Keep it pressed gently on the table the WHOLE")
    say("time.")
    say()
    say(f"The test has 3 steps of {step_len:.0f} seconds each. Before every")
    say(f"step you get a {args.read_time}-second pause to read what to do —")
    say("nothing is recorded during the pauses, so there is no rush.")
    say("(Want even more time per step? Re-run with e.g. --seconds 150)")
    say()
    input("Tip on the table? Press Enter when you are ready... ")

    # Coaching phases (equal thirds of the recording)
    phases = [
        "Draw a long straight line on the table: drag the tip AWAY from "
        "the robot as far as it reaches, then drag it back CLOSE to the "
        "robot. Take about 3 seconds each way. Repeat, slowly.",
        "Now draw the same back-and-forth line, but a hand's width "
        "further to the LEFT or RIGHT on the table.",
        "Keep the tip on the table and slowly rock the gripper: lean it "
        "forward (tip pointing away from you), then lean it backward "
        "(tip pointing toward you), while sliding it a little. Like "
        "tilting a pencil while its point stays on the paper.",
    ]

    if args.arm == "left":
        offsets = np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=float)
        signs = np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=float)
    else:
        offsets = np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=float)
        signs = np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=float)
    samples: list[np.ndarray] = []
    dt = 1.0 / args.rate
    phase_len = args.seconds / len(phases)

    for phase_idx, instruction in enumerate(phases):
        say(f"\n▶ STEP {phase_idx + 1}/{len(phases)}:")
        say(f"  {instruction}")
        say()
        # Reading pause — recording is NOT running yet.
        for i in range(args.read_time, 0, -1):
            print(
                f"\r  (take your time to read — this step starts in {i:2d}s, "
                "tip stays on the table) ",
                end="",
                flush=True,
            )
            time.sleep(1.0)
        print("\r" + " " * 72, end="")
        say("\r  ✱ GO — recording this step now:")

        t_start = time.time()
        t_end = t_start + phase_len
        while True:
            now = time.time()
            if now >= t_end:
                break
            remaining = t_end - now
            bar_done = int(30 * (now - t_start) / phase_len)
            print(
                f"\r  [{'#' * bar_done}{'.' * (30 - bar_done)}] "
                f"{remaining:4.0f}s left in this step, "
                f"{len(samples):4d} samples total ",
                end="",
                flush=True,
            )
            pos = bus.sync_read("Present_Position")
            q_deg = signs * np.array([pos[j] for j in _BODY_JOINTS]) + offsets
            samples.append(np.radians(q_deg))
            time.sleep(dt)
        say("\n  ✓ Step done. Keep the tip on the table.")

    say(f"\n✓ All steps done — {len(samples)} samples. You can let go of the")
    say("  arm now (lay it down gently, the motors are still off).")
    say("  Crunching the numbers...")

    # FK setup on the dual model (chains are independent; other arm at zero)
    model = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
    data = model.createData()
    eef_id = model.getFrameId(f"{args.arm}_eef_link")
    q_base = pin.neutral(model)
    joint_ids = [
        model.joints[model.getJointId(f"{args.arm}_{j}")].idx_q for j in _BODY_JOINTS
    ]

    Q = np.array(samples)

    def tip_heights(params: np.ndarray) -> np.ndarray:
        d = np.zeros(5)
        d[_FIT_JOINTS] = params[:3]
        tip = np.array([params[3], 0.0, params[4]])
        zs = np.empty(len(Q))
        for i, q_arm in enumerate(Q):
            q = q_base.copy()
            q[joint_ids] = q_arm + d
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            M = data.oMf[eef_id]
            zs[i] = (M.rotation @ tip + M.translation)[2]
        return zs

    def residuals(params: np.ndarray) -> np.ndarray:
        zs = tip_heights(params)
        return zs - params[5]  # params[5] = table height

    x0 = np.array([0.0, 0.0, 0.0, 0.10, 0.0, 0.0])
    z0 = tip_heights(x0)
    result = least_squares(residuals, x0, method="lm")
    d_deg = np.degrees(result.x[:3])
    zf = tip_heights(result.x)

    rms_before = float(np.std(z0))
    rms_after = float(np.std(zf))

    say()
    say("=" * 66)
    say("  RESULTS")
    say("=" * 66)
    say()
    say("The tip was really at ONE height the whole time. The model thought")
    say("its height wandered by:")
    say(f"    before correction:  ±{1000 * rms_before:5.1f} mm")
    say(f"    after  correction:  ±{1000 * rms_after:5.1f} mm")
    say()

    if rms_before < 0.003:
        say("✅ VERDICT: your model was already accurate (under 3 mm).")
        say("   Joint offsets are NOT what bends your straight lines —")
        say("   the likely culprit is servo stiffness / gravity sag.")
        say("   Nothing to paste; you can skip the other arm too.")
        return

    if rms_after > 0.5 * rms_before or rms_after > 0.005:
        say("⚠️  VERDICT: the numbers improved a little, but not enough to")
        say("   trust the result. Please re-run it — these three things")
        say("   make the biggest difference:")
        say("   1. Make the back-and-forth line LONGER: from as far as the")
        say("      arm can reach all the way back near the robot's base.")
        say("   2. Move SLOWER — each line should take a slow 1-2-3 count")
        say("      out and the same back.")
        say("   3. In the last step, exaggerate the tilting: lean the")
        say("      gripper clearly forward, then clearly backward, keeping")
        say("      the tip touching the table.")
        say("   Also make sure the tip never lifts off and you press only")
        say("   lightly (bending the plastic fakes a height change).")
        return

    say("✅ VERDICT: real joint-zero errors found and corrected:")
    for name_idx, dd in zip(_FIT_JOINTS, d_deg):
        say(f"       {_BODY_JOINTS[name_idx]:>14s}: {dd:+.2f}°")
    say()
    new = offsets.copy()
    new[_FIT_JOINTS] += d_deg
    var = f"{args.arm.upper()}_ARM_HW_TO_URDF_OFFSETS_DEG"
    say("NEXT STEP: open src/common/configs.py and replace this line:")
    say(f"  {var} = [" + ", ".join(f"{v:.2f}" for v in new) + "]")
    say("(These corrections are on top of the offsets that were active")
    say(" during this recording, so pasting the line above is all you do.)")


if __name__ == "__main__":
    main()
