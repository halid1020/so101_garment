#!/usr/bin/env python3
"""Drive ONE joint of ONE real SO-101 arm through a small rotation and verify it moved.

This is a HARDWARE integration check, not an automated unit test: it opens the
serial buses and physically rotates a motor. It is deliberately named
``motor_rotation_check.py`` (not ``test_*.py``) so ``make test-integration`` /
unittest discovery does NOT pick it up and try to drive absent hardware in CI.

Arm/side mapping (load-bearing — see CLAUDE.md and configs.py):
    arm 0  ->  follower_0  ->  RIGHT arm/handle
    arm 1  ->  follower_1  ->  LEFT  arm/handle

What it does:
  1. Connects to both arms (torque enabled, POSITION mode).
  2. Reads the current pose of both arms.
  3. Commands ONLY the chosen joint on the chosen arm to rotate by ``--delta``
     degrees (every other joint, and the whole other arm, is re-commanded to
     its current position so nothing else moves).
  4. Reads the joint back and reports commanded vs. measured rotation.
  5. Returns the joint to where it started, then disables torque.

Examples:
    # Rotate the RIGHT arm's wrist_roll by +20 deg and back:
    python test/integration/motor_rotation_check.py --arm 0 --joint wrist_roll

    # Rotate the LEFT arm's elbow by -15 deg over 3 s, don't return:
    python test/integration/motor_rotation_check.py \
        --arm left --joint elbow_flex --delta -15 --duration 3 --no-return

Requires PYTHONPATH=.:src (set by `source setup.sh`).
"""

import argparse
import sys
from pathlib import Path

import yaml

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from src.so101_dual_arm import SO101DualArm  # noqa: E402

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# --arm accepts an index or a side name; both resolve to the bus-0/bus-1 index.
ARM_ALIASES = {
    "0": 0,
    "right": 0,
    "follower_0": 0,
    "1": 1,
    "left": 1,
    "follower_1": 1,
}


def load_yaml(filepath: Path) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rotate one joint of one real SO-101 arm and verify motion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        default="0",
        choices=sorted(ARM_ALIASES),
        metavar="{0|right, 1|left}",
        help="Which arm: index 0/1 or side name right/left (0=follower_0=RIGHT).",
    )
    p.add_argument(
        "--joint",
        default="wrist_roll",
        choices=JOINTS,
        help="Joint to rotate (default: wrist_roll).",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=20.0,
        help="Rotation in degrees relative to the current position "
        "(gripper: 0-100 units). Default: +20.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Seconds for the motion (default: 2.0).",
    )
    p.add_argument(
        "--no-return",
        dest="do_return",
        action="store_false",
        help="Leave the joint at the target instead of returning to start.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive safety confirmation.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arm_idx = ARM_ALIASES[args.arm]
    side = "RIGHT (follower_0)" if arm_idx == 0 else "LEFT (follower_1)"

    print("Motor rotation check")
    print(f"  arm     : {arm_idx}  -> {side}")
    print(f"  joint   : {args.joint}")
    print(f"  delta   : {args.delta:+.2f} (degrees; gripper in 0-100 units)")
    print(f"  duration: {args.duration:.2f} s")
    print(f"  return  : {args.do_return}")

    if not args.yes:
        print("\n⚠️  A motor will physically rotate. Keep the workspace clear.")
        try:
            input("Press Enter to proceed (Ctrl+C to abort)... ")
        except KeyboardInterrupt:
            print("\nAborted.")
            return 1

    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
    }

    print("\nConnecting to the arms...")
    dual_arm = SO101DualArm(config)

    try:
        # Read the starting pose of BOTH arms; we re-command everything to its
        # current value so only the one target joint on the one target arm moves.
        pose_0, pose_1 = dual_arm.read_positions()
        target_pose = pose_0 if arm_idx == 0 else pose_1
        other_pose = pose_1 if arm_idx == 0 else pose_0

        start = target_pose[args.joint]
        goal = start + args.delta
        print(f"\nStart {args.joint}: {start:.2f}")
        print(f"Goal  {args.joint}: {goal:.2f}")

        desired_target = dict(target_pose)
        desired_target[args.joint] = goal
        # Pass each arm its own desired dict in bus order (0 then 1).
        desired_0 = desired_target if arm_idx == 0 else dict(other_pose)
        desired_1 = dict(other_pose) if arm_idx == 0 else desired_target

        print("Rotating...")
        dual_arm.move_to_joint_pose(desired_0, desired_1, args.duration)
        dual_arm.hold_position(0.3)

        pose_0, pose_1 = dual_arm.read_positions()
        measured = (pose_0 if arm_idx == 0 else pose_1)[args.joint]
        moved = measured - start
        print(f"\nMeasured {args.joint}: {measured:.2f}")
        print(f"Commanded rotation: {args.delta:+.2f}")
        print(f"Measured  rotation: {moved:+.2f}")
        # Tolerance is generous: Feetech position resolution + gravity sag on an
        # unsupported link means a few degrees of steady-state error is normal.
        ok = abs(moved - args.delta) <= max(3.0, abs(args.delta) * 0.2)
        print(
            "Result:",
            "PASS — motor rotated as commanded"
            if ok
            else "CHECK — measured rotation differs from command (see notes above)",
        )

        if args.do_return:
            print("\nReturning to start...")
            desired_target[args.joint] = start
            desired_0 = desired_target if arm_idx == 0 else dict(other_pose)
            desired_1 = dict(other_pose) if arm_idx == 0 else desired_target
            dual_arm.move_to_joint_pose(desired_0, desired_1, args.duration)

        return 0 if ok else 2
    finally:
        dual_arm.disable_torque()


if __name__ == "__main__":
    raise SystemExit(main())
