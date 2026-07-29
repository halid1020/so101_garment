#!/usr/bin/env python3
"""Spin a bare SO-101 gripper MOTOR through a full 360 deg and verify it tracks.

⚠️  REMOVE THE GRIPPER MECHANICS FIRST. The jaws/linkage only travel a small
arc; forcing a full turn against them will damage the gripper. This test is for
a FREE motor shaft (mechanics detached), to confirm the STS3215 servo itself can
rotate a whole revolution.

Why this is not just ``motor_rotation_check.py --joint gripper``:
  The gripper motor is configured ``RANGE_0_100`` and its calibration clamps it
  to the jaw arc (``Min/Max_Position_Limit`` ~1588 ticks ~= 140 deg, plus a
  ``Homing_Offset``). To exercise a full turn this script temporarily:
    * zeroes ``Homing_Offset`` (raw Present_Position == absolute encoder tick),
    * widens the position limits to the full single-turn range (0..4095),
    * drives RAW ticks (bypassing the 0-100 calibration),
  then RESTORES the original homing offset and limits on exit. The STS3215 in
  POSITION mode is single-turn absolute: 4096 ticks = 360 deg, so one sweep
  0 -> 4095 is one full revolution.

This is a HARDWARE check, not an automated test: it is named
``gripper_full_turn_check.py`` (not ``test_*.py``) so unittest discovery /
``make test-integration`` will NOT pick it up.

Arm/side mapping (see CLAUDE.md): arm 0 = follower_0 = RIGHT, arm 1 = LEFT.

Example:
    python test/integration/gripper_full_turn_check.py --arm 0

Requires PYTHONPATH=.:src (set by `source setup.sh`).
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from src.so101_dual_arm import SO101DualArm  # noqa: E402

MOTOR = "gripper"
RESOLUTION = 4096  # STS3215: ticks per full 360 deg (from feetech tables)
MAX_TICK = RESOLUTION - 1  # 4095
TICKS_PER_DEG = RESOLUTION / 360.0

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
        description="Rotate a bare gripper motor a full 360 deg and verify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        default="0",
        choices=sorted(ARM_ALIASES),
        metavar="{0|right, 1|left}",
        help="Which arm (0=follower_0=RIGHT, 1=follower_1=LEFT).",
    )
    p.add_argument(
        "--step-deg",
        type=float,
        default=30.0,
        help="Increment between goal positions, degrees (default: 30).",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.15,
        help="Pause after each step, seconds, to let the motor settle "
        "(default: 0.15).",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of forward-and-back full sweeps (default: 1).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive safety confirmation.",
    )
    return p.parse_args()


def sweep(bus, ticks, dwell):
    """Drive the gripper through a list of raw goal ticks; return (goal, measured) pairs."""
    trace = []
    for goal in ticks:
        bus.write("Goal_Position", MOTOR, int(goal), normalize=False)
        time.sleep(dwell)
        measured = bus.read("Present_Position", MOTOR, normalize=False)
        trace.append((int(goal), int(measured)))
        deg = (measured % RESOLUTION) * 360.0 / RESOLUTION
        print(
            f"    goal {int(goal):4d}  measured {int(measured):4d}  (~{deg:6.1f} deg)"
        )
    return trace


def main() -> int:
    args = parse_args()
    arm_idx = ARM_ALIASES[args.arm]
    side = "RIGHT (follower_0)" if arm_idx == 0 else "LEFT (follower_1)"

    print("Gripper full-turn check")
    print(f"  arm     : {arm_idx}  -> {side}")
    print(
        f"  step    : {args.step_deg:.1f} deg  ({args.step_deg * TICKS_PER_DEG:.0f} ticks)"
    )
    print(f"  cycles  : {args.cycles}")
    print("\n⚠️  The gripper MECHANICS MUST BE REMOVED — this spins the bare")
    print("    motor shaft a full 360 deg. Running against the jaws will break them.")
    if not args.yes:
        try:
            input(
                "Mechanics removed & workspace clear? Press Enter (Ctrl+C to abort)... "
            )
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
    bus = dual_arm.bus_0 if arm_idx == 0 else dual_arm.bus_1

    # Remember the calibrated homing/limits so we can put them back exactly.
    orig = {
        reg: bus.read(reg, MOTOR, normalize=False)
        for reg in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")
    }
    print(f"Saved original gripper calibration registers: {orig}")

    try:
        # Open up the full single-turn range and make raw ticks == encoder ticks.
        with bus.torque_disabled(MOTOR):
            bus.write("Homing_Offset", MOTOR, 0, normalize=False)
            bus.write("Min_Position_Limit", MOTOR, 0, normalize=False)
            bus.write("Max_Position_Limit", MOTOR, MAX_TICK, normalize=False)
        bus.enable_torque(MOTOR)

        step = max(1, int(round(args.step_deg * TICKS_PER_DEG)))
        up = list(range(0, MAX_TICK, step)) + [MAX_TICK]
        down = list(reversed(up))

        measured_min = MAX_TICK
        measured_max = 0
        for c in range(args.cycles):
            print(f"\nCycle {c + 1}/{args.cycles}: forward 0 -> 360 deg")
            trace = sweep(bus, up, args.dwell)
            print("Cycle: back 360 -> 0 deg")
            trace += sweep(bus, down, args.dwell)
            measured_min = min(measured_min, min(m for _, m in trace))
            measured_max = max(measured_max, max(m for _, m in trace))

        span_deg = (measured_max - measured_min) * 360.0 / RESOLUTION
        print(
            f"\nMeasured travel span: {measured_min}..{measured_max} ticks "
            f"(~{span_deg:.1f} deg)"
        )
        # A full turn should sweep essentially the whole encoder range; allow a
        # small margin for the step granularity and steady-state error.
        ok = span_deg >= 350.0
        print(
            "Result:",
            "PASS — motor completed a full 360 deg turn"
            if ok
            else "CHECK — motor did NOT span a full turn (blocked? limits? mechanics on?)",
        )
        return 0 if ok else 2
    finally:
        # Always restore the gripper's real calibration and relax the arms.
        print("\nRestoring original gripper calibration registers...")
        try:
            with bus.torque_disabled(MOTOR):
                for reg, val in orig.items():
                    bus.write(reg, MOTOR, int(val), normalize=False)
        except Exception as e:  # noqa: BLE001 — cleanup must not mask the result
            print(
                f"⚠️  Could not restore gripper registers: {e}\n"
                f"    Re-run tool/fit_joint_offsets.py or re-flash calibration: {orig}"
            )
        dual_arm.disable_torque()


if __name__ == "__main__":
    raise SystemExit(main())
