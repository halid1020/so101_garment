#!/usr/bin/env python3
"""Measure the Quest controller's handle axis in its tracked body frame.

Why: the teleop mapping points the gripper along the controller's HANDLE
(top -> bottom). The handle's direction inside the tracked (aim) frame is a
property of the headset/APK conventions that we have repeatedly guessed
wrong — this tool measures it instead.

Procedure (takes ~10 seconds):
  1. Run this script with the Quest connected (same as teleop).
  2. Pick up the RIGHT controller. Hold it so the HANDLE is PLUMB
     VERTICAL — imagine the handle is a nail you want to hammer straight
     down into the floor. Grip naturally, don't twist.
  3. Hold still through the countdown; the script averages the pose stream
     and prints the HANDLE_AXIS line to paste into src/common/configs.py.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from meta_quest_teleop.reader import MetaQuestReader  # noqa: E402

WORLD_DOWN = np.array([0.0, 0.0, -1.0])  # reader's ROS world frame, z up


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Quest handle axis")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--seconds", type=float, default=3.0, help="sampling time")
    args = parser.parse_args()

    print("🎮 Connecting to Meta Quest...")
    reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    print()
    print("=" * 64)
    print(f"Hold the {args.hand.upper()} controller with the HANDLE PLUMB")
    print("VERTICAL (like a nail pointing straight down at the floor).")
    print("Natural grip, no twist. Hold still...")
    print("=" * 64)
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1.0)

    print(f"📐 Sampling for {args.seconds:.0f} s — keep holding...")
    axes = []
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        tf = reader.get_hand_controller_transform_ros(hand=args.hand)
        if tf is not None:
            # World "down" expressed in the controller's body frame IS the
            # handle's top->bottom axis while the handle is held vertical.
            axes.append(tf[:3, :3].T @ WORLD_DOWN)
        time.sleep(0.02)
    reader.stop()

    if len(axes) < 20:
        print(f"❌ Only {len(axes)} samples — is the controller tracked? Retry.")
        sys.exit(1)

    mean_axis = np.mean(axes, axis=0)
    mean_axis /= np.linalg.norm(mean_axis)
    spread = np.degrees(
        np.max([np.arccos(np.clip(a @ mean_axis, -1, 1)) for a in axes])
    )

    print(f"\n✓ {len(axes)} samples, worst deviation from mean: {spread:.1f}°")
    if spread > 5.0:
        print("⚠️  You moved quite a bit — consider re-running for a cleaner value.")

    print("\nPaste this into src/common/configs.py (replacing HANDLE_AXIS = None):\n")
    print(f"HANDLE_AXIS = [{mean_axis[0]:.4f}, {mean_axis[1]:.4f}, {mean_axis[2]:.4f}]")
    # Context: angle from the analytic guess family (-y tilted toward +z)
    pitch = np.degrees(np.arctan2(mean_axis[2], -mean_axis[1]))
    print(
        f"\n(implied handle pitch ≈ {pitch:.1f} degree, x-component "
        f"{mean_axis[0]:+.3f} — large |x| means the handle leans sideways "
        f"in the tracked frame, which the old scalar offset could never "
        f"express)"
    )


if __name__ == "__main__":
    main()
