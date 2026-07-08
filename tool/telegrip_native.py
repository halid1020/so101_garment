#!/usr/bin/env python3
"""Drive the two SO-101 arms with the ORIGINAL Telegrip stack (DipFlip/telegrip).

This script is *only* an entry point: every line of control code that runs —
web UI, WebXR VR streaming, keyboard input, PyBullet IK, LeRobot motor I/O —
comes from the unmodified Telegrip checkout. All this wrapper does is

  1. locate the Telegrip checkout (default: ../telegrip next to this repo)
     and put it on ``sys.path``,
  2. copy this rig's motor calibrations to where LeRobot's ``SOFollower``
     looks for Telegrip's arm ids (``left_follower`` / ``right_follower``),
  3. translate this repo's port mapping (src/conf/robot.yaml) into
     Telegrip's ``--left-port`` / ``--right-port`` CLI overrides, and
  4. call ``telegrip.main.main_cli()``.

Arm mapping (load-bearing, see CLAUDE.md): follower_0 = RIGHT arm
(PORT_ID_0), follower_1 = LEFT arm (PORT_ID_1).

Usage (see markdowns/telegrip_native.md for full instructions):
    venv/bin/python tool/telegrip_native.py                 # full stack
    venv/bin/python tool/telegrip_native.py --no-viz        # headless PyBullet
    venv/bin/python tool/telegrip_native.py --autoconnect --log-level info

Unrecognized arguments are forwarded verbatim to Telegrip's own CLI
(--no-robot, --no-vr, --no-keyboard, --https-port, ...).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TELEGRIP_ROOT = REPO_ROOT.parent / "telegrip"

# Telegrip arm id -> this repo's calibration file (follower_0=RIGHT, 1=LEFT).
CALIBRATION_SOURCES = {
    "right_follower": REPO_ROOT / "src/calibration_files/follower_0.json",
    "left_follower": REPO_ROOT / "src/calibration_files/follower_1.json",
}


def default_ports() -> dict[str, str]:
    """Left/right serial ports from this repo's robot.yaml (flat file)."""
    conf = yaml.safe_load((REPO_ROOT / "src/conf/robot.yaml").read_text())
    conf = conf.get("robot", conf)  # tolerate a nested {"robot": ...} layout
    return {
        "right": conf.get("PORT_ID_0", "/dev/ttyACM0"),
        "left": conf.get("PORT_ID_1", "/dev/ttyACM1"),
    }


def lerobot_calibration_dir() -> Path:
    """Mirror lerobot.utils.constants without importing lerobot up front."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    lerobot_home = Path(
        os.environ.get("HF_LEROBOT_HOME", str(Path(hf_home) / "lerobot"))
    ).expanduser()
    calibration = Path(
        os.environ.get("HF_LEROBOT_CALIBRATION", str(lerobot_home / "calibration"))
    ).expanduser()
    # Robot.calibration_dir = HF_LEROBOT_CALIBRATION / "robots" / Robot.name
    return calibration / "robots" / "so_follower"


def sync_calibrations(refresh: bool) -> None:
    """Copy this rig's calibrations to the ids Telegrip instantiates."""
    dest_dir = lerobot_calibration_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for robot_id, src in CALIBRATION_SOURCES.items():
        if not src.exists():
            sys.exit(f"❌ Missing calibration source {src} — run arm calibration first")
        dst = dest_dir / f"{robot_id}.json"
        if dst.exists() and not refresh:
            if dst.read_bytes() != src.read_bytes():
                print(
                    f"⚠️  {dst} differs from {src.name} — keeping the existing "
                    "file (pass --refresh-calibration to overwrite)"
                )
            continue
        shutil.copy2(src, dst)
        print(f"📋 Calibration: {src.name} → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other options are forwarded to Telegrip's CLI unchanged.",
    )
    parser.add_argument(
        "--telegrip-root",
        type=Path,
        default=DEFAULT_TELEGRIP_ROOT,
        help=f"Path to the Telegrip checkout (default: {DEFAULT_TELEGRIP_ROOT})",
    )
    parser.add_argument(
        "--left-port", default=None, help="Override LEFT-arm (follower_1) port"
    )
    parser.add_argument(
        "--right-port", default=None, help="Override RIGHT-arm (follower_0) port"
    )
    parser.add_argument(
        "--refresh-calibration",
        action="store_true",
        help="Overwrite left/right_follower.json even if they already exist",
    )
    args, forwarded = parser.parse_known_args()

    telegrip_root = args.telegrip_root.resolve()
    if not (telegrip_root / "telegrip" / "main.py").exists():
        sys.exit(
            f"❌ Telegrip checkout not found at {telegrip_root}\n"
            "   Clone it first:\n"
            f"   git clone https://github.com/DipFlip/telegrip {telegrip_root}"
        )
    sys.path.insert(0, str(telegrip_root))

    try:
        import pybullet  # noqa: F401  (Telegrip hard-requires it)
    except ImportError:
        sys.exit(
            "❌ pybullet is not installed in this venv (Telegrip needs it):\n"
            f"   {REPO_ROOT}/venv/bin/pip install pybullet"
        )

    ports = default_ports()
    left_port = args.left_port or ports["left"]
    right_port = args.right_port or ports["right"]

    sync_calibrations(args.refresh_calibration)

    print("=" * 60)
    print("TELEGRIP (original stack) → dual SO-101")
    print(f"  checkout : {telegrip_root}")
    print(f"  LEFT  arm: follower_1 on {left_port}")
    print(f"  RIGHT arm: follower_0 on {right_port}")
    print("=" * 60)

    # Hand off to Telegrip's own CLI — from here on it is 100% their code.
    sys.argv = [
        "telegrip",
        "--left-port",
        left_port,
        "--right-port",
        right_port,
        *forwarded,
    ]
    from telegrip.main import main_cli

    main_cli()


if __name__ == "__main__":
    main()
