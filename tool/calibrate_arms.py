"""One-shot calibration for all four SO-101 arms (2 followers + 2 leaders).

Walks the four arms in a fixed order — follower right, follower left,
leader right, leader left — and runs LeRobot's interactive
``lerobot-calibrate`` for each, so the whole rig is calibrated in a
single session instead of four separate commands.

Each arm's PORT comes from ``src/conf/sensor_map.yaml``
(``tool/test_sensor_rates.py --assign``); its calibration ID is fixed by
side in ``robot.yaml`` (followers ``ROBOT_NAME_0/1`` = follower_0/1,
leaders ``LEADER_ID_LEFT/RIGHT`` = leader_left/right). An arm not
assigned in the sensor map is skipped with a warning, never fatal.

LeRobot writes each calibration into its own cache
(``$HF_LEROBOT_CALIBRATION/{robots/so_follower,teleoperators/so_leader}/
<id>.json``). Leaders are read from there directly by
``common.follower_bus.connect_leader_bus``; followers are ALSO copied
into this repo's ``src/calibration_files/<name>.json`` afterwards, which
is where ``connect_follower_bus`` (and the dual-arm stack) load them.

Usage:

    venv/bin/python tool/calibrate_arms.py            # all four, in order
    venv/bin/python tool/calibrate_arms.py --dry-run  # print the plan only
    venv/bin/python tool/calibrate_arms.py --only followers
    venv/bin/python tool/calibrate_arms.py --only leader_left --only follower_0

Calibration is interactive: for each arm LeRobot asks you to move every
joint through its full range, so keep the terminal focused and follow its
prompts. The arms must be powered and reachable on their assigned ports.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_root = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ArmCalib:
    """One arm's calibration recipe (pure data — no hardware)."""

    label: str  # human label, e.g. "follower right"
    device_arg: str  # draccus group: "robot" or "teleop"
    device_type: str  # "so101_follower" / "so101_leader"
    calib_id: str  # calibration id (also the cache file stem)
    side: str  # "right" / "left"
    port: str | None  # assigned port, or None if unassigned
    copy_to: Path | None  # repo calibration file (followers only)

    @property
    def assigned(self) -> bool:
        return bool(self.port)

    def calibrate_argv(self) -> list[str]:
        """The ``lerobot-calibrate`` command for this arm."""
        return [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_calibrate",
            f"--{self.device_arg}.type={self.device_type}",
            f"--{self.device_arg}.port={self.port}",
            f"--{self.device_arg}.id={self.calib_id}",
        ]


def build_calibration_plan(
    sensor_map: dict, robot_conf: dict, calib_files_dir: Path
) -> list[ArmCalib]:
    """The ordered four-arm plan from the sensor map + robot.yaml.

    Followers copy their result into ``calib_files_dir/<name>.json``;
    leaders are consumed straight from LeRobot's cache. An unassigned arm
    is still returned (``port=None``) so the plan/dry-run lists it as
    skipped. Pure — unit-tested.
    """
    arms = sensor_map.get("arms") or {}
    leaders = sensor_map.get("leaders") or {}

    def leader_port(side: str) -> str | None:
        entry = leaders.get(side) or {}
        return entry.get("port") if isinstance(entry, dict) else None

    follower_id = {
        "right": robot_conf["ROBOT_NAME_0"],
        "left": robot_conf["ROBOT_NAME_1"],
    }
    leader_id = {
        "right": robot_conf["LEADER_ID_RIGHT"],
        "left": robot_conf["LEADER_ID_LEFT"],
    }

    plan: list[ArmCalib] = []
    for side in ("right", "left"):
        fid = follower_id[side]
        plan.append(
            ArmCalib(
                label=f"follower {side}",
                device_arg="robot",
                device_type="so101_follower",
                calib_id=fid,
                side=side,
                port=arms.get(side),
                copy_to=calib_files_dir / f"{fid}.json",
            )
        )
    for side in ("right", "left"):
        plan.append(
            ArmCalib(
                label=f"leader {side}",
                device_arg="teleop",
                device_type="so101_leader",
                calib_id=leader_id[side],
                side=side,
                port=leader_port(side),
                copy_to=None,
            )
        )
    return plan


def follower_cache_fpath(calib_id: str, cache_root: Path) -> Path:
    """Where LeRobot writes a so101_follower calibration for ``calib_id``."""
    return cache_root / "robots" / "so_follower" / f"{calib_id}.json"


def select_arms(plan: list[ArmCalib], only: list[str]) -> list[ArmCalib]:
    """Filter the plan by ``--only`` tokens (groups or calib ids).

    Tokens: ``followers`` / ``leaders`` (whole group) or an exact calib id
    (``follower_0``, ``leader_left`` …). Empty ``only`` selects every arm.
    Pure — unit-tested.
    """
    if not only:
        return list(plan)
    wanted = set(only)
    out = []
    for arm in plan:
        group = "followers" if arm.device_arg == "robot" else "leaders"
        if group in wanted or arm.calib_id in wanted:
            out.append(arm)
    return out


def _load_configs() -> tuple[dict, dict]:
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    sensor_map = load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
    robot_conf = yaml.safe_load((_root / "src/conf/robot.yaml").read_text())
    return sensor_map, robot_conf


def _print_plan(plan: list[ArmCalib]) -> None:
    print("\nCalibration plan (order: follower right/left, leader right/left):")
    for arm in plan:
        if arm.assigned:
            dest = f" → {arm.copy_to}" if arm.copy_to else ""
            print(f"  • {arm.label:<15} id={arm.calib_id:<13} port={arm.port}{dest}")
        else:
            print(f"  • {arm.label:<15} id={arm.calib_id:<13} UNASSIGNED — will skip")


def _calibrate_one(arm: ArmCalib, cache_root: Path) -> bool:
    """Run one arm's interactive calibration; copy followers into the repo.

    Returns True on success, False if the arm was skipped or failed.
    """
    if not arm.assigned:
        print(f"⚠️  {arm.label} not assigned in sensor_map.yaml — skipped")
        return False
    if not Path(arm.port).exists():  # type: ignore[arg-type]
        print(f"⚠️  {arm.label} port {arm.port} missing — replug/--assign; skipped")
        return False

    print(f"\n▶ calibrating {arm.label} ({arm.calib_id}) on {arm.port}")
    print("  follow LeRobot's prompts: move every joint through its full range")
    result = subprocess.run(arm.calibrate_argv())
    if result.returncode != 0:
        print(f"❌ {arm.label} calibration failed (exit {result.returncode})")
        return False

    if arm.copy_to is not None:
        src = follower_cache_fpath(arm.calib_id, cache_root)
        if not src.is_file():
            print(f"⚠️  expected calibration at {src} not found — repo copy skipped")
            return False
        arm.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, arm.copy_to)
        print(f"  ✓ copied {src.name} → {arm.copy_to}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TOKEN",
        help="Calibrate a subset: 'followers', 'leaders', or an exact id "
        "(follower_0, follower_1, leader_left, leader_right). Repeatable; "
        "default is all four.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the exact commands, then exit without running.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before calibrating.",
    )
    args = parser.parse_args()

    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    sensor_map, robot_conf = _load_configs()
    plan = select_arms(
        build_calibration_plan(sensor_map, robot_conf, _root / "src/calibration_files"),
        args.only,
    )
    if not plan:
        raise SystemExit(f"❌ no arms match --only {args.only}")

    _print_plan(plan)

    if args.dry_run:
        print("\nCommands (--dry-run, nothing run):")
        for arm in plan:
            if arm.assigned:
                print("  " + " ".join(arm.calibrate_argv()))
        return

    assigned = [a for a in plan if a.assigned]
    if not assigned:
        raise SystemExit("❌ none of the selected arms are assigned — run --assign")
    if not args.yes:
        reply = input(f"\nCalibrate {len(assigned)} arm(s) now? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return

    ok, skipped = 0, 0
    for arm in plan:
        if _calibrate_one(arm, HF_LEROBOT_CALIBRATION):
            ok += 1
        else:
            skipped += 1
    print(f"\n✅ calibrated {ok} arm(s); {skipped} skipped/failed.")


if __name__ == "__main__":
    main()
