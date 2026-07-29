"""Connect a single follower arm's Feetech bus for passive tooling.

Used by hardware inspection/setup tools (``tool/test_sensor_rates.py``,
``tool/set_wrist_roll_limits.py``) that operate on ONE arm without the
full ``SO101DualArm`` stack (which opens both buses and re-enables
torque). Torque is disabled right after connecting and no motion is
ever commanded — the arm goes limp.

Side mapping is the repo's verified convention: follower_0 (PORT_ID_0)
is the RIGHT arm, follower_1 (PORT_ID_1) the LEFT.
"""

from pathlib import Path
from typing import Dict

import draccus
import yaml
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

_root = Path(__file__).resolve().parent.parent.parent


def follower_motors() -> dict[str, Motor]:
    """The six-motor layout shared by every SO-101 arm bus.

    Followers and leaders use the identical Feetech sts3215 layout (ids
    1–6, five DEGREES body joints + a RANGE_0_100 gripper), so this
    describes any SO-101 bus, leader or follower.
    """
    return {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def connect_follower_bus(arm: str, port: str | None = None) -> FeetechMotorsBus:
    """Connect ONE follower bus read-only-ish: torque disabled, no motion.

    Args:
        arm: "right" (follower_0 / PORT_ID_0) or "left" (follower_1).
        port: serial-port override; defaults to robot.yaml's port for
            that arm (ttyACM devices enumerate in unstable order, so
            callers may resolve the port themselves).
    """
    robot_conf = yaml.safe_load((_root / "src/conf/robot.yaml").read_text())
    port_key, name_key = (
        ("PORT_ID_0", "ROBOT_NAME_0")
        if arm == "right"
        else ("PORT_ID_1", "ROBOT_NAME_1")
    )
    port = port or robot_conf[port_key]
    name = robot_conf[name_key]
    calib_path = _root / f"src/calibration_files/{name}.json"
    with open(calib_path) as f, draccus.config_type("json"):
        calibration = draccus.load(Dict[str, MotorCalibration], f)

    bus = FeetechMotorsBus(
        port=port,
        motors=follower_motors(),
        calibration=calibration,
    )
    print(f"🔌 connecting {arm} arm ({name}) on {port} ...")
    bus.connect(True)
    bus.disable_torque(num_retry=3)  # passive: safe to leave unattended
    print(f"  ✓ connected, torque disabled ({arm} arm is limp)")
    return bus


def _leader_calib_dir() -> Path:
    """The LeRobot calibration directory for SO-101 leader teleoperators.

    Leaders are calibrated with ``lerobot-calibrate --teleop.type=
    so101_leader`` (or ``lerobot-setup-motors``), which writes
    ``<HF_LEROBOT_CALIBRATION>/teleoperators/so_leader/<id>.json`` — a
    different tree from the followers' repo-local ``calibration_files/``.
    """
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

    return HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so_leader"


def discover_leader_calib_ids() -> list[str]:
    """Sorted ids of the calibrated SO-101 leaders (``leader_0`` …).

    Empty if no leader has been calibrated yet (directory missing/empty).
    """
    directory = _leader_calib_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def connect_leader_bus(port: str, calib_id: str) -> FeetechMotorsBus:
    """Connect ONE leader bus read-only-ish, calibrated to degrees.

    Mirrors :func:`connect_follower_bus` but loads the leader's
    calibration from the LeRobot teleoperator cache by id, so a
    ``sync_read("Present_Position")`` returns joint DEGREES about the
    calibrated mid-range. Torque is disabled (leaders are passive and
    must stay back-drivable). No motion is ever commanded.

    Args:
        port: serial port the leader enumerated on (resolve via the
            sensor map — ttyACM devices reorder across replugs).
        calib_id: calibration id, e.g. ``"leader_0"`` (see
            :func:`discover_leader_calib_ids`).

    Raises:
        FileNotFoundError: no calibration file for ``calib_id`` — the
            caller treats leaders as optional and skips on this.
    """
    calib_path = _leader_calib_dir() / f"{calib_id}.json"
    if not calib_path.is_file():
        raise FileNotFoundError(
            f"no leader calibration '{calib_id}' at {calib_path} — "
            "calibrate it with: lerobot-calibrate "
            f"--teleop.type=so101_leader --teleop.port={port} "
            f"--teleop.id={calib_id}"
        )
    with open(calib_path) as f, draccus.config_type("json"):
        calibration = draccus.load(Dict[str, MotorCalibration], f)

    bus = FeetechMotorsBus(
        port=port,
        motors=follower_motors(),
        calibration=calibration,
    )
    print(f"🔌 connecting leader ({calib_id}) on {port} ...")
    bus.connect(True)
    bus.disable_torque(num_retry=3)  # passive: safe to leave unattended
    print(f"  ✓ connected, torque disabled (leader {calib_id} is limp)")
    return bus
