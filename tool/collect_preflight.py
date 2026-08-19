#!/usr/bin/env python3
"""Pre-collection readiness check for the dual SO-101 rig.

Runs a battery of checks that the rig is at the final state suitable for data
collection and prints a green/red table. Non-destructive: it reads config and
(optionally) opens cameras/RealSense read-only; it never commands the arms.

    python tool/collect_preflight.py                # full check
    python tool/collect_preflight.py --no-hardware  # config/files only (CI)

Exit code is 0 when nothing FAILed (WARN is tolerated), else 1 — so it can gate
a collection script. Checks:

* sensor map assigned (both followers, cameras; leaders optional);
* follower + leader calibration files present;
* ready/rest pose configs valid;
* recording.yaml loads;
* every enabled camera opens at its configured resolution (hardware);
* RealSense present when realsense.enabled (hardware);
* disk headroom on the dataset volume;
* CPU governor = performance and USB autosuspend disabled (real-time jitter);
* an advisory on system load and the tactile-cable wrist-roll limits.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

from common.recording.usb_topology import (
    UsbLocation,
    device_location,
    directory_location,
    group_by_hub,
)

_root = Path(__file__).resolve().parent.parent

# Disk thresholds (GB free on the dataset volume) — match the recorder's.
_DISK_REFUSE_GB = 2.0
_DISK_WARN_GB = 10.0

OK, WARN, FAIL = "OK", "WARN", "FAIL"


class CheckResult(NamedTuple):
    name: str
    level: str  # OK | WARN | FAIL
    detail: str


# ── Pure classifiers (unit-tested) ───────────────────────────────────────────


def classify_disk(free_gb: float) -> CheckResult:
    if free_gb < _DISK_REFUSE_GB:
        return CheckResult(
            "disk space", FAIL, f"{free_gb:.1f} GB free (< {_DISK_REFUSE_GB:.0f})"
        )
    if free_gb < _DISK_WARN_GB:
        return CheckResult(
            "disk space", WARN, f"{free_gb:.1f} GB free (< {_DISK_WARN_GB:.0f})"
        )
    return CheckResult("disk space", OK, f"{free_gb:.1f} GB free")


def classify_governors(governors: list[str]) -> CheckResult:
    """All-performance = OK; anything else = WARN (jitter risk)."""
    uniq = sorted(set(governors))
    if not uniq:
        return CheckResult("cpu governor", WARN, "could not read scaling_governor")
    if uniq == ["performance"]:
        return CheckResult("cpu governor", OK, "performance")
    return CheckResult(
        "cpu governor", WARN, f"{','.join(uniq)} (set 'performance' to cut jitter)"
    )


def classify_usb_autosuspend(value: str) -> CheckResult:
    """usbcore autosuspend: -1/0 disabled = OK; a positive delay = WARN."""
    try:
        n = int(value.strip())
    except ValueError:
        return CheckResult("usb autosuspend", WARN, f"unparseable ({value!r})")
    if n <= 0:
        return CheckResult("usb autosuspend", OK, "disabled")
    return CheckResult(
        "usb autosuspend", WARN, f"{n}s (autosuspend can drop UVC frames)"
    )


_TOPOLOGY = "usb topology"
# Indent for the per-hub lines under a check's summary, matching the column _fmt
# starts the detail at (2 + 7 + 1 + 22 + 1).
_INDENT = "\n" + " " * 33


def classify_usb_topology(
    storage: "dict[str, UsbLocation | None]",
    sensors: "dict[str, UsbLocation | None]",
) -> CheckResult:
    """Judge how the dataset drive and the rig's sensors share USB hubs. Pure.

    A hub carrying both the drive being written to and the sensors being recorded
    is the serious case, and it FAILs: losing that hub does not interrupt a
    session, it deletes the file the session is being written into. Sensors
    sharing a hub only with each other is a WARN -- an episode can be paused and
    resumed, so the cost is recoverable -- and is often unavoidable on a rig with
    two arms and three cameras.
    """
    groups = group_by_hub({**storage, **sensors})
    if not groups:
        return CheckResult(_TOPOLOGY, WARN, "no USB devices resolved — cannot judge")
    lines = [
        f"{hub}: {', '.join(sorted(labels))}"
        for hub, labels in sorted(groups.items())
        if len(labels) > 1
    ]
    storage_hubs = {loc.hub for loc in storage.values() if loc is not None}
    clashes = sorted(
        hub
        for hub in storage_hubs
        if any(label in sensors for label in groups.get(hub, []))
    )
    if clashes:
        return CheckResult(
            _TOPOLOGY,
            FAIL,
            "the dataset drive shares a hub with the rig — one hub fault takes "
            "the session AND the drive it is written to; move the drive to a "
            "port on another controller" + _INDENT + _INDENT.join(lines),
        )
    if lines:
        return CheckResult(
            _TOPOLOGY,
            WARN,
            "sensors share a hub — a hub fault pauses every stream on it at once"
            + _INDENT
            + _INDENT.join(lines),
        )
    return CheckResult(_TOPOLOGY, OK, f"{len(groups)} hubs, nothing shared")


def sensor_map_status(sm: dict) -> CheckResult:
    arms = sm.get("arms") or {}
    cams = sm.get("cameras") or {}
    leaders = sm.get("leaders") or {}
    missing = [s for s in ("left", "right") if s not in arms]
    if missing:
        return CheckResult(
            "sensor map", FAIL, f"followers unassigned: {missing} — run --assign"
        )
    detail = f"{len(cams)} cameras, followers OK, {len(leaders)} leaders"
    return CheckResult("sensor map", OK, detail)


def pose_status(name: str, pose: dict) -> CheckResult:
    needed = {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
    missing = needed - set(pose)
    if missing:
        return CheckResult(name, FAIL, f"missing joints: {sorted(missing)}")
    return CheckResult(name, OK, "6 joints")


# ── Checks (filesystem + hardware) ────────────────────────────────────────────


def check_sensor_map() -> CheckResult:
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    if not SENSOR_MAP_PATH.exists():
        return CheckResult(
            "sensor map", FAIL, "not found — run tool/test_sensor_rates.py --assign"
        )
    return sensor_map_status(load_sensor_map(SENSOR_MAP_PATH))


def check_follower_calibration() -> CheckResult:
    cal = _root / "src/calibration_files"
    missing = [
        f"{n}.json"
        for n in ("follower_0", "follower_1")
        if not (cal / f"{n}.json").exists()
    ]
    if missing:
        return CheckResult("follower calibration", FAIL, f"missing {missing}")
    return CheckResult("follower calibration", OK, "follower_0, follower_1")


def check_leader_calibration() -> CheckResult:
    from common.follower_bus import discover_leader_calib_ids

    ids = discover_leader_calib_ids()
    if not ids:
        return CheckResult(
            "leader calibration",
            WARN,
            "none found (leaders optional; lerobot-calibrate)",
        )
    return CheckResult("leader calibration", OK, ", ".join(ids))


def check_pose_configs() -> CheckResult:
    import yaml

    results = []
    for name in ("ready_pos", "rest_pos"):
        path = _root / f"src/conf/{name}.yaml"
        if not path.exists():
            return CheckResult("pose configs", FAIL, f"{name}.yaml missing")
        results.append(pose_status(name, yaml.safe_load(path.read_text()) or {}))
    bad = [r for r in results if r.level != OK]
    if bad:
        return CheckResult("pose configs", FAIL, bad[0].detail)
    return CheckResult("pose configs", OK, "ready_pos, rest_pos")


def check_recording_config() -> CheckResult:
    from common.config_parser import load_recording_config

    try:
        cfg = load_recording_config()
    except Exception as e:
        return CheckResult("recording.yaml", FAIL, str(e))
    n = sum(1 for c in cfg["cameras"].values() if c["enabled"])
    rs = "realsense on" if cfg.get("realsense", {}).get("enabled") else "realsense off"
    return CheckResult("recording.yaml", OK, f"{n} cameras enabled, {rs}")


def _default_dataset_dir() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME)


def check_disk(dataset_dir: "Path | None") -> CheckResult:
    probe = dataset_dir if dataset_dir is not None else _default_dataset_dir()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    return classify_disk(free_gb)


def check_cpu_governor() -> CheckResult:
    govs = []
    for p in sorted(
        Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
    ):
        try:
            govs.append(p.read_text().strip())
        except OSError:
            pass
    return classify_governors(govs)


def check_usb_autosuspend() -> CheckResult:
    path = Path("/sys/module/usbcore/parameters/autosuspend")
    if not path.exists():
        return CheckResult("usb autosuspend", WARN, "usbcore parameter not found")
    try:
        return classify_usb_autosuspend(path.read_text())
    except OSError as e:
        return CheckResult("usb autosuspend", WARN, str(e))


def check_usb_topology(dataset_dir: "Path | None") -> CheckResult:
    """Map the dataset drive and every assigned sensor onto its USB hub."""
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    sensors: dict[str, UsbLocation | None] = {}
    if SENSOR_MAP_PATH.exists():
        sensor_map = load_sensor_map(SENSOR_MAP_PATH)
        for name, node in (sensor_map.get("cameras") or {}).items():
            sensors[f"camera {name}"] = device_location(node)
        for side, node in (sensor_map.get("arms") or {}).items():
            sensors[f"follower {side}"] = device_location(node)
        for side, entry in (sensor_map.get("leaders") or {}).items():
            node = entry.get("port") if isinstance(entry, dict) else entry
            if node:
                sensors[f"leader {side}"] = device_location(node)
    target = dataset_dir if dataset_dir is not None else _default_dataset_dir()
    storage = {f"dataset drive {target}": directory_location(target)}
    return classify_usb_topology(storage, sensors)


def check_load_advisory() -> CheckResult:
    import os

    try:
        load1 = os.getloadavg()[0]
        cpus = os.cpu_count() or 1
    except OSError:
        return CheckResult("system load", WARN, "unavailable")
    ratio = load1 / cpus
    level = WARN if ratio > 0.5 else OK
    return CheckResult(
        "system load",
        level,
        f"1-min load {load1:.1f} over {cpus} CPUs — stop other apps",
    )


def check_cameras() -> list[CheckResult]:
    import cv2  # type: ignore[import]

    from common.config_parser import load_recording_config
    from tool.meta_quest_teleopration import overlay_sensor_map_devices
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    cfg = load_recording_config()
    cameras = cfg["cameras"]
    # Match what the recorder opens: a stream assigned in sensor_map.yaml is
    # resolved to its stable by-path node, not the recording.yaml index.
    if SENSOR_MAP_PATH.exists():
        cameras = overlay_sensor_map_devices(cameras, load_sensor_map(SENSOR_MAP_PATH))
    out: list[CheckResult] = []
    for name, cam in cameras.items():
        if not cam["enabled"]:
            continue
        cap = cv2.VideoCapture(cam["device"], cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                out.append(
                    CheckResult(
                        f"camera {name}", FAIL, f"device {cam['device']} won't open"
                    )
                )
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            ok, frame = cap.read()
            if not ok or frame is None:
                out.append(CheckResult(f"camera {name}", FAIL, "opened but no frame"))
                continue
            h, w = frame.shape[:2]
            level = OK if (w, h) == (cam["width"], cam["height"]) else WARN
            out.append(
                CheckResult(
                    f"camera {name}",
                    level,
                    f"{w}x{h} (want {cam['width']}x{cam['height']})",
                )
            )
        finally:
            cap.release()
    return out


def check_realsense() -> CheckResult | None:
    from common.config_parser import load_recording_config

    rs_cfg = load_recording_config().get("realsense")
    if not rs_cfg or not rs_cfg["enabled"]:
        return None
    try:
        import pyrealsense2 as rs  # type: ignore[import]

        devices = list(rs.context().query_devices())
    except Exception as e:
        return CheckResult("realsense", FAIL, f"pyrealsense error: {e}")
    if not devices:
        return CheckResult("realsense", FAIL, "enabled but no device found")
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    return CheckResult(
        "realsense", OK, f"{len(devices)} device(s): {','.join(serials)}"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

_COLOR = {OK: "\033[92m", WARN: "\033[93m", FAIL: "\033[91m"}
_RESET = "\033[0m"


def _fmt(r: CheckResult, color: bool) -> str:
    tag = f"[{r.level}]"
    if color:
        tag = f"{_COLOR[r.level]}{tag}{_RESET}"
    return f"  {tag:<7} {r.name:<22} {r.detail}"


def run_checks(hardware: bool, dataset_dir: "Path | None" = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    file_checks: list[Callable[[], CheckResult]] = [
        check_sensor_map,
        check_follower_calibration,
        check_leader_calibration,
        check_pose_configs,
        check_recording_config,
        partial(check_disk, dataset_dir),
        check_cpu_governor,
        check_usb_autosuspend,
        partial(check_usb_topology, dataset_dir),
        check_load_advisory,
    ]
    for fn in file_checks:
        try:
            results.append(fn())
        except Exception as e:  # a check must never crash the preflight
            # functools.partial has no __name__; fall back to the wrapped fn.
            label = getattr(fn, "__name__", "") or getattr(
                getattr(fn, "func", None), "__name__", ""
            )
            results.append(CheckResult(label or "check", FAIL, f"check error: {e}"))
    results.append(
        CheckResult(
            "wrist-roll limits",
            WARN,
            "not auto-checked — run tool/set_wrist_roll_limits.py if tactile "
            "cameras are mounted",
        )
    )
    if hardware:
        try:
            results.extend(check_cameras())
        except Exception as e:
            results.append(CheckResult("cameras", FAIL, f"check error: {e}"))
        try:
            rs = check_realsense()
            if rs is not None:
                results.append(rs)
        except Exception as e:
            results.append(CheckResult("realsense", FAIL, f"check error: {e}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Skip camera/RealSense probing (config + files only)",
    )
    parser.add_argument("--no-color", action="store_true", help="Plain output")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Collection directory you will actually record into (the same "
        "--dir the collection tool gets). Disk headroom and USB topology are "
        "judged against this drive; without it the default dataset home is used",
    )
    args = parser.parse_args()

    results = run_checks(hardware=not args.no_hardware, dataset_dir=args.dir)
    color = not args.no_color and sys.stdout.isatty()
    print("\n🔎 Collection preflight\n")
    for r in results:
        print(_fmt(r, color))
    n_fail = sum(1 for r in results if r.level == FAIL)
    n_warn = sum(1 for r in results if r.level == WARN)
    print(f"\n  {len(results)} checks: {n_fail} FAIL, {n_warn} WARN\n")
    if n_fail:
        print("❌ not ready — resolve the FAILs above before collecting.")
        return 1
    print("✅ ready to collect (review any WARNs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
