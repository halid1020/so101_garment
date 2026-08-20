"""Binding devices to stream names from the console (the Signals tab).

A rig of this kind is assembled from identical-looking USB devices: six
gripper and wrist cameras that differ only by which socket they are in, and
four SO-101 buses that enumerate in whatever order they were plugged. So the
map from a stream NAME to a stable device path is worked out by identifying
each device physically -- press a gel and watch which camera shows it, wiggle an
arm and watch which port's ticks move -- and that is what this module serves.

The rules are here and the routes are in ``sensors_api``; the desktop tool
``tool/test_sensor_rates.py --assign`` remains the offline way to do the same
job, and both write the same per-machine file through the same loader and
saver, so neither can invent its own format.

Everything is refused while a collection session runs: a session holds every
camera and both arm buses, and a probe would either fail or disturb it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from tool.test_sensor_rates import (  # noqa: F401 — re-exported: the routes resolve nodes with it; the desktop tool owns these definitions
    ASSIGNABLE_CAMERA_NAMES,
    SENSOR_MAP_PATH,
    _drop_node,
    _drop_serial_node,
    _serial_node,
    discover_capture_devices,
    discover_realsense_devices,
    discover_serial_ports,
    load_sensor_map,
    save_sensor_map,
    stable_device_path,
)

ARM_ROLES = ("follower", "leader")
ARM_SIDES = ("right", "left")

# A joint has to move by more than the bus's own read jitter before it counts
# as "this is the arm you are wiggling"; the desktop tool uses the same figure.
WIGGLE_TICKS = 5


def read_map(path: "Path | None" = None) -> "dict[str, Any]":
    """The saved assignments, or empty sections if nothing is assigned yet."""
    path = Path(path or SENSOR_MAP_PATH)
    if not path.exists():
        return {"cameras": {}, "arms": {}, "leaders": {}, "realsense": {}}
    return load_sensor_map(path)


def write_map(sensor_map: "dict[str, Any]", path: "Path | None" = None) -> None:
    """Persist the whole map (the saver rewrites the file wholesale)."""
    save_sensor_map(Path(path or SENSOR_MAP_PATH), sensor_map)


def assign_camera(
    sensor_map: "dict[str, Any]", name: str, node: str
) -> "dict[str, Any]":
    """Bind a camera stream name to ``node``. Pure — unit-tested.

    A physical camera is one stream, so binding it to a name first clears it
    from whatever name it had before: an operator who realises two names are
    swapped fixes it by reassigning, not by hunting for the stale entry.
    """
    if name not in ASSIGNABLE_CAMERA_NAMES:
        raise ValueError(f"unknown camera stream name {name!r}")
    cameras = _drop_node(dict(sensor_map.get("cameras") or {}), node)
    cameras[name] = node
    out = dict(sensor_map)
    out["cameras"] = cameras
    return out


def assign_arm(
    sensor_map: "dict[str, Any]", role: str, side: str, node: str
) -> "dict[str, Any]":
    """Bind a serial port to a follower or leader side. Pure — unit-tested.

    ``arms`` (followers) and ``leaders`` are independent side namespaces -- a
    follower-right and a leader-right are different arms -- so only entries
    pointing at the SAME port are cleared, never the same side of the other
    namespace.
    """
    if role not in ARM_ROLES:
        raise ValueError(f"unknown arm role {role!r}")
    if side not in ARM_SIDES:
        raise ValueError(f"unknown side {side!r}")
    out = {
        **sensor_map,
        "arms": dict(sensor_map.get("arms") or {}),
        "leaders": dict(sensor_map.get("leaders") or {}),
    }
    _drop_serial_node(out, node)
    if role == "follower":
        out["arms"][side] = node
    else:
        out["leaders"][side] = {"port": node}
    return out


def assign_realsense(
    sensor_map: "dict[str, Any]", serial: str, name: str = "central"
) -> "dict[str, Any]":
    """Bind the central RGB-D camera by serial (it has no stable /dev node)."""
    return {**sensor_map, "realsense": {"serial": serial, "name": name}}


def clear_assignment(
    sensor_map: "dict[str, Any]", kind: str, key: str
) -> "dict[str, Any]":
    """Unbind one camera name, one arm side, or the depth camera. Pure."""
    out = {
        **sensor_map,
        "cameras": dict(sensor_map.get("cameras") or {}),
        "arms": dict(sensor_map.get("arms") or {}),
        "leaders": dict(sensor_map.get("leaders") or {}),
    }
    if kind == "camera":
        out["cameras"].pop(key, None)
    elif kind == "follower":
        out["arms"].pop(key, None)
    elif kind == "leader":
        out["leaders"].pop(key, None)
    elif kind == "realsense":
        out["realsense"] = {}
    else:
        raise ValueError(f"unknown assignment kind {kind!r}")
    return out


def _present(node: "str | None", present_nodes: "set[str]") -> bool:
    """Whether an assigned alias resolves to a device that is here now. Pure."""
    if not node:
        return False
    if node in present_nodes:
        return True
    try:
        real = str(Path(node).resolve(strict=True))
    except OSError:
        return False
    for candidate in present_nodes:
        try:
            if str(Path(candidate).resolve()) == real:
                return True
        except OSError:
            continue
    return False


def map_overview(
    sensor_map: "dict[str, Any]",
    camera_nodes: "list[str]",
    serial_nodes: "list[str]",
    realsense: "list[tuple[str, str]]",
) -> "dict[str, Any]":
    """What the pane lists: every name, its device, and whether it is here. Pure.

    An assignment that no longer resolves to a connected device is the failure
    this view exists to catch -- a camera moved to another socket reads as an
    absent stream at collection time, which is a confusing way to find out.
    """
    present_cameras = set(camera_nodes)
    present_serial = set(serial_nodes)
    cameras = []
    for name in ASSIGNABLE_CAMERA_NAMES:
        node = (sensor_map.get("cameras") or {}).get(name)
        cameras.append(
            {
                "name": name,
                "device": node,
                "present": _present(node, present_cameras),
            }
        )
    arms = []
    for role, section in (("follower", "arms"), ("leader", "leaders")):
        for side in ARM_SIDES:
            entry = (sensor_map.get(section) or {}).get(side)
            node = _serial_node(entry) if entry else None
            arms.append(
                {
                    "role": role,
                    "side": side,
                    "device": node,
                    "present": _present(node, present_serial),
                }
            )
    assigned_serial = (sensor_map.get("realsense") or {}).get("serial")
    return {
        "cameras": cameras,
        "arms": arms,
        "realsense": {
            "serial": assigned_serial,
            "name": (sensor_map.get("realsense") or {}).get("name"),
            "present": bool(assigned_serial)
            and assigned_serial in {s for s, _ in realsense},
        },
        "unassigned_cameras": sorted(
            n
            for n in present_cameras
            if not any(_present(str(c["device"] or ""), {n}) for c in cameras)
        ),
    }


def discover() -> "dict[str, Any]":
    """Every device that is here now. Slow: it opens each capture node."""
    return {
        "cameras": discover_capture_devices(),
        "serial": discover_serial_ports(),
        "realsense": [
            {"serial": s, "name": n} for s, n in discover_realsense_devices()
        ],
    }


def tick_rows(
    positions: "dict[str, Any]", baseline: "dict[str, Any] | None"
) -> "list[dict[str, Any]]":
    """Raw joint ticks and their movement from the baseline. Pure — unit-tested."""
    rows = []
    for joint, value in positions.items():
        delta = int(value) - int((baseline or {}).get(joint, value))
        rows.append(
            {
                "joint": joint,
                "value": int(value),
                "delta": delta,
                "moving": abs(delta) > WIGGLE_TICKS,
            }
        )
    return rows


class ArmProbe:
    """One serial port held open for the wiggle test, torque off.

    An SO-101 bus is identified by moving it: the operator wiggles an arm and
    watches which port's ticks move. The bus is opened UNCALIBRATED and with
    torque disabled -- this reads raw ticks to tell ports apart, it never
    commands an arm.
    """

    def __init__(self) -> None:
        self.port: str | None = None
        self._bus: Any = None
        self._baseline: dict | None = None
        self._positions: dict = {}
        self._lock = threading.Lock()
        self._error: str = ""
        self._read_at: float = 0.0

    def running(self) -> bool:
        return self._bus is not None

    def open(self, port: str) -> "dict[str, Any]":
        from lerobot.motors.feetech import FeetechMotorsBus

        from common.follower_bus import follower_motors

        self.close()
        bus = FeetechMotorsBus(port=port, motors=follower_motors())
        try:
            bus.connect(True)
            bus.disable_torque(num_retry=3)
        except Exception as exc:  # noqa: BLE001 — any failure = not an SO-101 bus
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"{port}: no 6-motor SO-101 bus here ({exc})")
        self._bus = bus
        self.port = port
        self._baseline = None
        return self.read()

    def read(self) -> "dict[str, Any]":
        if self._bus is None:
            return {"port": None, "joints": [], "error": "no port is open"}
        with self._lock:
            try:
                self._positions = self._bus.sync_read(
                    "Present_Position", normalize=False, num_retry=0
                )
                self._read_at = time.monotonic()
                self._error = ""
                if self._baseline is None:
                    self._baseline = dict(self._positions)
            except ConnectionError as exc:
                # Keep showing the last good read: a dropped frame on the bus is
                # not a reason to blank the ticks the operator is watching.
                self._error = str(exc)
            return {
                "port": self.port,
                "joints": tick_rows(self._positions, self._baseline),
                "error": self._error,
            }

    def rebase(self) -> None:
        """Take the current position as the new zero for the wiggle test."""
        with self._lock:
            self._baseline = dict(self._positions)

    def close(self) -> None:
        bus, self._bus = self._bus, None
        self.port = None
        self._baseline = None
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001 — cleanup must not raise
                pass
