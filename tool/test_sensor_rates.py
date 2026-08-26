"""Measure the maximum read frequency of tactile cameras and arm joints.

Reads the visual-tactile USB cameras (UVC, via OpenCV) and the arms'
joint positions (Feetech ``sync_read``) in unpaced tight loops and
reports the achieved rate per sensor — first each sensor alone (its true
ceiling), then all sensors simultaneously (USB/CPU contention check).
Both follower arms and, when connected, the two SO-101 leader arms are
read; leaders are OPTIONAL — an unplugged or unassigned leader is warned
about and skipped, never fatal.

The arms are passive: torque is disabled right after connecting and no
motion is ever commanded.

Devices enumerate in unstable order across replugs (up to four ttyACM
serial ports, several /dev/video nodes), so the tool carries a runtime
**assignment GUI**: on the first run (or with ``--assign``) it shows
each detected camera's live feed — press a gel to identify it, then
keys 1-4 name it (left_arm_left_gripper, ...) — and each serial port's
live raw joint ticks — wiggle an arm to identify it, then assign it a
role: follower right/left, or leader right/left. Assignments are saved to
``src/conf/sensor_map.yaml`` (per-machine, gitignored) and reused on
later runs. A leader's calibration id is fixed by side (LEADER_ID_LEFT /
LEADER_ID_RIGHT in robot.yaml), so assignment stores only its port.
Follower joints show as calibrated degrees; leader joints show as
calibrated degrees too (via that side's leader calibration).

Usage:

    venv/bin/python tool/test_sensor_rates.py            # first run: GUI
    venv/bin/python tool/test_sensor_rates.py --view     # live window
    venv/bin/python tool/test_sensor_rates.py --assign   # redo the GUI
    venv/bin/python tool/test_sensor_rates.py --arm right

Manual override (skips the map/GUI for cameras):

    venv/bin/python tool/test_sensor_rates.py \\
        --camera left_arm_left_gripper=/dev/video4 \\
        --camera left_arm_right_gripper=/dev/video6

Find raw device nodes with ``--list-cameras``. Tips: if a camera caps
at ~5-10 Hz at 640x480 it is likely delivering uncompressed YUYV —
retry with ``--fourcc MJPG``. ``v4l2-ctl --list-formats-ext -d
/dev/videoN`` shows what the device supports.
"""

import argparse
import glob
import os
import struct
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import cv2  # type: ignore[import]
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from common.sensor_view import quiet_qt_warnings  # noqa: E402

# Stream names offered by the --assign GUI, each bound to a stable
# /dev/v4l/by-path node in sensor_map.yaml. The four tactile gripper cameras
# plus the two follower wrist cameras; the wrist names match the recorder's
# dataset feature keys (observation.images.wrist_camera_*), so a by-path
# assignment here also drives the recording stack (build_recording_stack
# overlays these nodes onto recording.yaml).
ASSIGNABLE_CAMERA_NAMES = [
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
    "wrist_camera_left",
    "wrist_camera_right",
    "central",
]
# Which of the assignable streams are the tactile gel cameras. One list, so that
# --tactile, the recorder and the console cannot disagree about what "tactile"
# means; the names also carry the wiring (which arm, which finger), which is what
# an operator needs when one of four identical-looking streams looks wrong.
TACTILE_CAMERA_NAMES = (
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)
SENSOR_MAP_PATH = _root / "src/conf/sensor_map.yaml"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fourcc_str(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip()


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────


def parse_camera_spec(spec: str) -> tuple[str, "int | str"]:
    """``"name=/dev/video4"`` -> (name, dev); bare ``"4"`` -> auto name."""
    if "=" in spec:
        name, _, dev = spec.partition("=")
        name, dev = name.strip(), dev.strip()
        if not name or not dev:
            raise ValueError(f"bad camera spec {spec!r} (want NAME=DEV)")
    else:
        name, dev = "", spec.strip()
        if not dev:
            raise ValueError("empty camera spec")
    device: int | str = int(dev) if dev.lstrip("-").isdigit() else dev
    return name or f"camera[{dev}]", device


def stable_device_path(node: "int | str", dev_root: "str | Path" = "/dev") -> str:
    """Prefer a replug-stable alias for a /dev node.

    by-path (stable per physical USB socket — identical devices often
    lack unique serials, so by-id can be ambiguous) is tried first,
    then by-id, else the raw node is returned.
    """
    if isinstance(node, int):
        node = f"/dev/video{node}"
    try:
        real = Path(node).resolve(strict=True)
    except OSError:
        return str(node)
    for sub in ("serial/by-path", "v4l/by-path", "serial/by-id", "v4l/by-id"):
        directory = Path(dev_root) / sub
        if not directory.is_dir():
            continue
        for link in sorted(directory.iterdir()):
            try:
                if link.resolve() == real:
                    return str(link)
            except OSError:
                continue
    return str(node)


def load_sensor_map(path: Path) -> dict:
    """Read the saved assignment map; missing sections become empty.

    ``arms`` holds follower ports (``{side: node}``); ``leaders`` holds
    the optional leader arms (``{side: {port: node}}`` — the calibration
    id is fixed by side in robot.yaml, not stored here). Old maps without
    a ``leaders`` section load as ``{}`` (leaders off); a legacy ``id``
    key is loaded but ignored.
    """
    data = yaml.safe_load(path.read_text()) or {}
    return {
        "cameras": dict(data.get("cameras") or {}),
        "arms": dict(data.get("arms") or {}),
        "leaders": dict(data.get("leaders") or {}),
        "realsense": dict(data.get("realsense") or {}),
    }


def save_sensor_map(path: Path, sensor_map: dict) -> None:
    body = yaml.safe_dump(
        {
            "cameras": sensor_map["cameras"],
            "arms": sensor_map["arms"],
            "leaders": sensor_map.get("leaders", {}),
            "realsense": sensor_map.get("realsense", {}),
        },
        sort_keys=True,
    )
    path.write_text(
        "# Sensor assignments written by tool/test_sensor_rates.py.\n"
        "# Per-machine (gitignored) — re-run with --assign to redo.\n"
        "# arms: follower ports; leaders: optional leader {port};\n"
        "# realsense: central RGB-D device {serial, name}.\n" + body
    )


def grid_tiles(tiles: "list[np.ndarray]", max_per_row: int = 3) -> np.ndarray:
    """Lay tiles out in rows of at most ``max_per_row``, padding to align."""
    rows = [tiles[i : i + max_per_row] for i in range(0, len(tiles), max_per_row)]
    row_imgs = []
    for row in rows:
        height = max(t.shape[0] for t in row)
        row = [
            (
                t
                if t.shape[0] == height
                else cv2.resize(t, (int(t.shape[1] * height / t.shape[0]), height))
            )
            for t in row
        ]
        row_imgs.append(np.hstack(row))
    width = max(r.shape[1] for r in row_imgs)
    padded = [
        (
            r
            if r.shape[1] == width
            else np.hstack(
                [r, np.zeros((r.shape[0], width - r.shape[1], 3), dtype=r.dtype)]
            )
        )
        for r in row_imgs
    ]
    return np.vstack(padded)


# ── Probes ───────────────────────────────────────────────────────────────────


class SensorProbe:
    """Base: a thread reading one sensor as fast as it can, timestamping."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stamps: list[float] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def read_once(self) -> bool:
        """One blocking read. Returns success."""
        raise NotImplementedError

    def reset(self) -> None:
        self.stamps = []
        self.errors = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ok = self.read_once()
            except ConnectionError:
                ok = False
            if ok:
                self.stamps.append(time.monotonic())
            else:
                self.errors += 1
                time.sleep(0.005)  # don't spin on a dead sensor

    def summary(self) -> str:
        n = len(self.stamps)
        if n < 2:
            return f"{self.name:<24} reads={n} errors={self.errors} — too few reads"
        span = self.stamps[-1] - self.stamps[0]
        hz = (n - 1) / span
        iv_ms = np.diff(np.asarray(self.stamps)) * 1000.0
        p50, p95, p99 = np.percentile(iv_ms, [50, 95, 99])
        return (
            f"{self.name:<24} {hz:7.1f} Hz  reads={n:<6d} "
            f"interval p50={p50:.1f} p95={p95:.1f} p99={p99:.1f} "
            f"max={iv_ms.max():.1f} ms  errors={self.errors}"
        )


class CameraProbe(SensorProbe):
    """Unpaced ``VideoCapture.read()`` loop — blocks until each new frame."""

    def __init__(
        self,
        name: str,
        device: "int | str",
        width: int,
        height: int,
        fps: int,
        fourcc: str,
    ) -> None:
        super().__init__(name)
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap: cv2.VideoCapture | None = None
        self.last_frame: np.ndarray | None = None

    def open(self) -> None:
        # Pin the V4L2 backend: the default fallback chain can silently open
        # a DIFFERENT physical camera via FFMPEG when the index is a
        # metadata node, breaking the index<->device mapping.
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise SystemExit(
                f"❌ cannot open camera {self.name} ({self.device}) "
                "(check --list-cameras / replug / re-run with --assign)"
            )
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap
        print(
            f"  📷 {self.name} ({self.device}): negotiated "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps "
            f"fourcc={_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) or '?'}"
        )

    def read_once(self) -> bool:
        assert self.cap is not None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.last_frame = frame  # BGR; reference swap is GIL-atomic
            return True
        return False

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class JointProbe(SensorProbe):
    """Unpaced Feetech ``sync_read`` of all six Present_Position values."""

    def __init__(self, name: str, bus) -> None:
        super().__init__(name)
        self.bus = bus
        self.last_positions: dict[str, float] | None = None

    def read_once(self) -> bool:
        # num_retry=0: a dropped status packet counts as an error tick
        # instead of hiding inside a retried (slower) read.
        positions = self.bus.sync_read("Present_Position", num_retry=0)
        if len(positions) == 6:
            self.last_positions = positions
            return True
        return False


# ── Device discovery ─────────────────────────────────────────────────────────


def _video_sort_key(dev: str) -> int:
    digits = "".join(c for c in dev if c.isdigit())
    return int(digits) if digits else 0


# VIDIOC_QUERYCAP, and the capability bit that says "this node produces video".
# Every UVC camera exposes a second, adjacent node carrying only per-frame
# metadata: it opens like a camera and then never grabs. Asking the driver what a
# node IS costs one ioctl and no stream negotiation, where opening every node to
# find out costs a capture attempt on each -- roughly half of them doomed -- and
# a line of OpenCV warning output apiece.
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
# struct v4l2_capability: driver[16] card[32] bus_info[32] version[u32]
# capabilities[u32] device_caps[u32] reserved[3*u32].
_QUERYCAP_STRUCT = "16s32s32sIII3I"


def is_capture_node(dev: str) -> bool:
    """Whether a /dev/video node can capture video at all. No stream opened.

    ``device_caps`` describes this node specifically, where ``capabilities``
    describes the whole physical device -- so a metadata node of a camera that
    can capture reports the capture bit in the second and not the first, and only
    the second answers the question being asked here.
    """
    import fcntl

    buf = bytearray(struct.calcsize(_QUERYCAP_STRUCT))
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return False
    finally:
        os.close(fd)
    _driver, _card, _bus, _ver, caps, device_caps, *_ = struct.unpack(
        _QUERYCAP_STRUCT, bytes(buf)
    )
    return bool((device_caps or caps) & _V4L2_CAP_VIDEO_CAPTURE)


def capture_nodes() -> list[str]:
    """Every /dev/video node the driver calls a capture device. Opens nothing."""
    return [
        dev
        for dev in sorted(glob.glob("/dev/video*"), key=_video_sort_key)
        if is_capture_node(dev)
    ]


def discover_capture_devices() -> list[str]:
    """The /dev/video nodes that actually deliver frames (V4L2 + grab).

    Two stages, cheap first: ask the driver which nodes are capture devices at
    all, then actually grab from those to weed out the ones that are busy or
    broken. Only the second stage opens a stream.
    """
    devices = []
    for dev in capture_nodes():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened() and cap.grab():
            devices.append(dev)
        cap.release()
    return devices


def discover_serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))


def discover_realsense_devices() -> list:
    """Connected RealSense devices as ``(serial, name)`` pairs.

    The RealSense is identified by its globally-unique serial (stable across
    replug), not a /dev/video node. ``pyrealsense2`` is imported lazily so the
    tool runs on machines without it; any failure yields an empty list rather
    than raising (RealSense is optional).
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import]

        out = []
        for dev in rs.context().query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out.append((serial, name))
        return sorted(out)
    except Exception:  # noqa: BLE001 — no pyrealsense2, no permissions, no device
        return []


def select_realsense_serial(devices: list, prefer: "str | None" = None) -> "str | None":
    """Pick one serial from ``(serial, name)`` devices. Pure — unit-tested.

    Empty → None; a single device → its serial; several → ``prefer`` when it is
    still connected, otherwise the first (the caller warns). ``prefer`` lets a
    re-run keep the previously-assigned device when several are attached.
    """
    serials = [s for s, _ in devices]
    if not serials:
        return None
    if prefer and prefer in serials:
        return prefer
    return serials[0]


def list_cameras() -> None:
    devices = sorted(glob.glob("/dev/video*"), key=_video_sort_key)
    if not devices:
        print("no /dev/video* devices found")
    for dev in devices:
        if not is_capture_node(dev):
            print(f"  {dev}: not a capture device (metadata node)")
            continue
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened() and cap.grab():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {dev}: capture device ({w}x{h} default)")
        else:
            print(f"  {dev}: capture device, but busy or not delivering")
        cap.release()
    rs_devices = discover_realsense_devices()
    if rs_devices:
        for serial, name in rs_devices:
            print(f"  RealSense: {name} (serial {serial})")
    else:
        print("  no RealSense devices found (or pyrealsense2 unavailable)")


# ── Assignment GUI ───────────────────────────────────────────────────────────

_ASSIGN_WINDOW = "assign sensors"


def _put_lines(img: np.ndarray, lines: list) -> None:
    """Draw ``(text, colour)`` pairs top-down with a thin shadow box."""
    for i, (text, colour) in enumerate(lines):
        y = 24 + 26 * i
        cv2.putText(img, text, (8, y), _FONT, 0.55, (0, 0, 0), 4)
        cv2.putText(img, text, (8, y), _FONT, 0.55, colour, 1)


def _drop_node(assigned: dict, node: str) -> dict:
    """Remove entries already pointing at ``node`` (reassignment)."""
    real = Path(node).resolve()
    return {k: v for k, v in assigned.items() if Path(v).resolve() != real}


def _assign_cameras(devices: list, existing: dict) -> dict:
    assigned = dict(existing)
    for i, dev in enumerate(devices):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                lines = [
                    (f"camera {i + 1}/{len(devices)}: {dev}", (0, 255, 0)),
                    ("press a gel to identify this camera", (255, 255, 255)),
                ]
                for k, name in enumerate(ASSIGNABLE_CAMERA_NAMES):
                    node = assigned.get(name)
                    tag = f"  [{node}]" if node else ""
                    colour = (0, 255, 255) if node else (255, 255, 255)
                    lines.append((f" {k + 1}  {name}{tag}", colour))
                lines.append((" s skip   q finish cameras", (255, 255, 255)))
                _put_lines(frame, lines)
                cv2.imshow(_ASSIGN_WINDOW, frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("s"):
                    break
                if key in (27, ord("q")):
                    return assigned
                if ord("1") <= key <= ord(str(len(ASSIGNABLE_CAMERA_NAMES))):
                    name = ASSIGNABLE_CAMERA_NAMES[key - ord("1")]
                    node = stable_device_path(dev)
                    assigned = _drop_node(assigned, node)
                    assigned[name] = node
                    print(f"  ✓ {name} = {node}")
                    break
        finally:
            cap.release()
    return assigned


def _serial_node(entry: "str | dict") -> str:
    """The /dev node of a serial role entry (follower str or leader dict)."""
    return entry["port"] if isinstance(entry, dict) else entry


def _drop_serial_node(sensor_map: dict, node: str) -> None:
    """Remove any follower OR leader role currently pointing at ``node``.

    A physical port is one arm, so reassigning it must clear it from
    wherever it lived — but ``arms`` and ``leaders`` are independent side
    namespaces (a follower-right and a leader-right are different arms),
    so only same-*node* entries are dropped, never same-*side*.
    """
    real = Path(node).resolve()

    def keep(entry: "str | dict") -> bool:
        try:
            return Path(_serial_node(entry)).resolve() != real
        except OSError:
            return True

    sensor_map["arms"] = {s: v for s, v in sensor_map["arms"].items() if keep(v)}
    sensor_map["leaders"] = {s: v for s, v in sensor_map["leaders"].items() if keep(v)}


def _tick_lines(positions: dict, baseline: "dict | None") -> list:
    """``(text, colour)`` rows for the live raw joint ticks (wiggle test)."""
    lines = []
    for joint, value in positions.items():
        delta = int(value) - int((baseline or {}).get(joint, value))
        colour = (0, 255, 0) if abs(delta) > 5 else (200, 200, 200)
        lines.append((f" {joint:<14}{int(value):>6}  d={delta:+d}", colour))
    return lines


def _assign_serial(ports: list, sensor_map: dict) -> None:
    """Assign each serial port to a role: follower or leader, right/left.

    Wiggle an arm to see which port's ticks move, then press a role key.
    A leader role stores only the port; its calibration id is fixed by
    side in robot.yaml. Mutates ``sensor_map`` in place ("arms" =
    followers, "leaders" = {side: {port}}).
    """
    from lerobot.motors.feetech import FeetechMotorsBus

    from common.follower_bus import follower_motors

    for i, port in enumerate(ports):
        # Uncalibrated identification bus: raw ticks only, torque off.
        bus = FeetechMotorsBus(port=port, motors=follower_motors())
        try:
            bus.connect(True)
            bus.disable_torque(num_retry=3)
        except Exception as e:  # noqa: BLE001 — any failure = not an SO-101 bus
            print(f"  {port}: no 6-motor bus ({e}) — skipped")
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001
                pass
            continue
        try:
            if _interact_serial_port(bus, port, i, len(ports), sensor_map):
                return  # user pressed q — finish the whole serial step
        finally:
            try:
                bus.disconnect()
            except Exception as e:  # noqa: BLE001 — cleanup must not raise
                print(f"⚠️  {port} disconnect failed: {e}")


def _interact_serial_port(bus, port: str, i: int, n: int, sensor_map: dict) -> bool:
    """Drive the assign menu for one port. Returns True iff the user quit.

    A single ``waitKey`` role menu: the raw joint ticks stream live so
    the wiggle stays visible while the operator picks a role. A leader
    role stores only the port (its calibration id is fixed by side in
    robot.yaml).
    """
    baseline: dict | None = None
    positions: dict = {}
    while True:
        try:
            positions = bus.sync_read("Present_Position", normalize=False, num_retry=0)
            if baseline is None:
                baseline = dict(positions)
        except ConnectionError:
            pass  # keep showing the last good read
        panel = np.zeros((480, 640, 3), dtype=np.uint8)
        lines: list = [
            (f"serial port {i + 1}/{n}: {port}", (0, 255, 0)),
            ("wiggle ONE arm - watch the ticks", (255, 255, 255)),
            ("", (255, 255, 255)),
        ]
        lines += _tick_lines(positions, baseline)
        lines.append(("", (255, 255, 255)))
        lines += [
            (" 1 follower right    2 follower left", (255, 255, 255)),
            (" 3 leader right      4 leader left", (255, 255, 255)),
            (" s skip   q finish", (255, 255, 255)),
        ]
        _put_lines(panel, lines)
        cv2.imshow(_ASSIGN_WINDOW, panel)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            return False
        if key in (27, ord("q")):
            return True
        node = stable_device_path(port)
        if key in (ord("1"), ord("2")):
            side = "right" if key == ord("1") else "left"
            _drop_serial_node(sensor_map, node)
            sensor_map["arms"][side] = node
            print(f"  ✓ follower {side} = {node}")
            return False
        if key in (ord("3"), ord("4")):
            side = "right" if key == ord("3") else "left"
            _drop_serial_node(sensor_map, node)
            sensor_map["leaders"][side] = {"port": node}
            print(f"  ✓ leader {side} = {node}")
            return False


def run_assignment(sensor_map: dict) -> dict:
    """Interactive OpenCV assignment of cameras and arm serial ports."""
    print("\n▶ sensor assignment — work in the OpenCV window")
    devices = discover_capture_devices()
    if devices:
        print(f"  cameras: {len(devices)} capture device(s) found")
        sensor_map["cameras"] = _assign_cameras(devices, sensor_map["cameras"])
    else:
        print("  no capture devices found — camera step skipped")
    ports = discover_serial_ports()
    if ports:
        print(f"  serial: probing {len(ports)} port(s) for follower/leader buses")
        _assign_serial(ports, sensor_map)
    else:
        print("  no serial ports found — arm step skipped")
    _assign_realsense(sensor_map)
    cv2.destroyAllWindows()
    return sensor_map


def _assign_realsense(sensor_map: dict) -> None:
    """Detect the central RealSense and store its serial (no gel identify).

    The RealSense is keyed by its stable serial, so assignment is just
    detection: pick the connected device (keeping the previously-assigned one
    when several are attached) and save ``{serial, name}``. Skipped with a note
    when none is present or ``pyrealsense2`` is unavailable.
    """
    rs_devices = discover_realsense_devices()
    if not rs_devices:
        print("  no RealSense found — central-camera step skipped")
        return
    prev = (sensor_map.get("realsense") or {}).get("serial") or None
    serial = select_realsense_serial(rs_devices, prefer=prev)
    name_by_serial = dict(rs_devices)
    if len(rs_devices) > 1:
        print(f"  ⚠️  {len(rs_devices)} RealSense devices — using serial {serial}")
    sensor_map["realsense"] = {"serial": serial, "name": "central"}
    print(f"  ✓ central RealSense = {name_by_serial.get(serial, '?')} ({serial})")


# ── Measurement phases ───────────────────────────────────────────────────────


def _live_hz(probe: SensorProbe) -> float:
    """Reads in the last second (cheap live-rate estimate)."""
    now = time.monotonic()
    recent = [s for s in probe.stamps[-1200:] if s > now - 1.0]
    return float(len(recent))


# One cell per side, wide enough for two value columns in a large font.
_SIDE_PANEL_W = 640
_SIDE_PANEL_H = 460
_BODY_AND_GRIP = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _probe_side(name: str) -> str:
    return "left" if "left" in name else "right"


def _probe_role(name: str) -> str:
    return "leader" if name.startswith("leader") else "follower"


def _fmt_joint_value(joints: "dict | None", joint: str) -> str:
    """Right-aligned value string for one joint, or ``--`` when absent/None."""
    if joints is None or joints.get(joint) is None:
        return "   --"
    return f"{joints[joint]:7.2f}"


def _side_panel(
    side: str,
    c1_label: str,
    c1_joints: "dict | None",
    c1_hz: "float | None",
    c2_label: str,
    c2_joints: "dict | None",
    c2_hz: "float | None",
) -> np.ndarray:
    """One cell for a side: two labelled joint columns in a large font.

    ``cN_joints`` is a ``{joint_name: value | None}`` dict (or None); the
    columns are data, not probes, so both the sensor-rate tool and the
    teleop view (``common.sensor_view``) render identically.
    """
    panel = np.zeros((_SIDE_PANEL_H, _SIDE_PANEL_W, 3), dtype=np.uint8)
    cv2.putText(panel, f"{side.upper()} ARM", (15, 46), _FONT, 1.2, (0, 255, 0), 3)

    # Column headers + live Hz under each.
    cv2.putText(panel, c1_label, (300, 92), _FONT, 0.9, (0, 200, 255), 2)
    cv2.putText(panel, c2_label, (490, 92), _FONT, 0.9, (0, 200, 255), 2)
    h1 = f"{c1_hz:.0f} Hz" if c1_hz is not None else "--"
    h2 = f"{c2_hz:.0f} Hz" if c2_hz is not None else "--"
    cv2.putText(panel, h1, (300, 124), _FONT, 0.7, (150, 150, 150), 1)
    cv2.putText(panel, h2, (490, 124), _FONT, 0.7, (150, 150, 150), 1)

    order = list(c1_joints or {}) or list(c2_joints or {}) or _BODY_AND_GRIP
    for i, joint in enumerate(order):
        y = 172 + 48 * i
        cv2.putText(panel, joint, (15, y), _FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(
            panel,
            _fmt_joint_value(c1_joints, joint),
            (300, y),
            _FONT,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            panel,
            _fmt_joint_value(c2_joints, joint),
            (490, y),
            _FONT,
            0.8,
            (255, 255, 255),
            2,
        )
    return panel


def _joint_grid(joints: "list[JointProbe]") -> "np.ndarray | None":
    """One cell per side (follower + leader together), sides side by side.

    Columns within a cell are follower then leader; a missing arm shows
    ``--``. A side with no arms at all is dropped.
    """
    by_key = {(_probe_role(jp.name), _probe_side(jp.name)): jp for jp in joints}
    sides = [
        s
        for s in ("left", "right")
        if any((r, s) in by_key for r in ("follower", "leader"))
    ]
    if not sides:
        return None

    def col(jp: "JointProbe | None") -> "tuple[dict | None, float | None]":
        return (jp.last_positions, _live_hz(jp)) if jp is not None else (None, None)

    panels = []
    for s in sides:
        fj, fhz = col(by_key.get(("follower", s)))
        lj, lhz = col(by_key.get(("leader", s)))
        panels.append(_side_panel(s, "follower", fj, fhz, "leader", lj, lhz))
    return np.hstack(panels)


def _vstack_pad(blocks: "list[np.ndarray]") -> np.ndarray:
    """Stack blocks vertically, right-padding narrower ones to equal width."""
    width = max(b.shape[1] for b in blocks)
    padded = [
        (
            b
            if b.shape[1] == width
            else np.hstack(
                [b, np.zeros((b.shape[0], width - b.shape[1], 3), dtype=b.dtype)]
            )
        )
        for b in blocks
    ]
    return np.vstack(padded)


def _fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Downscale (never upscale) ``img`` to fit within ``max_w`` x ``max_h``."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)))


# Display size of each camera tile in the live view (four in a row ≈ 1280
# wide, so the composite fits a laptop screen without shrinking the labels).
_CAM_TILE_W = 320
_CAM_TILE_H = 240


def _camera_short_label(name: str) -> str:
    """Compact camera name for the tile overlay (e.g. ``left-left``,
    ``wrist-left``)."""
    return (
        name.replace("_arm", "")
        .replace("_gripper", "")
        .replace("_camera", "")
        .replace("_", "-")
    )


def run_view(
    probes: list[SensorProbe],
    cameras: "list[CameraProbe]",
    joints: "list[JointProbe]",
    max_w: int = 1280,
    max_h: int = 720,
) -> None:
    """Live window: a row of tactile cameras above a row of arm cells.

    Top row: the tactile cameras in one row (name-sorted → left-left,
    left-right, right-left, right-right). Bottom row: one cell per side
    (left | right), each showing that side's follower and leader joints
    in two columns. Runs until q/Esc; the probes keep free-running
    underneath, so the rate summary printed afterwards reflects the same
    contention as the headless simultaneous phase.
    """
    print("\n▶ live view — press q or Esc in the window to stop")
    for p in probes:
        p.reset()
        p.start()
    try:
        while True:
            cam_tiles = []
            for cam in cameras:
                frame = cam.last_frame
                if frame is None:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                frame = cv2.resize(frame, (_CAM_TILE_W, _CAM_TILE_H))
                cv2.putText(
                    frame,
                    f"{_camera_short_label(cam.name)} {_live_hz(cam):.0f}Hz",
                    (6, 26),
                    _FONT,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cam_tiles.append(frame)
            blocks = []
            if cam_tiles:
                # One row of cameras; the sensor map is name-sorted, so the
                # order is left-left, left-right, right-left, right-right.
                blocks.append(grid_tiles(cam_tiles, max_per_row=max(len(cam_tiles), 1)))
            joint_grid = _joint_grid(joints)
            if joint_grid is not None:
                blocks.append(joint_grid)
            if not blocks:
                break
            cv2.imshow(
                "sensor rates", _fit_to_screen(_vstack_pad(blocks), max_w, max_h)
            )
            key = cv2.waitKey(33) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for p in probes:
            p.stop()
        cv2.destroyAllWindows()
    print("\nrates during the live view:")
    for p in probes:
        print("  " + p.summary())


def run_phase(label: str, probes: list[SensorProbe], duration: float) -> None:
    print(f"\n▶ {label} — {duration:.0f} s")
    for p in probes:
        p.reset()
        p.start()
    time.sleep(duration)
    for p in probes:
        p.stop()
    for p in probes:
        print("  " + p.summary())


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    quiet_qt_warnings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        choices=["right", "left", "both", "none"],
        default="both",
        help="which follower arm(s) to read (default both; none=cameras only)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="[NAME=]DEV",
        help="camera override, e.g. left_arm_left_gripper=/dev/video4 "
        "(repeatable; bare DEV also accepted; skips the saved map/GUI)",
    )
    parser.add_argument(
        "--assign",
        action="store_true",
        help="re-run the sensor-assignment GUI and update src/conf/sensor_map.yaml",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--request-fps",
        type=int,
        default=0,
        help="fps to request from the driver (0 = leave the device default)",
    )
    parser.add_argument(
        "--fourcc",
        default="",
        help='pixel format to request, e.g. "MJPG" (empty = device default)',
    )
    parser.add_argument(
        "--solo",
        type=float,
        default=5.0,
        help="seconds of per-sensor solo measurement (0 = skip solo phases)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="seconds of simultaneous all-sensors measurement",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="probe /dev/video* and exit"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="live window with camera feeds + joint readouts instead of the "
        "timed phases (q/Esc to stop; rates reported on exit)",
    )
    parser.add_argument(
        "--no-leaders",
        action="store_true",
        help="skip the leader arms even if assigned (leaders are optional; "
        "an unplugged/failed leader is warned about and skipped anyway)",
    )
    parser.add_argument(
        "--view-max-width",
        type=int,
        default=1280,
        help="max width of the --view window; the composite is scaled down "
        "to fit (default 1280)",
    )
    parser.add_argument(
        "--view-max-height",
        type=int,
        default=720,
        help="max height of the --view window (default 720)",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    from common.follower_bus import (
        connect_follower_bus,
        connect_leader_bus,
        leader_calib_id_for_side,
    )

    # Resolve sensor assignments: CLI --camera wins; otherwise the saved
    # map; no map and no --camera => the assignment GUI runs.
    sensor_map = (
        load_sensor_map(SENSOR_MAP_PATH)
        if SENSOR_MAP_PATH.exists()
        else {"cameras": {}, "arms": {}, "leaders": {}}
    )
    if args.assign or (not SENSOR_MAP_PATH.exists() and not args.camera):
        sensor_map = run_assignment(sensor_map)
        save_sensor_map(SENSOR_MAP_PATH, sensor_map)
        print(f"  💾 assignments saved to {SENSOR_MAP_PATH}")

    camera_specs: "list[tuple[str, int | str]]"
    if args.camera:
        camera_specs = [parse_camera_spec(spec) for spec in args.camera]
    else:
        camera_specs = sorted(sensor_map["cameras"].items())

    sides = {"both": ["right", "left"], "right": ["right"], "left": ["left"]}.get(
        args.arm, []
    )
    if not camera_specs and not sides:
        parser.error(
            "nothing to test: assign sensors with --assign, or give "
            "--camera and/or an --arm"
        )

    probes: list[SensorProbe] = []
    cameras: list[CameraProbe] = []
    joints: list[JointProbe] = []
    buses = []
    try:
        for name, device in camera_specs:
            # Cameras are OPTIONAL in the view: a stream that is unplugged
            # (its by-path node is gone) or won't open warns and is skipped,
            # so the view still shows whatever IS connected. Only an explicit
            # --camera override that fails is worth an abort (handled below).
            if isinstance(device, str) and not Path(device).exists():
                print(f"⚠️  camera {name} device {device} missing — skipped")
                continue
            cam = CameraProbe(
                name, device, args.width, args.height, args.request_fps, args.fourcc
            )
            try:
                cam.open()
            except SystemExit as e:
                print(f"⚠️  {e}")
                continue
            cameras.append(cam)
            probes.append(cam)

        for side in sides:
            port = sensor_map["arms"].get(side)
            if port is not None and not Path(port).exists():
                raise SystemExit(
                    f"❌ {side} arm port {port} is missing — "
                    "replug or re-run with --assign"
                )
            try:
                bus = connect_follower_bus(side, port=port)
            except Exception as e:  # noqa: BLE001 — name the failing side
                raise SystemExit(
                    f"❌ {side} arm failed to connect "
                    f"({port or 'robot.yaml default port'}): {e}\n"
                    "   Power the arm on, or restrict with --arm, or "
                    "re-run with --assign."
                ) from e
            buses.append(bus)
            jp = JointProbe(f"follower[{side}]", bus)
            joints.append(jp)
            probes.append(jp)

        # Leaders are OPTIONAL: any problem (unassigned/unplugged/no
        # calibration/handshake fail) warns and skips — never aborts.
        if not args.no_leaders:
            for side, entry in sorted(sensor_map.get("leaders", {}).items()):
                port = entry.get("port") if isinstance(entry, dict) else None
                if not port:
                    print(f"⚠️  leader {side} not assigned — skipped")
                    continue
                calib_id = leader_calib_id_for_side(side)
                if not Path(port).exists():
                    print(f"⚠️  leader {side} port {port} missing — skipped")
                    continue
                try:
                    bus = connect_leader_bus(port, calib_id)
                except Exception as e:  # noqa: BLE001 — leaders are optional
                    print(f"⚠️  leader {side} ({calib_id}) failed to connect: {e}")
                    continue
                buses.append(bus)
                jp = JointProbe(f"leader[{side}]", bus)
                joints.append(jp)
                probes.append(jp)

        if args.view:
            run_view(
                probes,
                cameras,
                joints,
                max_w=args.view_max_width,
                max_h=args.view_max_height,
            )
        else:
            if args.solo > 0:
                for p in probes:
                    run_phase(f"solo: {p.name}", [p], args.solo)
            run_phase("simultaneous: all sensors", probes, args.duration)
            print(
                "\n(solo = per-sensor ceiling; simultaneous = with USB/serial "
                "contention. Cameras are frame-rate-bound: the Hz above is "
                "what the device actually delivers.)"
            )
    finally:
        for cam in cameras:
            cam.close()
        for bus in buses:
            try:
                bus.disconnect()
            except Exception as e:  # noqa: BLE001 — cleanup must not raise
                print(f"⚠️  bus disconnect failed: {e}")


if __name__ == "__main__":
    main()
