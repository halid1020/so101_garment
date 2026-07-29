"""Measure the maximum read frequency of tactile cameras and arm joints.

Reads the visual-tactile USB cameras (UVC, via OpenCV) and the follower
arms' joint positions (Feetech ``sync_read``) in unpaced tight loops and
reports the achieved rate per sensor — first each sensor alone (its true
ceiling), then all sensors simultaneously (USB/CPU contention check).

The arms are passive: torque is disabled right after connecting and no
motion is ever commanded.

Devices enumerate in unstable order across replugs (four ttyACM serial
ports, several /dev/video nodes), so the tool carries a runtime
**assignment GUI**: on the first run (or with ``--assign``) it shows
each detected camera's live feed — press a gel to identify it, then
keys 1-4 name it (left_arm_left_gripper, ...) — and each serial port's
live raw joint ticks — wiggle a follower arm to identify it, then r/l
assigns right/left. Assignments are saved to ``src/conf/sensor_map.yaml``
(per-machine, gitignored) and reused on later runs.

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
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import cv2  # type: ignore[import]  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

GRIPPER_CAMERA_NAMES = [
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
]
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
    """Read the saved assignment map; missing sections become empty."""
    data = yaml.safe_load(path.read_text()) or {}
    return {
        "cameras": dict(data.get("cameras") or {}),
        "arms": dict(data.get("arms") or {}),
    }


def save_sensor_map(path: Path, sensor_map: dict) -> None:
    body = yaml.safe_dump(
        {"cameras": sensor_map["cameras"], "arms": sensor_map["arms"]},
        sort_keys=True,
    )
    path.write_text(
        "# Sensor assignments written by tool/test_sensor_rates.py.\n"
        "# Per-machine (gitignored) — re-run with --assign to redo.\n" + body
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


def discover_capture_devices() -> list[str]:
    """The /dev/video nodes that actually deliver frames (V4L2 + grab)."""
    devices = []
    for dev in sorted(glob.glob("/dev/video*"), key=_video_sort_key):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened() and cap.grab():
            devices.append(dev)
        cap.release()
    return devices


def discover_serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))


def list_cameras() -> None:
    devices = sorted(glob.glob("/dev/video*"), key=_video_sort_key)
    if not devices:
        print("no /dev/video* devices found")
        return
    for dev in devices:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened() and cap.grab():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {dev}: capture device ({w}x{h} default)")
        else:
            print(f"  {dev}: not a capture device (metadata node or busy)")
        cap.release()


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
                for k, name in enumerate(GRIPPER_CAMERA_NAMES):
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
                if ord("1") <= key <= ord(str(len(GRIPPER_CAMERA_NAMES))):
                    name = GRIPPER_CAMERA_NAMES[key - ord("1")]
                    node = stable_device_path(dev)
                    assigned = _drop_node(assigned, node)
                    assigned[name] = node
                    print(f"  ✓ {name} = {node}")
                    break
        finally:
            cap.release()
    return assigned


def _assign_arms(ports: list, existing: dict) -> dict:
    from lerobot.motors.feetech import FeetechMotorsBus

    from common.follower_bus import follower_motors

    assigned = dict(existing)
    for i, port in enumerate(ports):
        # Uncalibrated identification bus: raw ticks only, torque off.
        bus = FeetechMotorsBus(port=port, motors=follower_motors())
        try:
            bus.connect(True)
            bus.disable_torque(num_retry=3)
        except Exception as e:  # noqa: BLE001 — any failure = not a follower
            print(f"  {port}: no 6-motor bus ({e}) — skipped")
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001
                pass
            continue
        baseline: dict | None = None
        positions: dict = {}
        try:
            while True:
                try:
                    positions = bus.sync_read(
                        "Present_Position", normalize=False, num_retry=0
                    )
                    if baseline is None:
                        baseline = dict(positions)
                except ConnectionError:
                    pass  # keep showing the last good read
                panel = np.zeros((480, 640, 3), dtype=np.uint8)
                lines = [
                    (f"serial port {i + 1}/{len(ports)}: {port}", (0, 255, 0)),
                    ("wiggle ONE follower arm - watch the ticks", (255, 255, 255)),
                    ("", (255, 255, 255)),
                ]
                for joint, value in positions.items():
                    delta = int(value) - int((baseline or {}).get(joint, value))
                    moving = abs(delta) > 5
                    colour = (0, 255, 0) if moving else (200, 200, 200)
                    lines.append((f" {joint:<14}{int(value):>6}  d={delta:+d}", colour))
                lines.append(("", (255, 255, 255)))
                for side in ("right", "left"):
                    node = assigned.get(side)
                    if node:
                        lines.append((f" {side} arm = {node}", (0, 255, 255)))
                lines.append(
                    (" r right arm   l left arm   s skip   q finish", (255, 255, 255))
                )
                _put_lines(panel, lines)
                cv2.imshow(_ASSIGN_WINDOW, panel)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("s"):
                    break
                if key in (27, ord("q")):
                    return assigned
                if key in (ord("r"), ord("l")):
                    side = "right" if key == ord("r") else "left"
                    node = stable_device_path(port)
                    assigned = _drop_node(assigned, node)
                    assigned[side] = node
                    print(f"  ✓ {side} arm = {node}")
                    break
        finally:
            try:
                bus.disconnect()
            except Exception as e:  # noqa: BLE001 — cleanup must not raise
                print(f"⚠️  {port} disconnect failed: {e}")
    return assigned


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
        print(f"  serial: probing {len(ports)} port(s) for follower buses")
        sensor_map["arms"] = _assign_arms(ports, sensor_map["arms"])
    else:
        print("  no serial ports found — arm step skipped")
    cv2.destroyAllWindows()
    return sensor_map


# ── Measurement phases ───────────────────────────────────────────────────────


def _live_hz(probe: SensorProbe) -> float:
    """Reads in the last second (cheap live-rate estimate)."""
    now = time.monotonic()
    recent = [s for s in probe.stamps[-1200:] if s > now - 1.0]
    return float(len(recent))


def run_view(
    probes: list[SensorProbe],
    cameras: "list[CameraProbe]",
    joints: "list[JointProbe]",
) -> None:
    """Live window: all camera feeds in a grid + one joint panel per arm.

    Runs until q/Esc; the probes keep free-running underneath, so the
    rate summary printed afterwards reflects the same contention as the
    headless simultaneous phase.
    """
    print("\n▶ live view — press q or Esc in the window to stop")
    for p in probes:
        p.reset()
        p.start()
    try:
        while True:
            tiles = []
            for cam in cameras:
                frame = cam.last_frame
                if frame is None:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                else:
                    frame = frame.copy()
                cv2.putText(
                    frame,
                    f"{cam.name} {_live_hz(cam):.0f} Hz",
                    (8, 24),
                    _FONT,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                tiles.append(frame)
            for jp in joints:
                panel = np.zeros((480, 300, 3), dtype=np.uint8)
                cv2.putText(
                    panel,
                    f"{jp.name} {_live_hz(jp):.0f} Hz",
                    (8, 24),
                    _FONT,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                positions = jp.last_positions or {}
                for i, (joint, value) in enumerate(positions.items()):
                    cv2.putText(
                        panel,
                        f"{joint:<14} {value:8.2f}",
                        (8, 60 + 26 * i),
                        _FONT,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                tiles.append(panel)
            if not tiles:
                break
            cv2.imshow("sensor rates", grid_tiles(tiles, max_per_row=3))
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
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    from common.follower_bus import connect_follower_bus

    # Resolve sensor assignments: CLI --camera wins; otherwise the saved
    # map; no map and no --camera => the assignment GUI runs.
    sensor_map = (
        load_sensor_map(SENSOR_MAP_PATH)
        if SENSOR_MAP_PATH.exists()
        else {"cameras": {}, "arms": {}}
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
            if isinstance(device, str) and not Path(device).exists():
                raise SystemExit(
                    f"❌ camera {name} device {device} is missing — "
                    "replug or re-run with --assign"
                )
            cam = CameraProbe(
                name, device, args.width, args.height, args.request_fps, args.fourcc
            )
            cam.open()  # fail fast before touching the arms
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
            jp = JointProbe(f"joints[{side}]", bus)
            joints.append(jp)
            probes.append(jp)

        if args.view:
            run_view(probes, cameras, joints)
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
