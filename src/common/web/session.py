"""Supervising one collection session for the console, and the idle preview.

The rig is owned by one process. A collection session is therefore the
unchanged teleoperation recorder, started here as a subprocess with exactly the
flags the command-line front-end would have computed -- the same resolver backs
both, so a session started from the browser and one started from a terminal
cannot drift apart -- plus a monitor port, which is what lets the console show
the cameras and press the two allowed keys while it runs.

Nothing in this module imports aiohttp: what is worth testing is the resolution
of a request into a command, the refusals that stop a session that cannot work,
and the escalation ladder that ends one. The routes are in ``session_api``.

While no session is running the console may open the assigned cameras and the
follower buses itself, for a look before recording starts. Both previews are
explicit, and both are always released before a session is launched: two
processes cannot hold one camera, nor one serial port.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from common.recording.collection_settings import (
    SelectionError,
    is_resumable_dataset,
    read_existing_streams,
    resolve_new_selection,
    resolve_resume_selection,
    selection_to_teleop_flags,
)
from common.recording.dataset_edit import writability_problem
from common.web.lifecycle import valid_dataset_name

TELEOP = Path(__file__).resolve().parents[3] / "tool" / "meta_quest_teleopration.py"

# How the console ends a session. Quitting is a request the session honours at
# the top of its loop, which lets it park the arms, finish an in-flight episode
# and close the dataset properly -- so it is given time before anything harsher.
# An interrupt reaches the same teardown by another door. SIGKILL is absent on
# purpose: it would abandon an open episode and leave the arms energised.
QUIT_GRACE_S = 20.0
INTERRUPT_GRACE_S = 40.0


def stop_action(elapsed_s: float) -> str:
    """What ending a session should do next, ``elapsed_s`` after asking. Pure."""
    if elapsed_s < QUIT_GRACE_S:
        return "wait"
    if elapsed_s < INTERRUPT_GRACE_S:
        return "interrupt"
    return "terminate"


def session_refusals(
    root: Path,
    name: str,
    task: str,
    resuming: bool,
    exists: bool,
    running: bool,
) -> "list[str]":
    """Every reason this session cannot start, in the operator's terms. Pure."""
    reasons: list[str] = []
    if running:
        reasons.append("a session is already running — stop it first")
    problem = valid_dataset_name(name)
    if problem:
        reasons.append(problem)
    if not str(task).strip():
        reasons.append(
            "give the dataset an instruction — it is stored with every frame"
        )
    if exists and not resuming:
        reasons.append(
            f"{name!r} exists but holds no saved episodes (a session that quit "
            "before recording). Delete it on the Datasets tab, then start again"
        )
    problem = writability_problem(root)
    if problem:
        reasons.append(problem)
    return reasons


def _device_present(device: "str | int") -> bool:
    """Whether the device node is still there. Tells two failures apart.

    A camera that is gone and a camera that is present but was refused its share
    of the USB bandwidth both fail to open, and they want opposite fixes -- plug
    it back in, or record fewer streams. The node answers which it is.
    """
    if isinstance(device, int):
        return True
    try:
        return Path(str(device)).exists()
    except OSError:
        return False


def usb_budget_warnings(selected: "set[str]") -> "list[str]":
    """Warn when a stream selection over-subscribes one USB controller.

    Reads the assignments rather than any device, so a session can be warned
    about before it opens anything. An unassigned stream contributes nothing --
    its bus is unknowable from a bare device index.
    """
    from common.recording.usb_budget import selection_warnings
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    if not SENSOR_MAP_PATH.exists():
        return []
    nodes = (load_sensor_map(SENSOR_MAP_PATH) or {}).get("cameras") or {}
    return selection_warnings(selected, nodes)


def absent_stream_refusals(selected: "set[str]") -> "list[str]":
    """Refuse a selected stream whose assigned device node is not there.

    A camera that has been unplugged, or moved to a socket it was not assigned
    from, cannot be recorded: the recorder opens every camera before it creates
    the dataset, so the session would exit seconds after Start with its reason
    buried in the output tail. Said here instead, it is a sentence beside the
    form. This is a refusal rather than a warning because the outcome is
    certain, not likely.

    Reads the assignments and the filesystem only -- no device is opened. A
    stream with no assignment contributes nothing: a bare device index says
    nothing about whether its camera is plugged in.
    """
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    if not SENSOR_MAP_PATH.exists():
        return []
    nodes = (load_sensor_map(SENSOR_MAP_PATH) or {}).get("cameras") or {}
    return [
        f"camera '{name}' is assigned to {nodes[name]}, which is not there — "
        "plug it back in, untick it, or clear its assignment on the Signals tab"
        for name in sorted(selected)
        if name in nodes and not _device_present(nodes[name])
    ]


def resolve_plan(
    root: Path,
    name: str,
    task: str,
    options: "dict[str, Any]",
    config: "dict[str, Any]",
    running: bool = False,
) -> "dict[str, Any]":
    """Turn a start request into the session's stream selection and flags.

    Returns ``{resuming, cameras, depth, ee, fps, warnings, refusals, flags}``.
    A resumed dataset follows its own recorded settings and reports whatever it
    ignored; a new one takes the request, defaulting to the machine's enabled
    cameras. Refusals are collected rather than raised so the form can show
    everything that is wrong at once.
    """
    dataset_root = Path(root) / name
    exists = dataset_root.exists()
    resuming = is_resumable_dataset(dataset_root)
    leader = options.get("input") == "leader"

    refusals = session_refusals(Path(root), name, task, resuming, exists, running)
    known = set(config.get("cameras") or {})
    default_enabled = {
        n for n, c in (config.get("cameras") or {}).items() if c["enabled"]
    }
    rs_rgb_name = (config.get("realsense") or {}).get("rgb_name")

    warnings: list[str] = []
    selection: dict[str, Any] = {
        "cameras": set(),
        "depth": False,
        "ee": False,
        "fps": None,
    }
    try:
        if resuming:
            selection, warnings = resolve_resume_selection(
                read_existing_streams(dataset_root),
                rs_rgb_name,
                options.get("streams") or [],
                bool(options.get("depth")),
                not options.get("ee", True),
                options.get("fps"),
                leader=leader,
            )
        else:
            selection = resolve_new_selection(
                known,
                default_enabled,
                options.get("streams") or [],
                bool(options.get("depth")),
                bool(options.get("ee", True)) and not leader,
                options.get("fps"),
            )
    except SelectionError as exc:
        refusals.append(str(exc))

    # A warning, never a refusal: the per-bus figure is measured rather than
    # guaranteed, and the uvcvideo FIX_BANDWIDTH quirk can lift it, so the
    # operator is told what to expect and left to decide. The authority on
    # whether a stream really got its bandwidth is the camera open itself.
    warnings += usb_budget_warnings(set(selection["cameras"]))

    # A refusal, unlike the budget above: an absent camera is not a risk the
    # operator can weigh, it is a session that will exit as soon as it starts.
    refusals += absent_stream_refusals(set(selection["cameras"]))

    return {
        "resuming": resuming,
        "cameras": sorted(selection["cameras"]),
        "depth": bool(selection["depth"]),
        "ee": bool(selection["ee"]),
        "fps": selection["fps"],
        "warnings": warnings,
        "refusals": refusals,
        "flags": selection_to_teleop_flags(
            known,
            set(selection["cameras"]),
            bool(selection["depth"]),
            bool(selection["ee"]),
            selection["fps"],
        ),
    }


def teleop_argv(
    root: Path,
    name: str,
    task: str,
    plan: "dict[str, Any]",
    options: "dict[str, Any]",
    monitor_port: int,
    teleop: "Path | None" = None,
) -> "list[str]":
    """The command that runs this session. Pure — unit-tested."""
    argv = [
        str(teleop or TELEOP),
        "--input",
        str(options.get("input") or "quest"),
        "--record",
        "--repo-id",
        name,
        "--dataset-root",
        str(Path(root) / name),
        "--task",
        task,
        "--monitor-port",
        str(int(monitor_port)),
        *plan["flags"],
    ]
    if plan["resuming"]:
        argv.append("--resume")
    if options.get("ip_address"):
        argv += ["--ip-address", str(options["ip_address"])]
    if options.get("goal"):
        argv += ["--episode-goal", str(int(options["goal"]))]
    if options.get("sensor_view"):
        argv.append("--sensor-view")
    if options.get("no_streaming_encode"):
        argv.append("--no-streaming-encode")
    return argv


class SessionSupervisor:
    """The one collection session this console may run, and its output tail."""

    def __init__(
        self,
        root: Path,
        monitor_port: int = 8766,
        python: "str | None" = None,
        teleop: "Path | None" = None,
        tail_lines: int = 400,
    ) -> None:
        self.root = Path(root)
        self.monitor_port = int(monitor_port)
        self.python = python or sys.executable
        self.teleop = teleop or TELEOP
        self._proc: subprocess.Popen | None = None
        self._tail: deque = deque(maxlen=tail_lines)
        self._lock = threading.Lock()
        self._info: dict[str, Any] = {}
        self._stopping_since: float | None = None

    # ── State ────────────────────────────────────────────────────────────────

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def state(self) -> "dict[str, Any]":
        proc = self._proc
        with self._lock:
            tail = list(self._tail)
        return {
            "running": self.running(),
            "pid": None if proc is None else proc.pid,
            "returncode": None if proc is None else proc.poll(),
            "monitor_port": self.monitor_port,
            "stopping": self._stopping_since is not None,
            "tail": tail,
            **self._info,
        }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(
        self, name: str, task: str, plan: "dict[str, Any]", options: "dict[str, Any]"
    ) -> "dict[str, Any]":
        if self.running():
            raise RuntimeError("a session is already running")
        argv = teleop_argv(
            self.root, name, task, plan, options, self.monitor_port, self.teleop
        )
        env = dict(os.environ)
        # Collection is local-only: a stray Hub lookup while creating or
        # resuming the dataset would fail with a misleading credentials error.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        with self._lock:
            self._tail.clear()
        self._stopping_since = None
        self._proc = subprocess.Popen(
            [self.python, *argv],
            # NO stdin. Inherited, it would be the terminal the console itself
            # was launched in -- and a leader session reads its control keys
            # from stdin, so it would put that terminal into raw mode underneath
            # the operator's shell and then race the shell for every keystroke.
            # Neither reader gets a usable stream. The session is driven from
            # the page instead, which is where the operator is looking.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._info = {
            "name": name,
            "task": task,
            "argv": argv,
            "input": str(options.get("input") or "quest"),
            "resuming": plan["resuming"],
            "cameras": plan["cameras"],
            "depth": plan["depth"],
            "ee": plan["ee"],
            "fps": plan["fps"],
            "started": time.time(),
        }
        threading.Thread(target=self._read_output, daemon=True).start()
        return self.state()

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                with self._lock:
                    self._tail.append(line.rstrip("\n"))
        finally:
            proc.wait()
            proc.stdout.close()

    def signal_stop(self) -> None:
        """Note that the operator asked to end the session (the key is pressed
        by the caller, through the monitor). The ladder consults the clock."""
        if self._stopping_since is None:
            self._stopping_since = time.monotonic()

    def escalate(self) -> str:
        """Apply the next step of the stop ladder, and say which it was."""
        if not self.running() or self._stopping_since is None:
            return "done"
        action = stop_action(time.monotonic() - self._stopping_since)
        proc = self._proc
        if proc is None:
            return "done"
        if action == "interrupt":
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        elif action == "terminate":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return action


class PreviewCameras:
    """The console's own camera threads, for a look while nothing is recording.

    Only the assigned UVC cameras: the depth device is heavier to open and
    belongs to the recorder, and a preview exists to answer "is this camera
    pointing where I think" before a session, not to duplicate the recorder.
    """

    def __init__(self) -> None:
        self.data_manager: Any = None
        self.captures: list = []
        # What was ASKED for, which is not what opened: a camera the USB bus
        # refused is skipped, so comparing the running captures against a repeat
        # request would see a difference every time and restart the preview on
        # every poll. The request is the identity of a preview, not its result.
        self.requested: "list[tuple[str, str]]" = []
        # Streams that were asked for and could not be opened, with the reason,
        # so the page can say which camera is missing instead of quietly
        # showing one tile fewer than the rig has cameras.
        self.skipped: "list[dict[str, str]]" = []

    def running(self) -> bool:
        return bool(self.captures)

    @staticmethod
    def _capture_settings(name: str) -> "dict[str, Any]":
        """How this stream is configured for recording, or preview defaults.

        A preview exists to answer "is this camera pointing where I think, and
        does it look right" before a session. It can only answer the second half
        if it opens the camera the way the recorder will -- same size, same
        pixel format, same exposure -- so the settings come from recording.yaml
        rather than from a hard-coded guess. A device with no entry there is the
        Signals tab identifying an unassigned camera, and gets the defaults.
        """
        defaults: "dict[str, Any]" = {
            "width": 640,
            "height": 480,
            "fps": 30,
            "rotate180": False,
            "fourcc": "MJPG",
            "controls": {},
        }
        try:
            from common.camera_controls import CONTROL_NAMES
            from common.config_parser import load_recording_config

            cfg = (load_recording_config()["cameras"] or {}).get(name)
        except Exception:  # noqa: BLE001 — a broken config must not stop a preview
            return defaults
        if not cfg:
            return defaults
        return {
            "width": cfg["width"],
            "height": cfg["height"],
            "fps": cfg["fps"],
            "rotate180": cfg["rotate180"],
            "fourcc": cfg.get("fourcc") or "MJPG",
            "controls": {k: cfg.get(k) for k in CONTROL_NAMES},
        }

    @staticmethod
    def _disabled_streams() -> "set[str]":
        """Stream names ``recording.yaml`` knows about and turns off.

        A camera the rig no longer carries stays in the map -- clearing the
        assignment would throw away which socket it was measured in -- and is
        switched off in the recording config instead. The preview follows that,
        so an unplugged camera is not opened, not reported missing on every
        poll, and not offered as a tile no session would record.
        """
        try:
            from common.config_parser import load_recording_config

            cameras = load_recording_config()["cameras"] or {}
        except Exception:  # noqa: BLE001 — a broken config must not stop a preview
            return set()
        return {name for name, cfg in cameras.items() if not cfg["enabled"]}

    def start(self, specs: "list[tuple[str, str]] | None" = None) -> "list[str]":
        """Open the given ``(name, device)`` cameras, or the assigned ones.

        The Signals tab passes one unassigned device to identify it; the
        Collect tab passes nothing and gets the assigned set. Either way these
        are the console's own captures, and they are released before a session
        starts.
        """
        from common.data_manager_dual import DualDataManager
        from common.recording.cameras import CameraCapture
        from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

        asked = specs is not None
        if specs is None:
            sensor_map = (
                load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
            )
            off = self._disabled_streams()
            specs = sorted(
                (n, d)
                for n, d in (sensor_map.get("cameras") or {}).items()
                if n not in off
            )
        if not specs:
            raise RuntimeError(
                "no camera to preview — none is assigned on the Signals tab "
                "(or with the sensor-assignment tool), or every assigned one is "
                "disabled in src/conf/recording.yaml"
            )
        wanted = sorted((str(n), str(d)) for n, d in specs)
        if self.running():
            # Resolve the request BEFORE comparing: asking for the assigned set
            # while a single Signals camera is up used to fall through to
            # "already running" and hand the Collect tab that one camera.
            if wanted == sorted(self.requested):
                return [c.name for c in self.captures]
            self.stop()

        self.requested = wanted
        self.skipped = []
        self.data_manager = DualDataManager()
        for name, device in specs:
            settings = self._capture_settings(str(name))
            cam = CameraCapture(name=str(name), device=device, **settings)
            if not cam.open():
                reason = (
                    "opened but delivered no frames — most likely the USB "
                    "bandwidth budget (see the Signals tab)"
                    if _device_present(device)
                    else "could not be opened — unplugged, or in another socket"
                )
                print(f"⚠️  preview camera '{name}' ({device}) skipped: {reason}")
                self.skipped.append({"name": str(name), "reason": reason})
                continue
            cam.start(self.data_manager)
            self.captures.append(cam)
        if not self.captures:
            self.data_manager = None
            self.requested = []
            if asked:
                devices = ", ".join(str(d) for _n, d in specs)
                raise RuntimeError(f"{devices} could not be opened")
            raise RuntimeError(
                "no assigned camera could be opened — they may be unplugged, in "
                "a different socket than when they were assigned, or refused "
                "their share of the USB bandwidth (try fewer cameras)"
            )
        return [c.name for c in self.captures]

    def stop(self) -> None:
        for cam in self.captures:
            try:
                cam.stop()
            except Exception:
                pass
        self.captures = []
        self.requested = []
        self.skipped = []
        self.data_manager = None

    def frame(self, name: str):
        if self.data_manager is None:
            return None
        return self.data_manager.get_rgb_image(name)

    def stream_names(self) -> "list[str]":
        return [c.name for c in self.captures]

    def missing(self) -> "list[dict[str, str]]":
        """Asked-for streams that are not delivering, with why. For the page.

        Two ways in: a camera that never opened (recorded at start), and one
        that opened, worked, and was later given up on as starved. Both leave
        the operator a tile short, so both belong in the same list.
        """
        starved = [
            {
                "name": cam.name,
                "reason": "stopped delivering and was given up on — most likely "
                "the USB bandwidth budget",
            }
            for cam in self.captures
            if getattr(cam, "starved", False)
        ]
        return [*self.skipped, *starved]


class PreviewArms:
    """The follower arms read while nothing is recording, torque off.

    The joint table on the Collect tab is fed by the session once one runs. With
    no session there is nothing publishing joints at all, and an operator about
    to record wants the same question answered first: are both arms on the bus
    the map says, and does the console see them move? So the console reads them
    itself -- calibrated, torque disabled, nothing commanded, the arms limp --
    and reports in the same shape the session's monitor uses, with the command
    half empty because nothing is commanding them.

    Released before a session starts, like the camera preview: one process owns
    a serial port.
    """

    RATE_HZ = 10.0

    def __init__(self) -> None:
        self._buses: dict[str, Any] = {}
        self._state: dict[str, Any] = {}
        self._read_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def running(self) -> bool:
        return bool(self._buses)

    def start(self, map_path: "Path | None" = None) -> "list[str]":
        """Open both assigned follower buses. Returns the sides it got.

        ``map_path`` is the assignment file to read (the console passes its own,
        which a test points at a copy, so nothing here can open the machine's
        real arms by accident).
        """
        if self.running():
            return sorted(self._buses)
        from common.follower_bus import connect_follower_bus
        from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

        path = Path(map_path or SENSOR_MAP_PATH)
        sensor_map = load_sensor_map(path) if path.exists() else {}
        ports = sensor_map.get("arms") or {}
        if not ports:
            raise RuntimeError(
                "no follower arms are assigned yet — assign them on the Signals "
                "tab (or with the sensor-assignment tool) first"
            )
        failures = []
        for side in ("left", "right"):
            port = ports.get(side)
            if not port:
                failures.append(f"{side}: not assigned")
                continue
            try:
                self._buses[side] = connect_follower_bus(side, str(port))
            except Exception as exc:  # noqa: BLE001 — any failure is the same
                failures.append(f"{side}: {exc}")
        if not self._buses:
            raise RuntimeError("; ".join(failures) or "no follower bus could be opened")
        self._stop.clear()
        for side in self._buses:
            thread = threading.Thread(
                target=self._read_loop,
                args=(side,),
                name=f"preview-arm-{side}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return sorted(self._buses)

    def _read_loop(self, side: str) -> None:
        import numpy as np

        from common.joint_frames import hw_to_urdf
        from common.recording.features import BODY_JOINTS

        period = 1.0 / self.RATE_HZ
        bus = self._buses[side]
        while not self._stop.is_set():
            try:
                positions = bus.sync_read("Present_Position", num_retry=2)
            except Exception:  # noqa: BLE001 — a dropped packet is not a failure
                time.sleep(period)
                continue
            urdf = hw_to_urdf(
                side, np.array([positions[j] for j in BODY_JOINTS], dtype=np.float64)
            )
            with self._lock:
                self._state[side] = (urdf, positions["gripper"] / 100.0)
                self._read_at[side] = time.monotonic()
            time.sleep(period)

    def snapshot(self) -> "dict[str, Any] | None":
        """The joint block the Collect tab draws, or ``None`` if nothing is open."""
        from common.recording.monitor_server import joint_snapshot

        if not self.running():
            return None
        now = time.monotonic()
        with self._lock:
            state = dict(self._state)
            read_at = dict(self._read_at)
        if not state:
            return None  # opened, but no arm has answered yet
        measured: list[float] = []
        grippers: dict[str, Any] = {}
        for side in ("left", "right"):
            entry = state.get(side)
            measured.extend([0.0] * 5 if entry is None else list(entry[0]))
            grippers[side] = None if entry is None else entry[1]
        joints = joint_snapshot(
            measured,
            grippers,
            {side: (None, None, None) for side in ("left", "right")},
            now,
        )
        # A side that is not open has no numbers, rather than the zeros that
        # stood in for it while the other side was assembled.
        for side in ("left", "right"):
            if side not in state:
                joints[side]["state"] = {k: None for k in joints[side]["state"]}
        ages = [now - t for t in read_at.values()]
        return {
            "joints": joints,
            "teleop_active": False,
            "joint_drift_s": round(max(ages), 4) if ages else None,
            "source": "preview",
        }

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        for bus in self._buses.values():
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001 — cleanup must not raise
                pass
        self._buses = {}
        with self._lock:
            self._state = {}
            self._read_at = {}
