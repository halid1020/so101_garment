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

While no session is running the console may open the assigned cameras itself,
for a look before recording starts. That preview is explicit, and it is always
released before a session is launched: two processes cannot hold one camera.
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

    def running(self) -> bool:
        return bool(self.captures)

    def start(self, specs: "list[tuple[str, str]] | None" = None) -> "list[str]":
        """Open the given ``(name, device)`` cameras, or the assigned ones.

        The Signals tab passes one unassigned device to identify it; the
        Collect tab passes nothing and gets the assigned set. Either way these
        are the console's own captures, and they are released before a session
        starts.
        """
        if self.running():
            if specs is not None and sorted(specs) != sorted(
                (c.name, str(c.device)) for c in self.captures
            ):
                self.stop()
            else:
                return [c.name for c in self.captures]
        from common.data_manager_dual import DualDataManager
        from common.recording.cameras import CameraCapture
        from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

        asked = specs is not None
        if specs is None:
            sensor_map = (
                load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
            )
            specs = sorted((sensor_map.get("cameras") or {}).items())
        if not specs:
            raise RuntimeError(
                "no cameras are assigned yet — assign them on the Signals tab "
                "(or with the sensor-assignment tool) first"
            )
        self.data_manager = DualDataManager()
        for name, device in specs:
            cam = CameraCapture(
                name=name,
                device=device,
                width=640,
                height=480,
                fps=30,
                rotate180=False,
                fourcc="MJPG",
            )
            if not cam.open():
                print(
                    f"⚠️  preview camera '{name}' ({device}) failed to open — skipped"
                )
                continue
            cam.start(self.data_manager)
            self.captures.append(cam)
        if not self.captures:
            self.data_manager = None
            if asked:
                devices = ", ".join(str(d) for _n, d in specs)
                raise RuntimeError(f"{devices} could not be opened")
            raise RuntimeError(
                "no assigned camera could be opened — they may be unplugged, or "
                "in a different socket than when they were assigned"
            )
        return [c.name for c in self.captures]

    def stop(self) -> None:
        for cam in self.captures:
            try:
                cam.stop()
            except Exception:
                pass
        self.captures = []
        self.data_manager = None

    def frame(self, name: str):
        if self.data_manager is None:
            return None
        return self.data_manager.get_rgb_image(name)

    def stream_names(self) -> "list[str]":
        return [c.name for c in self.captures]
