"""Run a trained policy on the PHYSICAL followers (on-robot inference).

The real-hardware analogue of ``tool/eval_sim_policy.py``: instead of rolling a
checkpoint out in the twin, this builds observations from the live cameras and
the followers' measured joints, runs the SAME inference (``load_policy`` ->
preprocess -> ``select_action`` -> postprocess), and sends the predicted 12-D
action to the followers through the exact conversion the recorder and
``tool/replay_on_robot.py`` use. So the action definition the dataset froze is
what drives the arms — no separate real-inference code path.

The policy input must match training: the enabled cameras (``recording.yaml``)
supply ``observation.images.<name>`` and both followers supply the 12-D
``observation.state`` (URDF degrees + gripper open fractions). Capture at the
resolution the dataset was collected at.

Inference does not have to run here. With ``--server`` the forward pass happens
on another machine (``tool/policy_server.py``) and action *chunks* come back --
which is nearly free, because a chunking policy already plans many steps from
one observation, so the link carries one observation per chunk rather than one
per tick. The rig keeps the cameras, the buses, the timing and every safety
decision either way; only the weights move. The next chunk is fetched in the
background while the current one is still being executed, and if it does not
arrive the arms HOLD their last goal and then stop (``--stall-hold`` /
``--stall-abort``) rather than run on stale plans.

``--web`` serves a live view of the run on loopback (``common.web.policy_view``):
the observation the policy was last shown, the chunk it planned -- replayed in
the URDF twin -- what the arms did with it, and whether it arrived in time. From
there a rollout can be HELD, advanced ONE CHUNK at a time, or resumed, which is
how a failed grasp is examined: the plan that produced the motion is still on
the screen beside the motion. Every rollout is also written to a run log for
afterwards (``--no-log`` to skip).

SAFETY: the followers MOVE (unless ``--dry-run``). After a confirmation (skip
with ``--yes``) the arms ramp slowly to the policy's first action, then run at
``--hz`` until ``--seconds`` elapse or Ctrl+C. Torque is disabled again on exit,
including on error. ``--dry-run`` reads sensors and prints the chosen actions
but never enables torque or writes a goal. The view can only ever ask for LESS
motion or end the run -- it cannot enable torque, and it cannot arm a rollout
the terminal did not. Keep the workspace clear.

Usage:

    venv/bin/python tool/run_policy_real.py \\
        --checkpoint <run>/checkpoints/last/pretrained_model \\
        --task "fold the towel" --dry-run          # infer only, no motion

    venv/bin/python tool/run_policy_real.py \\
        --checkpoint <ckpt> --task "fold the towel" --hz 15 --seconds 60

    venv/bin/python tool/run_policy_real.py \\
        --server http://127.0.0.1:8765 --task "fold the towel" --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.policy_run import RunControl, prefetch_threshold  # noqa: E402
from tool.replay_on_robot import (  # noqa: E402
    _connect_followers,
    _ramp_to,
    action_to_goal,
    present_to_urdf,
    split_action,
)

_SIDES = ("left", "right")

#: How long to wait for the first action before giving up. Generous: a remote
#: host may still be warming a GPU kernel, and nothing is under torque yet.
_FIRST_ACTION_TIMEOUT_S = 60.0


def policy_action_to_goals(action12) -> "dict[str, dict]":
    """A 12-D policy action (URDF deg + gripper frac) → per-side hardware goals.

    Composes ``split_action`` (the 12-channel layout) with ``action_to_goal``
    (the URDF->hardware conversion), so the policy's action reaches the motors
    through the exact path a recorded action does in ``replay_on_robot``. Pure —
    unit-tested.
    """
    per_side = split_action(action12)
    return {s: action_to_goal(s, *per_side[s]) for s in _SIDES}


def read_state(buses: dict) -> np.ndarray:
    """Assemble the 12-D ``observation.state`` from both followers' present pose."""
    state = np.zeros(12, dtype=np.float64)
    for s in _SIDES:
        pos = buses[s].sync_read("Present_Position", num_retry=2)
        base = 0 if s == "left" else 6
        state[base : base + 6] = present_to_urdf(s, pos)
    return state


def _start_cameras(data_manager):
    """Open + start the recording.yaml cameras, publishing into ``data_manager``.

    Reuses the teleop tool's stream resolution so the on-robot policy sees the
    same camera set (and stable by-path devices) the dataset was collected with.
    Returns the started capture objects (RGB cameras plus any central RealSense).
    """
    from types import SimpleNamespace

    from common.camera_controls import CONTROL_NAMES
    from common.config_parser import load_recording_config
    from common.recording.cameras import CameraCapture
    from tool.meta_quest_teleopration import (
        build_realsense_capture,
        overlay_sensor_map_devices,
        resolve_camera_streams,
    )
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    rec_cfg = load_recording_config()
    shim = SimpleNamespace(
        enable_camera=[], disable_camera=[], tactile=False, central_depth=False
    )
    streams = resolve_camera_streams(rec_cfg, shim)
    sensor_map = load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
    if sensor_map:
        streams = overlay_sensor_map_devices(streams, sensor_map)

    caps: list = []
    for name, cfg in streams.items():
        cam = CameraCapture(
            name=name,
            device=cfg["device"],
            width=cfg["width"],
            height=cfg["height"],
            fps=cfg["fps"],
            rotate180=cfg["rotate180"],
            fourcc=cfg["fourcc"],
            controls={k: cfg.get(k) for k in CONTROL_NAMES},
        )
        if not cam.open():
            for c in caps:
                c.stop()
            raise SystemExit(
                f"❌ camera '{name}' failed to open (device {cfg['device']})"
            )
        caps.append(cam)

    rs = build_realsense_capture(rec_cfg, shim, sensor_map, for_view=False)
    if rs is not None and rs.open():
        caps.append(rs)

    for c in caps:
        c.start(data_manager)
    return caps


def _gather_images(data_manager, names: "list[str]", timeout_s: float = 5.0) -> dict:
    """Latest RGB frame for each camera; waits briefly for all to arrive."""
    deadline = time.monotonic() + timeout_s
    while True:
        images = {n: data_manager.get_rgb_image(n) for n in names}
        if all(v is not None for v in images.values()):
            return images
        if time.monotonic() > deadline:
            missing = [n for n, v in images.items() if v is None]
            raise SystemExit(f"❌ no camera frames for {missing} after {timeout_s}s")
        time.sleep(0.05)


def stall_decision(
    have_action: bool, stalled_s: float, hold_s: float, abort_s: float
) -> str:
    """What the control loop should do this tick. Pure -- unit-tested.

    ``serve`` an action if there is one. Otherwise the arms are under torque
    with nothing new to command, so: ``hold`` the last goal briefly (a chunk
    boundary or one slow round trip is not an emergency), ``warn`` once the wait
    is long enough to be worth an operator's attention, and ``abort`` past the
    point where a hold has stopped being a pause and started being a hang --
    which lands in the caller's cleanup and takes torque off.
    """
    if have_action:
        return "serve"
    if stalled_s >= abort_s:
        return "abort"
    if stalled_s >= hold_s:
        return "warn"
    return "hold"


class LocalActionSource:
    """Inference in this process: one ``select_action`` per tick, as before.

    The policy's own action queue means only every ``n_action_steps``-th call
    touches the GPU; the rest are dequeues. Kept exactly as it was so that
    running without ``--server`` is unchanged.
    """

    def __init__(self, checkpoint: str, device: str, task: str) -> None:
        import torch

        from tool.eval_sim_policy import build_batch, load_policy

        self.torch = torch
        self.build_batch = build_batch
        self.policy, self.pre, self.post, self.type = load_policy(checkpoint, device)
        self.policy.reset()
        self.device = device
        self.task = task
        self.cameras: "list[str] | None" = None  # whatever the rig is configured with
        self.fatal: "str | None" = None
        self.last_error: "str | None" = None
        self.round_trip_s = 0.0  # nothing travels; kept so both sources report alike
        self._latest: "tuple[np.ndarray, dict] | None" = None

    def describe(self) -> str:
        return f"local '{self.type}' policy on {self.device}"

    #: Inference here has no chunk to step through: the policy's own queue is
    #: internal, so a tick either infers or dequeues and the operator sees one
    #: action at a time.
    chunked = False
    last_chunk: "np.ndarray | None" = None
    last_chunk_seq = 0
    last_chunk_at = 0.0

    def offer(self, state: np.ndarray, images: dict) -> None:
        self._latest = (state, images)

    def drain(self) -> None:
        """Nothing is queued here; the next take() infers from what is offered."""

    def set_paused(self, paused: bool) -> None:
        """No background fetching to pause."""

    def last_sent(self) -> "tuple[np.ndarray, dict] | None":
        return self._latest

    def take(self) -> "np.ndarray | None":
        if self._latest is None:
            return None
        state, images = self._latest
        return _infer(
            self.policy,
            self.pre,
            self.post,
            self.build_batch,
            state,
            images,
            self.task,
            self.device,
            self.torch,
        )

    @property
    def depth(self) -> int:
        return 0


class RemoteActionSource:
    """Inference on another machine; action chunks arrive ahead of being needed.

    The rig keeps capturing at the control rate and feeds every observation into
    a rolling window, because a policy with more than one observation step was
    trained on adjacent frames. When the local action queue runs low, the window
    is sent and the next chunk requested on a background thread, so the round
    trip overlaps motion the rig is already executing and the network never sits
    inside the control loop.

    A request that the server rejects outright (4xx) is fatal: the observation
    or the session is wrong and repeating it cannot help. A dropped connection
    or a timeout is not -- the next tick tries again, and the caller's stall
    policy decides how long that is allowed to go on.
    """

    def __init__(
        self,
        url: str,
        task: str,
        actions_per_chunk: "int | None" = None,
        prefetch: "int | None" = None,
        timeout_s: float = 20.0,
        hz: float = 30.0,
    ) -> None:
        import json

        from common.policy_wire import ObservationWindow

        self.url = url.rstrip("/")
        self.task = task
        self.timeout_s = float(timeout_s)

        meta = json.loads(self._rpc("/reset", b""))
        self.type = meta["policy_type"]
        self.session = meta["session"]
        self.cameras = sorted(meta["cameras"])
        self.n_obs_steps = int(meta["n_obs_steps"])
        self.server_actions = int(meta["n_action_steps"])
        self.actions = min(
            int(actions_per_chunk or self.server_actions), self.server_actions
        )
        # The static default covers a fast policy; the threshold below grows it
        # to cover whatever round trip this link turns out to have.
        self.prefetch = max(1, self.actions // 3)
        self.explicit_prefetch = None if prefetch is None else int(prefetch)
        self.hz = float(hz)
        self.window = ObservationWindow(self.cameras, self.n_obs_steps)

        self._queue: "deque[np.ndarray]" = deque()
        self._lock = threading.Lock()
        self._inflight = False
        self._seq = 0
        self.fatal: "str | None" = None
        self.last_error: "str | None" = None
        self.round_trip_s = 0.0
        self.server_infer_s = 0.0
        self._paused = False
        self._last_sent: "tuple[np.ndarray, dict] | None" = None
        self.last_chunk: "np.ndarray | None" = None
        self.last_chunk_seq = 0
        self.last_chunk_at = 0.0

    #: A chunk is a visible unit here, so a rollout can be stepped one at a time.
    chunked = True

    @property
    def threshold(self) -> int:
        """Queue depth at which the next chunk is requested. See policy_run."""
        return prefetch_threshold(
            self.prefetch,
            self.explicit_prefetch,
            self.round_trip_s,
            self.hz,
            self.actions,
        )

    def drain(self) -> None:
        """Forget what is queued: it was planned from an older observation."""
        with self._lock:
            self._queue.clear()

    def set_paused(self, paused: bool) -> None:
        """While paused the window keeps filling but no chunk is requested."""
        with self._lock:
            self._paused = bool(paused)

    def last_sent(self) -> "tuple[np.ndarray, dict] | None":
        """The observation the last request carried -- what the policy was shown."""
        return self._last_sent

    def describe(self) -> str:
        return (
            f"remote '{self.type}' policy at {self.url} "
            f"({self.n_obs_steps} obs step(s) per request, {self.actions} actions per chunk)"
        )

    def _rpc(self, path: str, data: "bytes | None" = None) -> bytes:
        import urllib.request

        request = urllib.request.Request(
            self.url + path,
            data=data,
            method="GET" if data is None else "POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def offer(self, state: np.ndarray, images: dict) -> None:
        """Record this tick's observation and, if the queue is low, ask for more."""
        self.window.push(state, images)
        with self._lock:
            if (
                self._inflight
                or self._paused
                or self.fatal is not None
                or len(self._queue) > self.threshold
            ):
                return
            self._inflight = True
            self._seq += 1
            seq = self._seq
        self._last_sent = (state, images)
        threading.Thread(
            target=self._fetch, args=(self.window.steps(), seq), daemon=True
        ).start()

    def _fetch(self, steps, seq: int) -> None:
        import urllib.error

        from common.policy_wire import decode_chunk, encode_request

        started = time.perf_counter()
        try:
            message = encode_request(
                steps, self.task, session=self.session, seq=seq, actions=self.actions
            )
            chunk, header = decode_chunk(self._rpc("/act", message))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()[:200]
            self.last_error = f"HTTP {exc.code}: {body}"
            if 400 <= exc.code < 500:
                self.fatal = self.last_error
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a transport hiccup must not kill the loop
            self.last_error = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                self._queue.extend(np.asarray(a, dtype=float) for a in chunk)
                self.last_chunk = np.asarray(chunk, dtype=float)
                self.last_chunk_seq = seq
                self.last_chunk_at = time.time()
                self.round_trip_s = time.perf_counter() - started
                self.server_infer_s = float(
                    (header.get("timings") or {}).get("infer_s", 0.0)
                )
                self.last_error = None
        finally:
            with self._lock:
                self._inflight = False

    def take(self) -> "np.ndarray | None":
        with self._lock:
            return self._queue.popleft() if self._queue else None

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)


def _first_action(source, timeout_s: float):
    """Block until the source has an action, or give up with a clear reason.

    Only used for the ramp target: from then on the loop never blocks, because
    blocking with arms under torque is exactly what the stall policy is for.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if source.fatal:
            raise SystemExit(f"❌ {source.fatal}")
        action = source.take()
        if action is not None:
            return action
        if time.monotonic() > deadline:
            raise SystemExit(
                f"❌ no action within {timeout_s:.0f}s"
                + (f" ({source.last_error})" if source.last_error else "")
            )
        time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", help="Path to a .../pretrained_model dir (local inference)"
    )
    parser.add_argument(
        "--server",
        help="Base URL of a tool/policy_server.py (remote inference); excludes --checkpoint",
    )
    parser.add_argument(
        "--task", required=True, help="Language task string fed to the policy"
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Control rate (Hz)")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="Run duration before stopping"
    )
    parser.add_argument(
        "--device", default=None, help="cpu/cuda (default auto, local only)"
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Remote: execute at most N actions per request (default: the policy's own chunk)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Remote: request the next chunk once the queue falls to N actions",
    )
    parser.add_argument(
        "--stall-hold",
        type=float,
        default=0.5,
        help="Remote: seconds without an action before warning (the last goal is held)",
    )
    parser.add_argument(
        "--stall-abort",
        type=float,
        default=2.0,
        help="Remote: seconds without an action before stopping and releasing torque",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Infer + print actions but never enable torque or command the arms",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the 'arms will move' confirmation"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve a live view of the rollout on loopback: what the policy was "
        "shown, what it planned, what the arms did, and a hold/step throttle",
    )
    parser.add_argument("--web-port", type=int, default=8767, help="Live view port")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write the run log (default: $SO101_OUTPUT_DIR/policy_runs/<stamp>)",
    )
    args = parser.parse_args()

    if args.hz <= 0 or args.seconds <= 0:
        raise SystemExit("❌ --hz and --seconds must be > 0")
    if bool(args.checkpoint) == bool(args.server):
        raise SystemExit(
            "❌ pass exactly one of --checkpoint (local) or --server (remote)"
        )
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise SystemExit(f"❌ checkpoint not found: {args.checkpoint}")
    if args.stall_abort <= args.stall_hold:
        raise SystemExit("❌ --stall-abort must be greater than --stall-hold")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    source: "RemoteActionSource | LocalActionSource"
    if args.server:
        print(f"🌐 handshaking with {args.server} ...")
        source = RemoteActionSource(
            args.server,
            args.task,
            actions_per_chunk=args.actions_per_chunk,
            prefetch=args.prefetch,
            hz=args.hz,
        )
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 loading policy from {args.checkpoint} on {device} ...")
        source = LocalActionSource(str(args.checkpoint), device, args.task)
    print(f"  ✓ {source.describe()}")

    if not args.dry_run and not args.yes:
        print(
            "\n⚠️  The follower arms will MOVE under policy control: ramp to the "
            "first action, then run.\n   Clear the workspace. Press Enter to "
            "proceed (Ctrl+C to abort)..."
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("aborted")

    from common.data_manager_dual import DualDataManager

    data_manager = DualDataManager()
    captures = _start_cameras(data_manager)
    image_names = [c.name for c in captures]

    # A camera set that does not match what the policy was trained on produces a
    # shape error deep inside inference; catch it here, before any torque.
    if source.cameras is not None and sorted(image_names) != sorted(source.cameras):
        for c in captures:
            c.stop()
        data_manager.request_shutdown()
        raise SystemExit(
            f"❌ rig cameras {sorted(image_names)} do not match the policy's "
            f"{sorted(source.cameras)}"
        )

    buses = _connect_followers()

    # The throttle exists whether or not anyone is watching: the loop consults
    # it every tick, and only the view (if asked for) ever changes it.
    control = RunControl()
    control.publish(
        hz=args.hz,
        dry_run=bool(args.dry_run),
        task=args.task,
        source=source.describe(),
        cameras=list(image_names),
        chunked=bool(source.chunked),
        ticks_total=int(args.seconds * args.hz),
    )

    view = None
    if args.web:
        from common.web.policy_view import PolicyView

        view = PolicyView(
            control, source, data_manager, image_names, port=args.web_port
        )
        view.start()
        print(f"🖥️  live view on http://127.0.0.1:{args.web_port}/")

    run_log = None
    if not args.no_log:
        from common.policy_log import RunLog

        run_log = RunLog.create(task=args.task, source=source.describe(), hz=args.hz)
        print(f"📝 run log: {run_log.root}")

    dt = 1.0 / args.hz
    n_ticks = int(args.seconds * args.hz)
    torque_on = False
    try:
        # Warm up the observation, then run one inference for the ramp target.
        images = _gather_images(data_manager, image_names)
        state = read_state(buses)
        source.offer(state, images)
        action12 = _first_action(source, _FIRST_ACTION_TIMEOUT_S)
        goals = policy_action_to_goals(action12)

        if args.dry_run:
            print(
                "🧪 dry-run: inferring at "
                f"{args.hz:.0f} Hz for {args.seconds:.0f}s, NO motor writes"
            )
        else:
            for b in buses.values():
                b.enable_torque()
            torque_on = True
            print("🏁 ramping to the policy's first action ...")
            _ramp_to(buses, goals, duration=3.0)
            time.sleep(0.5)
            print("🔴 running policy ...")

        stalled_since: "float | None" = None
        warned = False
        holds = 0
        started_at = time.time()
        for tick in range(n_ticks):
            t0 = time.perf_counter()
            if control.stopping:
                print("⏹️  stopped from the live view")
                break
            # A mode change invalidates whatever was planned before it; a paused
            # rollout keeps filling its window but asks for nothing.
            if control.queue_stale():
                source.drain()
            source.set_paused(control.mode == "hold")

            images = _gather_images(data_manager, image_names)
            state = read_state(buses)
            source.offer(state, images)
            if source.fatal:
                print(f"⛔ {source.fatal} — stopping")
                break

            gated = control.decide(source.depth if source.chunked else 1)
            action12 = None if gated == "hold" else source.take()
            what = stall_decision(
                action12 is not None,
                0.0 if stalled_since is None else time.perf_counter() - stalled_since,
                args.stall_hold,
                args.stall_abort,
            )
            if what == "serve":
                stalled_since = None
                warned = False
                goals = policy_action_to_goals(action12)
                if not args.dry_run:
                    for s in _SIDES:
                        buses[s].sync_write(
                            "Goal_Position", goals[s], normalize=True, num_retry=2
                        )
            elif control.mode != "run":
                # Held or stepping: the pause is the operator's, so neither the
                # warning nor the abort applies -- the arms keep their last goal.
                holds += 1
                stalled_since = None
                warned = False
            else:
                holds += 1
                if stalled_since is None:
                    stalled_since = time.perf_counter()
                if what == "abort":
                    print(
                        f"⛔ no action for {args.stall_abort:.1f}s "
                        f"({source.last_error or 'no reply yet'}) — stopping"
                    )
                    break
                if what == "warn" and not warned:
                    warned = True
                    print(
                        f"⚠️  holding position: no action for {args.stall_hold:.1f}s "
                        f"({source.last_error or 'chunk not back yet'})"
                    )

            control.publish(
                tick=tick,
                t=tick / args.hz,
                state=[round(v, 3) for v in state],
                commanded=None if action12 is None else [round(v, 3) for v in action12],
                queue=source.depth,
                round_trip_s=round(source.round_trip_s, 4),
                server_infer_s=round(getattr(source, "server_infer_s", 0.0), 4),
                holds=holds,
                last_error=source.last_error,
                served=action12 is not None,
            )
            if run_log is not None:
                run_log.tick(
                    t=time.time() - started_at,
                    state=state,
                    commanded=action12,
                    mode=control.mode,
                    queue=source.depth,
                )
                run_log.note_chunk(source)

            if tick % max(1, int(args.hz)) == 0 and action12 is not None:
                extra = (
                    f"  queue={source.depth:>3} rtt={source.round_trip_s * 1e3:4.0f}ms"
                    if args.server
                    else ""
                )
                print(
                    f"  t={tick / args.hz:5.1f}s  [{control.mode}]  "
                    f"action[:6]={action12[:6].round(2)}{extra}"
                )
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        else:
            print(f"✓ finished {n_ticks} ticks")
    except KeyboardInterrupt:
        print("\n⏹️  interrupted")
    finally:
        if torque_on:
            for b in buses.values():
                try:
                    b.disable_torque(num_retry=3)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️  could not disable torque on a follower: {e}")
        data_manager.request_shutdown()
        for c in captures:
            try:
                c.stop()
            except Exception:  # noqa: BLE001
                pass
        if run_log is not None:
            run_log.close()
            print(f"📝 run log written: {run_log.root}")
        if view is not None:
            view.stop()


def _infer(
    policy, preprocessor, postprocessor, build_batch, state, images, task, device, torch
) -> np.ndarray:
    """One policy step: observation -> 12-D action (numpy). Mirrors eval_sim_policy."""
    batch = build_batch(state, images, task, device)
    with torch.no_grad():
        batch = preprocessor(batch)
        action = policy.select_action(batch)
        action = postprocessor(action)
    return np.asarray(action.squeeze(0).to("cpu")).astype(float)


if __name__ == "__main__":
    main()
