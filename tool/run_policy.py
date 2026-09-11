"""Run a trained policy on the rig: the physical followers, or the twin.

The deployment tool. It builds observations from the live cameras and the
followers' measured joints, runs the SAME inference (``load_policy`` ->
preprocess -> ``select_action`` -> postprocess) that ``tool/eval_sim_policy.py``
runs, and sends the predicted 12-D action to the followers through the exact
conversion the recorder and ``tool/replay_on_robot.py`` use. So the action
definition the dataset froze is what drives the arms — no separate
real-inference code path.

``--sim`` puts the digital twin where the bench stands (``common.policy_rig``),
with no camera, no bus and no motor anywhere. Everything else is identical — the
page, the arming handshake, the throttle, the splice, the stall ladder, the run
log — so a procedure rehearsed in simulation is the procedure the arms will run,
in the same order. It is how this tool is checked without a robot in the room.
(For MANY scored episodes on chosen seeds instead of one interactive rollout,
``tool/run_policy_sim.py`` is the batch harness.)

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

``--web`` serves a live view of the run on loopback (``common.web.policy_view``)
and is meant to be the WHOLE interface: the observation the policy was last
shown, the chunk it planned -- drawn in the URDF twin as the arms now (blue)
beside where that plan sends them (orange) -- what the arms did with it, and
whether it arrived in time. From there a rollout is given its task, armed,
HELD, advanced ONE CHUNK at a time, resumed and re-spliced. That is how a failed
grasp is examined: the plan that produced the motion is still on the screen
beside the motion.

ONE RUN, MANY TRIALS. Stop ends a trial and RELEASES the arms -- torque off, the
rig free to be handled -- without ending the run: the page, the cameras, the
host session and the log are all still there, because a rig is usually attempted
more than once and re-launching between attempts costs a handshake, a ramp and
the comparison of one splice against another on one scene. Reset scene does the
same and begins the next trial, putting the scene back where the scene is a data
structure. The process itself ends on Ctrl+C. Every rollout is written to a run
log, one row per tick and numbered by trial (``--no-log`` to skip). Each attempt
can also be JUDGED from the page -- success, failure or discard, with a note --
and the verdict is appended against its trial number, so two checkpoints are
compared from the logs rather than from somebody's memory. ``--log-frames``
additionally keeps the camera frames each plan was drawn from, which is what
``tool/analyse_policy_inputs.py`` needs to attribute a plan to the streams that
produced it.

SAFETY: the followers MOVE (unless ``--dry-run`` or ``--sim``). Consent is taken once, before
any torque -- at the terminal by default, or ON THE PAGE when ``--web`` is used
(``--arm-at-terminal`` puts the prompt back, ``--yes`` skips it). The page is
unauthenticated on loopback, so with ``--web`` anyone who can reach the port can
begin the motion. The arms then ramp slowly to the policy's first action and run
at ``--hz`` until ``--seconds`` elapse (by default they do not: a run lasts until
it is ended), Stop or Reset releases them, or Ctrl+C ends the run. Torque comes
off at every one of those, and again on exit, including on error. Consent is
taken per TRIAL: a run armed from the page is disarmed by a release and must be
armed again, so the next motion is authorised by somebody looking at the rig as
it is now. ``--dry-run`` reads sensors and prints the chosen actions but never
enables torque or writes a goal. Keep the workspace clear.

Usage:

    # everything else -- task, arming, throttle, splice -- happens on the page
    venv/bin/python tool/run_policy.py --server http://127.0.0.1:8765 --web

    # the same procedure, rehearsed against the twin: no robot required
    venv/bin/python tool/run_policy.py --sim handover --web \\
        --server http://127.0.0.1:8765 \\
        --camera-map wrist_camera_left=wrist_left,wrist_camera_right=wrist_right

    venv/bin/python tool/run_policy.py \\
        --checkpoint <run>/checkpoints/last/pretrained_model \\
        --task "fold the towel" --dry-run          # infer only, no motion

    venv/bin/python tool/run_policy.py \\
        --checkpoint <ckpt> --task "fold the towel" --hz 15 --seconds 60
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from actoris_harena.deploy.chunk_metrics import compare, summarise  # noqa: E402
from actoris_harena.deploy.chunking import (  # noqa: E402
    DEFAULT_BLEND_WINDOW,
    DEFAULT_ENSEMBLE_WEIGHT,
    DEFAULT_EXECUTE_RATIO,
    STRATEGIES,
)

from common.policy_client import LocalActionSource, RemoteActionSource  # noqa: E402
from common.policy_rig import RAMP_S, parse_camera_map  # noqa: E402
from common.policy_run import RunControl  # noqa: E402
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

    from actoris_harena.recording.camera_controls import CONTROL_NAMES

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


class BenchRig:
    """The physical followers behind ``common.policy_rig.Rig``.

    Everything that can move a motor or open a camera lives here, and the
    control loop above talks only to the protocol -- which is what lets the twin
    stand in this class's place for a rehearsal, with the page, the arming
    handshake, the throttle and the log all unchanged.
    """

    def __init__(self, camera_map: "dict[str, str] | None" = None) -> None:
        from common.data_manager_dual import DualDataManager

        self.camera_map = dict(camera_map or {})
        self.frames = DualDataManager()
        self.captures = _start_cameras(self.frames)
        self._names = [c.name for c in self.captures]
        self.cameras = sorted(self.camera_map.get(n, n) for n in self._names)
        self.buses: dict = {}
        self.torque_on = False

    def connect(self) -> None:
        self.buses = _connect_followers()

    def observe(self) -> "tuple[np.ndarray, dict]":
        images = _gather_images(self.frames, self._names)
        return read_state(self.buses), images

    def command(self, action12) -> None:
        goals = policy_action_to_goals(action12)
        for s in _SIDES:
            self.buses[s].sync_write(
                "Goal_Position", goals[s], normalize=True, num_retry=2
            )

    def enable(self, first_action12) -> None:
        """Torque on, then ramp to the policy's first action. The one door."""
        for bus in self.buses.values():
            bus.enable_torque()
        self.torque_on = True
        print("🏁 ramping to the policy's first action ...")
        _ramp_to(self.buses, policy_action_to_goals(first_action12), duration=RAMP_S)
        time.sleep(0.5)

    def release(self) -> None:
        """Let the arms go, and leave everything else running.

        The end of a TRIAL, not of the run: the buses stay open, the cameras
        keep publishing and the page keeps serving, because the operator is
        about to handle the rig and will very likely want another attempt. The
        one thing that changes is that the followers can be moved by hand.
        """
        if not self.torque_on:
            return
        for bus in self.buses.values():
            try:
                bus.disable_torque(num_retry=3)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  could not disable torque on a follower: {e}")
        self.torque_on = False

    def shutdown(self) -> None:
        self.release()
        self.frames.request_shutdown()
        for c in self.captures:
            try:
                c.stop()
            except Exception:  # noqa: BLE001
                pass


def build_sim_rig(task: str, seed: int, camera_map: "dict[str, str]", hz: float = 30.0):
    """The twin standing where the bench stands. ``(rig, env, scenario)``.

    A rehearsal, not an evaluation: one scenario, one scene, no scoring.
    ``tool/run_policy_sim.py`` is the harness that scores many of them.

    The scenario is HANDED BACK rather than thrown away, because that value is
    the whole of what "put it back the way it was" means here: reset from the
    page spawns the same objects in the same places, so a second attempt is
    compared against the first rather than against a different problem.
    """
    from common.policy_rig import TwinRig
    from sim_datagen.env import CAMERAS, PickPlaceTwinEnv
    from tool.eval_sim_policy import _scenario_for_seed

    scenario = _scenario_for_seed(task, seed)
    # Stepped at the rate it is driven at: the control tick is a whole number of
    # integrator substeps, so an env left at the default while the loop runs at
    # another rate advances a different amount of simulated time per action.
    env = PickPlaceTwinEnv(task, fps=hz)
    env.reset(scenario)
    rig = TwinRig(env, cameras=list(CAMERAS), camera_map=camera_map, fps=hz)
    return rig, env, scenario


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


def _hold(rig, dry_run: bool) -> None:
    """Let a held tick pass on a rig whose clock only moves when commanded.

    The bench keeps ticking whatever anyone does -- its servos hold the last
    goal and the world carries on. The twin does not: it advances one control
    step per ``command``, so without this a paused rollout would freeze
    simulated time, and the preview-then-step cycle would never show the arms
    settling. ``hold`` is exactly that step with no new goal.
    """
    if dry_run:
        return
    holder = getattr(rig, "hold", None)
    if holder is not None:
        holder()


def _ticks(budget: "int | None"):
    """Tick indices for this run: ``budget`` of them, or without end."""
    return itertools.count() if budget is None else range(budget)


def tick_budget(seconds: float, hz: float) -> "int | None":
    """How many ticks a run gets, or ``None`` for no limit. Pure.

    ``--seconds 0`` means "until somebody stops it", which is what a session
    spent comparing splices from the live view needs: one ramp, one workspace,
    one scene, and as long as it takes.
    """
    if seconds <= 0:
        return None
    return int(seconds * hz)


def resolve_launch(
    web: bool,
    seconds: "float | None",
    start_mode: "str | None",
    arm_at_terminal: bool,
    yes: bool,
    dry_run: bool,
) -> "tuple[float, str, bool]":
    """Fill in what ``--web`` implies: duration, throttle, who consents. Pure.

    ``run_policy.py --server URL --web`` is meant to be the whole command,
    with everything else decided on the page, so under ``--web`` the defaults
    change to the ones that console needs:

      * no time limit -- a session spent stepping through plans and comparing
        splices cannot know in advance how long it wants, and the old 30 s
        default silently ended one at 900 ticks;
      * ``preview`` -- the plan is shown in the twin before anything moves, and
        Run is one click away;
      * consent taken IN the page, because the page asks the same question the
        terminal did and the operator is already looking at it.

    Any flag given explicitly wins, and ``--arm-at-terminal`` puts the prompt
    back. A dry run consents to nothing, having no torque to enable.
    """
    resolved_seconds = 0.0 if seconds is None else float(seconds)
    if start_mode is None:
        start_mode = "preview" if web else "run"
    arm_from_view = bool(web and not arm_at_terminal and not yes and not dry_run)
    return resolved_seconds, start_mode, arm_from_view


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
        "--task",
        default=None,
        help="Language task string fed to the policy. Optional with --web: the "
        "run then waits for one to be typed on the page",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Control rate (Hz)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Run duration before the process ends (default: no limit; the "
        "page ends TRIALS, Ctrl+C ends the run)",
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
        "--strategy",
        default="receding",
        choices=STRATEGIES,
        help="Remote: how an arriving chunk joins the one already executing. "
        "The default executes half of each chunk and then asks again, which "
        "keeps the plan young without a seam every tick; see common/chunking.py",
    )
    parser.add_argument(
        "--blend-window",
        type=int,
        default=DEFAULT_BLEND_WINDOW,
        help="Remote, --strategy blend: ticks to cross-fade out of the old plan",
    )
    parser.add_argument(
        "--ramp",
        choices=("linear", "exp"),
        default="linear",
        help="Remote, --strategy blend: shape of that cross-fade",
    )
    parser.add_argument(
        "--execute-ratio",
        type=float,
        default=DEFAULT_EXECUTE_RATIO,
        help="Remote, --strategy receding: fraction of each returned chunk to "
        "execute before asking again (0-1; 0.5 of a 12-action chunk is 6)",
    )
    parser.add_argument(
        "--new-weight",
        type=float,
        default=DEFAULT_ENSEMBLE_WEIGHT,
        help="Remote, --strategy ensemble: weight on the newly arrived plan",
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
        "--twin-port",
        type=int,
        default=None,
        help="Port for the live view's 3D twin, which is its own server and is "
        "embedded in the page (default: --web-port + 1)",
    )
    parser.add_argument(
        "--sim",
        nargs="?",
        const="handover",
        choices=("single", "handover", "handover_split"),
        default=None,
        help="Rehearse against the digital twin instead of the arms: builds the "
        "payload scene and runs the WHOLE procedure -- page, arming, throttle, "
        "splice, run log -- with no motor anywhere. No cameras, no buses",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=None,
        help="--sim: which scenario the twin starts in (default: the task's "
        "simple-mode seed, 0 for single and 14 for handover)",
    )
    parser.add_argument(
        "--camera-map",
        default=None,
        help="Remote: rename cameras on the way to the policy, e.g. "
        "'wrist_camera_left=wrist_left'. A checkpoint asks for the names its "
        "dataset used, which are not always the names the rig produces now",
    )
    parser.add_argument(
        "--arm-at-terminal",
        action="store_true",
        help="Take the 'the arms will move' consent at the terminal. Without "
        "it, --web takes that consent on the PAGE, which means anyone who can "
        "reach the port can start the arms; it binds loopback and is "
        "unauthenticated",
    )
    parser.add_argument(
        "--start-mode",
        choices=("run", "hold", "preview"),
        default=None,
        help="What the throttle is doing when the loop starts (default: "
        "'preview' with --web, 'run' without). 'preview' ramps to the first "
        "action and then waits, showing each queued chunk in the twin until "
        "you execute it from the live view",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write the run log (default: $SO101_OUTPUT_DIR/policy_runs/<stamp>)",
    )
    parser.add_argument(
        "--log-frames",
        action="store_true",
        help="Also keep the camera frames each plan was drawn from, under the "
        "run log's frames/. Off by default -- they are the largest part of an "
        "observation. Needed by tool/analyse_policy_inputs.py, which cannot "
        "attribute a plan to inputs it does not have",
    )
    args = parser.parse_args()

    if args.hz <= 0:
        raise SystemExit("❌ --hz must be > 0")
    if args.seconds is not None and args.seconds < 0:
        raise SystemExit("❌ --seconds must be >= 0 (0 means until stopped)")
    seconds, start_mode, arm_from_view = resolve_launch(
        bool(args.web),
        args.seconds,
        args.start_mode,
        bool(args.arm_at_terminal),
        bool(args.yes),
        bool(args.dry_run),
    )
    try:
        camera_map = parse_camera_map(args.camera_map)
    except ValueError as exc:
        raise SystemExit(f"❌ {exc}")
    if camera_map and not args.server:
        # LocalActionSource builds its batch from the names it is handed; only
        # the remote client renames on the way out.
        raise SystemExit("❌ --camera-map needs --server (remote inference)")

    if args.sim and not args.task:
        # A rehearsal knows its own task, so --sim alone is a whole command.
        from sim_datagen.env import TASKS

        args.task = TASKS[args.sim]
    if not args.task and not args.web:
        raise SystemExit("❌ --task is required without --web (nothing can set it)")
    if bool(args.checkpoint) == bool(args.server):
        raise SystemExit(
            "❌ pass exactly one of --checkpoint (local) or --server (remote)"
        )
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise SystemExit(f"❌ checkpoint not found: {args.checkpoint}")
    if args.stall_abort <= args.stall_hold:
        raise SystemExit("❌ --stall-abort must be greater than --stall-hold")
    if start_mode != "run" and not args.web:
        # Nothing else can move the throttle off preview or hold, so the run
        # would ramp to the first action and then sit there until it timed out.
        raise SystemExit(f"❌ --start-mode {start_mode} needs --web to leave it")

    if seconds == 0:
        print("ℹ️  no --seconds: this run ends on Ctrl+C (the page ends trials)")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    source: "RemoteActionSource | LocalActionSource"
    if args.server:
        print(f"🌐 handshaking with {args.server} ...")
        source = RemoteActionSource(
            args.server,
            args.task or "",
            actions_per_chunk=args.actions_per_chunk,
            prefetch=args.prefetch,
            hz=args.hz,
            strategy=args.strategy,
            blend_window=args.blend_window,
            ramp_kind=args.ramp,
            new_weight=args.new_weight,
            execute_ratio=args.execute_ratio,
            camera_map=camera_map,
        )
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 loading policy from {args.checkpoint} on {device} ...")
        source = LocalActionSource(str(args.checkpoint), device, args.task or "")
    print(f"  ✓ {source.describe()}")

    env = None
    scenario = None
    if args.sim:
        seed = args.sim_seed
        if seed is None:
            seed = 0 if args.sim == "single" else 14
        print(f"🧪 rehearsing in the twin: {args.sim}, scenario seed {seed}")
        rig, env, scenario = build_sim_rig(args.sim, seed, camera_map, hz=args.hz)
    else:
        rig = BenchRig(camera_map=camera_map)
    image_names = list(rig.cameras)

    # A camera set that does not match what the policy was trained on produces a
    # shape error deep inside inference; catch it here, before any torque. The
    # comparison is against the names AFTER --camera-map, because those are the
    # ones the policy will actually be sent.
    if source.cameras is not None and sorted(image_names) != sorted(source.cameras):
        rig.shutdown()
        raise SystemExit(
            f"❌ rig cameras {sorted(image_names)} do not match the policy's "
            f"{sorted(source.cameras)}"
        )

    if not args.sim:
        rig.connect()

    # The throttle exists whether or not anyone is watching: the loop consults
    # it every tick, and only the view (if asked for) ever changes it.
    n_ticks = tick_budget(seconds, args.hz)
    control = RunControl(mode=start_mode)
    control.publish(
        hz=args.hz,
        dry_run=bool(args.dry_run),
        task=args.task or "",
        source=source.describe(),
        cameras=list(image_names),
        chunked=bool(source.chunked),
        ticks_total=int(n_ticks or 0),
        torque=False,
    )

    view = None
    if args.web:
        from common.web.policy_view import PolicyView

        view = PolicyView(
            control,
            source,
            rig.frames,
            image_names,
            port=args.web_port,
            twin_port=args.twin_port,
            arm_from_view=arm_from_view,
            # The twin can put its own scene back; a real table cannot be
            # tidied by a button, so there the page holds the run and says
            # what the operator has to do.
            resettable="sim" if args.sim else "manual",
        )
        if not view.start():
            # Almost always another rollout still holding the port. Carrying on
            # would put an older run's cameras and throttle on the screen while
            # THESE arms move, which is worse than no view at all.
            rig.shutdown()
            raise SystemExit(
                f"❌ the live view could not start on port {args.web_port}: "
                f"{view.error}\n   Another rollout is probably still running "
                f"(pgrep -af run_policy). Stop it, or pass a different "
                f"--web-port."
            )
        print(f"🖥️  live view on http://127.0.0.1:{args.web_port}/")
        print("   Open it now: the cameras and the throttle are live already.")

    if not args.dry_run and not args.yes and not arm_from_view:
        # Asked here, not earlier: the cameras, the buses and the live view are
        # all up, so the workspace can be checked on the screen that will show
        # the rollout rather than from memory.
        extra = (
            f"\n   The throttle starts in '{start_mode}'" if start_mode != "run" else ""
        )
        which = "The SIMULATED arms" if args.sim else "The follower arms"
        print(
            f"\n⚠️  {which} will MOVE under policy control: ramp to the "
            f"first action, then {start_mode}.{extra}\n   Clear the "
            "workspace. Press Enter to proceed (Ctrl+C to abort)..."
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("aborted")

    run_log = None
    dt = 1.0 / args.hz
    task = args.task or ""
    # Declared before the try, because the cleanup below reads them and a run
    # that fails EARLY -- no server, no first action -- would otherwise die of
    # an UnboundLocalError that hides the reason it actually stopped.
    segments: "list[dict]" = []
    close_segment = None
    # Which attempt is being measured. A run outlives its trials, so a segment
    # carries this: two attempts under one splice are two rows, and averaging
    # them would hide the difference they were run to show.
    trial = [0]
    # Whether the arms are live: enabled, and not released since. Published
    # every tick, because the one thing an operator standing beside the rig
    # needs from the page is whether it is safe to take hold of it.
    live = [False]
    current = [getattr(source, "strategy", args.strategy)]
    try:
        if not task:
            # A run may be started with no task at all -- that is what makes the
            # one-command launch possible -- and it simply waits here. Nothing
            # has been inferred and no torque exists yet.
            print(
                f"🖥️  waiting for a task on http://127.0.0.1:{args.web_port}/ "
                f"(Ctrl+C to abort) ..."
            )
            while not task:
                task = str(control.snapshot().get("task") or "")
                time.sleep(0.05)
            print(f"📝 task: “{task}”")

        if not args.no_log:
            from common.policy_log import RunLog

            run_log = RunLog.create(
                task=task,
                source=source.describe(),
                hz=args.hz,
                save_frames=bool(args.log_frames),
                checkpoint=str(getattr(source, "checkpoint", "") or ""),
                policy_type=str(getattr(source, "type", "") or ""),
            )
            print(f"📝 run log: {run_log.root}")
            if args.log_frames:
                print("   keeping the frames behind each plan (--log-frames)")

        if arm_from_view and not args.dry_run:
            # Waited out BEFORE the first inference, so the plan the arms ramp
            # to was drawn from the workspace as it is when consent is given,
            # not as it was while somebody was still clearing it.
            print(
                f"🖥️  waiting for 'Enable arms' on "
                f"http://127.0.0.1:{args.web_port}/ (Ctrl+C to abort) ..."
            )
            while not control.armed:
                time.sleep(0.05)
            print("✅ armed from the live view")

        # Warm up the observation, then run one inference for the ramp target.
        state, images = rig.observe()
        source.offer(state, images)
        action12 = _first_action(source, _FIRST_ACTION_TIMEOUT_S)

        if args.dry_run:
            span = "until stopped" if n_ticks is None else f"for {seconds:.0f}s"
            print(f"🧪 dry-run: inferring at {args.hz:.0f} Hz {span}, NO motor writes")
        else:
            rig.enable(action12)
            live[0] = True
            print("🔴 running policy ...")

        # Set again by a reset: the next attempt's first action is reached at
        # the same deliberate pace as this one's, because after an attempt the
        # arms are wherever it left them and the new plan starts from the pose
        # the scene was put back to.
        needs_ramp = False

        stalled_since: "float | None" = None
        warned = False
        holds = 0
        # What the arms actually did, so the run can be judged after it rather
        # than only watched during it. See common/chunk_metrics.
        # One measurement SEGMENT per splice in force. Switching strategy from
        # the page closes the current segment and opens the next, so a session
        # spent comparing them ends with one row each rather than one average
        # over settings that were never in force at the same time.
        executed: "list[np.ndarray]" = []
        seams: "list[int]" = []
        round_trips: "list[float]" = []
        last_seq = 0

        def _close_segment(name: str) -> None:
            if executed:
                row = summarise(
                    np.asarray(executed),
                    seams,
                    held_ticks=segment_holds[0],
                    total_ticks=len(executed) + segment_holds[0],
                    round_trips_s=round_trips,
                )
                row["strategy"] = name
                row["trial"] = trial[0]
                segments.append(row)
            executed.clear()
            seams.clear()
            round_trips.clear()
            segment_holds[0] = 0

        segment_holds = [0]
        close_segment = _close_segment

        def _end_trial(restore: bool) -> None:
            """End this trial: measure it, put the arms down, hold.

            ``restore`` is the whole difference between the two things the page
            can ask for. Both RELEASE the rig, because nothing about a trial's
            end can be done to arms that are still stiff -- neither putting a
            scene back nor lifting a gripper off whatever it has folded itself
            around. Only a reset begins another attempt, which is what restoring
            the scene and counting a new trial mean.

            What deliberately survives: the cameras, the buses, the host
            session's socket, the page and the run log. Ending the process is
            Ctrl+C's job, and it is a different job.
            """
            nonlocal needs_ramp
            _close_segment(current[0])
            if restore and scenario is not None and hasattr(rig, "reset"):
                rig.reset(scenario)
            source.drain()
            start_over = getattr(source, "reset", None)
            if start_over is not None:
                start_over()
            if not args.dry_run:
                rig.release()
            live[0] = False
            if arm_from_view:
                # Consent was for the trial that has just ended. The button
                # comes back, and nothing is served until it is pressed again.
                control.disarm()
            needs_ramp = True
            if restore:
                trial[0] = run_log.new_trial() if run_log is not None else trial[0] + 1

        if view is not None:

            def on_outcome(outcome: str, notes: str):
                """The page judged this attempt. Returns None if nothing kept it."""
                if run_log is None:
                    return None
                record = run_log.trial_outcome(outcome, notes)
                mark = {"success": "✅", "failure": "❌", "discard": "🚫"}[outcome]
                print(
                    f"{mark} trial {record['trial']}: {outcome}"
                    + (f" — {notes}" if notes else "")
                )
                return record

            view.on_outcome = on_outcome
            view.logging = run_log is not None

            def on_splice_change(settings, _current=current):
                _close_segment(_current[0])
                _current[0] = settings["strategy"]
                print(f"🔀 splice is now {settings['strategy']}")

            view.on_splice_change = on_splice_change
        started_at = time.time()
        for tick in _ticks(n_ticks):
            t0 = time.perf_counter()
            # A mode change invalidates whatever was planned before it; a paused
            # rollout keeps filling its window but asks for nothing.
            if control.queue_stale():
                source.drain()
            # This trial is over, and no other is intended: the arms go free and
            # the run stays up. Not a stop in the old sense -- the process ends
            # on Ctrl+C, which is the only place that decision is now taken.
            if control.stop_requested():
                _end_trial(restore=False)
                print(
                    "⏹️  trial ended: plan dropped, holding"
                    if args.dry_run or args.sim
                    else "⏹️  trial ended: torque off — the arms are free to move"
                )
            # And the same, plus the next attempt: everything drawn from the
            # attempt just ended goes, the scene is put back where it can be,
            # and the throttle stays in hold so the operator decides when the
            # next one begins.
            if control.reset_requested():
                _end_trial(restore=True)
                print(
                    f"↺ trial {trial[0]}: scene restored, arms released, holding"
                    if scenario is not None
                    else f"↺ trial {trial[0]}: arms released — put the scene back"
                )
            source.set_paused(control.mode == "hold")

            state, images = rig.observe()
            source.offer(state, images)
            if source.fatal:
                print(f"⛔ {source.fatal} — stopping")
                break

            seq = int(getattr(source, "last_chunk_seq", 0) or 0)
            if seq != last_seq:
                # Where the new plan STARTS, which under 'append' is behind
                # every leftover rather than on this tick.
                seams.append(len(executed) + int(getattr(source, "boundary_offset", 0)))
                last_seq = seq
                if source.round_trip_s:
                    round_trips.append(float(source.round_trip_s))

            if arm_from_view and not control.armed:
                # Released and not taken up again. Held rather than served, and
                # WITHOUT consulting the throttle: a step's budget must not be
                # spent on a tick that was never going to move anything.
                gated = "hold"
            else:
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
                if needs_ramp:
                    needs_ramp = False
                    if not args.dry_run:
                        print("🏁 ramping to the new attempt's first action ...")
                        rig.enable(action12)
                        live[0] = True
                executed.append(np.asarray(action12, dtype=float))
                if not args.dry_run:
                    rig.command(action12)
            elif control.mode != "run":
                # Held or stepping: the pause is the operator's, so neither the
                # warning nor the abort applies -- the arms keep their last goal.
                holds += 1
                stalled_since = None
                warned = False
                _hold(rig, args.dry_run)
            else:
                holds += 1
                _hold(rig, args.dry_run)
                # Only a hold while RUNNING is the splice's doing; a pause the
                # operator asked for would otherwise swamp the measurement,
                # since preview holds on every single tick.
                segment_holds[0] += 1
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
                trial=trial[0],
                # float(), not round() alone: the twin observes in float32 and
                # numpy's round gives a numpy scalar back, which the view's JSON
                # encoder refuses -- so every status poll 500s and the whole page
                # goes blank. The bench happens to hand out float64.
                state=[round(float(v), 3) for v in state],
                commanded=(
                    None if action12 is None else [round(float(v), 3) for v in action12]
                ),
                queue=source.depth,
                round_trip_s=round(source.round_trip_s, 4),
                server_infer_s=round(getattr(source, "server_infer_s", 0.0), 4),
                holds=holds,
                last_error=source.last_error,
                served=action12 is not None,
                torque=live[0],
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
                if env is not None:
                    mark = "✓" if env.success() else "·"
                    extra += f"  place={env.place_error() * 1e3:4.0f}mm {mark}"
                print(
                    f"  t={tick / args.hz:5.1f}s  [{control.mode}]  "
                    f"action[:6]={action12[:6].round(2)}{extra}"
                )
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        else:
            print(f"✓ finished {n_ticks} ticks")  # never for an endless run
    except KeyboardInterrupt:
        print("\n⏹️  interrupted")
    finally:
        rig.shutdown()
        if args.server and close_segment is not None:
            close_segment(current[0])
        if segments:
            print("\n📐 what each splice did, trial by trial:\n")
            print(compare(segments))
        if run_log is not None:
            run_log.close()
            print(f"📝 run log written: {run_log.root}")
        if view is not None:
            view.stop()


if __name__ == "__main__":
    main()
