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
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.chunk_metrics import summarise  # noqa: E402
from common.chunking import (  # noqa: E402
    DEFAULT_BLEND_WINDOW,
    DEFAULT_ENSEMBLE_WEIGHT,
    STRATEGIES,
)
from common.policy_client import LocalActionSource, RemoteActionSource  # noqa: E402
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
        "--strategy",
        default="append",
        choices=STRATEGIES,
        help="Remote: how an arriving chunk joins the one already executing. "
        "'append' is what this tool has always done; see common/chunking.py",
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
        "--arm-from-view",
        action="store_true",
        help="Take the 'the arms will move' consent in the live view instead "
        "of at the terminal. The page then starts the arms, so anyone who can "
        "reach the port can; it binds loopback and is unauthenticated",
    )
    parser.add_argument(
        "--start-mode",
        choices=("run", "hold", "preview"),
        default="run",
        help="What the throttle is doing when the loop starts. 'preview' ramps "
        "to the first action and then waits, showing each queued chunk in the "
        "twin until you execute it from the live view (needs --web)",
    )
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
    if args.arm_from_view and not args.web:
        raise SystemExit("❌ --arm-from-view needs --web: nothing else can arm it")
    if args.start_mode != "run" and not args.web:
        # Nothing else can move the throttle off preview or hold, so the run
        # would ramp to the first action and then sit there until it timed out.
        raise SystemExit(f"❌ --start-mode {args.start_mode} needs --web to leave it")

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
            strategy=args.strategy,
            blend_window=args.blend_window,
            ramp_kind=args.ramp,
            new_weight=args.new_weight,
        )
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 loading policy from {args.checkpoint} on {device} ...")
        source = LocalActionSource(str(args.checkpoint), device, args.task)
    print(f"  ✓ {source.describe()}")

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
    control = RunControl(mode=args.start_mode)
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
            control,
            source,
            data_manager,
            image_names,
            port=args.web_port,
            arm_from_view=bool(args.arm_from_view),
        )
        view.start()
        print(f"🖥️  live view on http://127.0.0.1:{args.web_port}/")
        print("   Open it now: the cameras and the throttle are live already.")

    if not args.dry_run and not args.yes and not args.arm_from_view:
        # Asked here, not earlier: the cameras, the buses and the live view are
        # all up, so the workspace can be checked on the screen that will show
        # the rollout rather than from memory.
        extra = (
            f"\n   The throttle starts in '{args.start_mode}'"
            if args.start_mode != "run"
            else ""
        )
        print(
            "\n⚠️  The follower arms will MOVE under policy control: ramp to the "
            f"first action, then {args.start_mode}.{extra}\n   Clear the "
            "workspace. Press Enter to proceed (Ctrl+C to abort)..."
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("aborted")

    run_log = None
    if not args.no_log:
        from common.policy_log import RunLog

        run_log = RunLog.create(task=args.task, source=source.describe(), hz=args.hz)
        print(f"📝 run log: {run_log.root}")

    dt = 1.0 / args.hz
    n_ticks = int(args.seconds * args.hz)
    torque_on = False
    try:
        if args.arm_from_view and not args.dry_run:
            # Waited out BEFORE the first inference, so the plan the arms ramp
            # to was drawn from the workspace as it is when consent is given,
            # not as it was while somebody was still clearing it.
            print(
                f"🖥️  waiting for 'Enable arms' on "
                f"http://127.0.0.1:{args.web_port}/ (Ctrl+C to abort) ..."
            )
            while not control.armed:
                if control.stopping:
                    raise SystemExit("aborted from the live view")
                time.sleep(0.05)
            print("✅ armed from the live view")

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
        # What the arms actually did, so the run can be judged after it rather
        # than only watched during it. See common/chunk_metrics.
        executed: "list[np.ndarray]" = []
        seams: "list[int]" = []
        round_trips: "list[float]" = []
        last_seq = 0
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

            seq = int(getattr(source, "last_chunk_seq", 0) or 0)
            if seq != last_seq:
                # Where the new plan STARTS, which under 'append' is behind
                # every leftover rather than on this tick.
                seams.append(len(executed) + int(getattr(source, "boundary_offset", 0)))
                last_seq = seq
                if source.round_trip_s:
                    round_trips.append(float(source.round_trip_s))

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
                executed.append(np.asarray(action12, dtype=float))
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
        if args.server and executed:
            row = summarise(
                np.asarray(executed),
                seams,
                held_ticks=holds,
                total_ticks=len(executed) + holds,
                round_trips_s=round_trips,
            )
            ratio = row["seam_ratio"]
            rtt = row["round_trip_ms_median"]
            print(
                f"\n📐 {args.strategy}: "
                f"seam ratio {'—' if ratio is None else format(ratio, '.2f')}  "
                f"held {row['held_fraction']:.0%}  "
                f"path {row['path_length']:.1f}  "
                f"chunks {row['chunks']}  "
                f"rtt median {'—' if rtt is None else format(rtt, '.0f') + ' ms'}"
            )
        if run_log is not None:
            run_log.close()
            print(f"📝 run log written: {run_log.root}")
        if view is not None:
            view.stop()


if __name__ == "__main__":
    main()
