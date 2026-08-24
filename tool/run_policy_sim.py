"""Run a policy in the digital twin, with inference here or on another machine.

This is ``tool/run_policy_real.py`` with the bench swapped for the twin. The
client, the wire, the observation window and the chunk splice are the same code,
so a strategy measured here is the strategy the arms will execute -- and it can
be measured a hundred times, on chosen seeds, without anyone standing by the
workspace.

    # everything local, as eval_sim_policy does it
    venv/bin/python tool/run_policy_sim.py --checkpoint outputs/policies/... \\
        --task handover --episodes 5

    # policy on a GPU box, twin here, chunks spliced by the strategy named
    venv/bin/python tool/run_policy_sim.py --server http://127.0.0.1:8765 \\
        --task handover --strategy blend --camera-map scene=central

    # the comparison this exists for
    venv/bin/python tool/run_policy_sim.py --server http://127.0.0.1:8765 \\
        --task handover --sweep append,replace,blend,ensemble \\
        --pace virtual --latency-ticks 18 --episodes 5

PACING. ``--pace realtime`` holds 30 Hz against the wall clock and measures the
link you actually have. ``--pace virtual`` ignores the wall clock: the twin
advances exactly ``--latency-ticks`` ticks while a request is in flight, however
long it really took. Only the second is reproducible, and only the second lets a
strategy be swept against a delay the network will not oblige by producing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

os.environ.setdefault("MUJOCO_GL", "egl")

from common.chunk_metrics import compare, summarise  # noqa: E402
from common.chunking import STRATEGIES  # noqa: E402
from common.policy_rig import TwinRig  # noqa: E402

#: The twin renders 'scene' where the rig records 'central'; a checkpoint
#: trained on rig data asks for the latter. Offered as the default so the
#: common case needs no flag, and overridable because a sim-trained policy
#: wants no rename at all.
DEFAULT_CAMERA_MAP = {"scene": "central"}


def parse_camera_map(text: "str | None") -> "dict[str, str]":
    """``scene=central,wrist_camera_left=left`` -> a rename map."""
    if text is None:
        return dict(DEFAULT_CAMERA_MAP)
    if text.strip() in ("", "none"):
        return {}
    mapping = {}
    for pair in text.split(","):
        if "=" not in pair:
            raise SystemExit(f"❌ --camera-map wants name=name pairs, got '{pair}'")
        old, new = pair.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def make_source(args, strategy: str, hz: float):
    """A local or remote action source, configured for one strategy."""
    if args.server:
        from common.policy_client import RemoteActionSource

        return RemoteActionSource(
            args.server,
            args.task_string,
            actions_per_chunk=args.actions_per_chunk,
            prefetch=args.prefetch,
            hz=hz,
            strategy=strategy,
            blend_window=args.blend_window,
            ramp_kind=args.ramp,
            new_weight=args.new_weight,
            camera_map=parse_camera_map(args.camera_map),
            virtual_delay_ticks=(
                args.latency_ticks if args.pace == "virtual" else None
            ),
        )
    from common.policy_client import LocalActionSource

    return LocalActionSource(args.checkpoint, args.device, args.task_string)


def run_episode(env, source, scenario, args, strategy: str, composer=None) -> dict:
    """One rollout. Returns the metrics row for this episode."""
    from tool.eval_sim_policy import _released

    env.reset(scenario)
    source.drain()
    rig = TwinRig(
        env,
        cameras=list(args.camera_names),
        camera_wh=(args.camera_width, args.camera_height),
        camera_map=parse_camera_map(args.camera_map) if args.server else {},
    )

    executed: "list[np.ndarray]" = []
    seams: "list[int]" = []
    round_trips: "list[float]" = []
    held = 0
    success = False
    last_seq = 0
    dt = 1.0 / args.fps

    for tick in range(args.max_ticks):
        started = time.perf_counter()
        state, images = rig.observe()
        source.offer(state, images)
        if getattr(source, "fatal", None):
            raise SystemExit(f"❌ the host refused the request: {source.fatal}")

        seq = int(getattr(source, "last_chunk_seq", 0) or 0)
        if seq != last_seq:
            # A chunk landing is not a join: under 'append' the new plan waits
            # behind every leftover, so the boundary is that far ahead.
            seams.append(len(executed) + int(getattr(source, "boundary_offset", 0)))
            last_seq = seq
            trip = float(getattr(source, "round_trip_s", 0.0) or 0.0)
            if trip:
                round_trips.append(trip)

        action = source.take()
        if action is None:
            held += 1
            rig.hold()
        else:
            rig.command(action)
            executed.append(np.asarray(action, dtype=float))

        if composer is not None:
            composer.add(
                images,
                env.sim.render_overview(args.camera_width, args.camera_height),
                state,
                rig.last_action,
            )

        if env.success() and _released(env):
            success = True
            break

        if args.pace == "realtime":
            time.sleep(max(0.0, dt - (time.perf_counter() - started)))

    row = summarise(
        np.asarray(executed) if executed else np.zeros((0, 12)),
        seams,
        held_ticks=held,
        total_ticks=tick + 1,
        success=success,
        round_trips_s=round_trips,
    )
    row["strategy"] = strategy
    row["place_err_mm"] = round(float(env.place_error() * 1e3), 2)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=("single", "handover"), default="handover")
    parser.add_argument("--checkpoint", help="Local inference from this directory")
    parser.add_argument("--server", help="Remote inference at this base URL")
    parser.add_argument(
        "--strategy",
        default="append",
        choices=STRATEGIES,
        help="How an arriving chunk joins the one executing (default: append)",
    )
    parser.add_argument(
        "--sweep",
        help="Comma list of strategies to run in turn on the same seeds, "
        "then compare. Overrides --strategy.",
    )
    parser.add_argument(
        "--pace",
        choices=("realtime", "virtual"),
        default="realtime",
        help="realtime holds 30 Hz against the clock; virtual lets the twin "
        "advance only when an action is executed (reproducible)",
    )
    parser.add_argument(
        "--latency-ticks",
        type=int,
        default=None,
        help="Virtual pacing: pretend every round trip took this many ticks",
    )
    parser.add_argument("--seeds", choices=("simple", "val", "full"), default="val")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--simple-seed", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--actions-per-chunk", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument("--blend-window", type=int, default=5)
    parser.add_argument("--ramp", choices=("linear", "exp"), default="linear")
    parser.add_argument("--new-weight", type=float, default=0.7)
    parser.add_argument(
        "--camera-map",
        default=None,
        help="Rename cameras for the checkpoint, e.g. 'scene=central' "
        "(the default; pass 'none' for a sim-trained policy)",
    )
    parser.add_argument("--out", help="Write the metrics table here")
    parser.add_argument("--video-dir", help="Write one rollout video per episode")
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.server):
        raise SystemExit("❌ pass exactly one of --checkpoint (local) or --server")
    if args.pace == "virtual" and not args.server:
        raise SystemExit("❌ --pace virtual only means something with --server")
    if args.pace == "virtual" and args.latency_ticks is None:
        raise SystemExit("❌ --pace virtual needs --latency-ticks N")
    if args.latency_ticks is not None and args.pace != "virtual":
        raise SystemExit("❌ --latency-ticks only applies to --pace virtual")

    from sim_datagen.env import CAMERAS, TASKS, PickPlaceTwinEnv
    from sim_datagen.seeds import EVAL_SEEDS, VAL_SEEDS
    from tool.eval_sim_policy import _make_script, _pick_device, _scenario_for_seed

    args.task_string = TASKS[args.task]
    args.camera_names = list(CAMERAS)
    args.device = _pick_device(args.device) if args.checkpoint else "remote"

    if args.seeds == "simple":
        default_seed = 0 if args.task == "single" else 14
        seed = default_seed if args.simple_seed is None else args.simple_seed
        seeds = [seed] * (args.episodes or 5)
    else:
        pool = list(VAL_SEEDS if args.seeds == "val" else EVAL_SEEDS)
        seeds = pool[: args.episodes] if args.episodes else pool

    strategies = (
        [s.strip() for s in args.sweep.split(",")] if args.sweep else [args.strategy]
    )
    for name in strategies:
        if name not in STRATEGIES:
            raise SystemExit(f"❌ unknown strategy '{name}'")

    env = PickPlaceTwinEnv(args.task)
    print(f"▶ {args.task}: {len(seeds)} episode(s) x {len(strategies)} strategy(ies)")
    print(f"  cameras: {', '.join(args.camera_names)}   pace: {args.pace}")

    rows: "list[dict]" = []
    for strategy in strategies:
        source = make_source(args, strategy, args.fps)
        print(f"\n== {strategy} — {source.describe()}")
        per_episode = []
        for seed in seeds:
            scenario = _scenario_for_seed(args.task, seed)
            script = _make_script(args.task, scenario, env)
            args.max_ticks = int(
                np.ceil(
                    (args.max_seconds * args.fps)
                    if args.max_seconds is not None
                    else 1.5 * script.duration * args.fps
                )
            )
            row = run_episode(env, source, scenario, args, strategy)
            row["seed"] = seed
            per_episode.append(row)
            ratio = row["seam_ratio"]
            print(
                f"  seed {seed:>6}  {'✓' if row['success'] else '·'}  "
                f"seam {'—' if ratio is None else f'{ratio:.2f}'}  "
                f"held {row['held_fraction']:.0%}  "
                f"{row['place_err_mm']:.0f} mm"
            )
        rows.extend(per_episode)

    table = compare(_fold(rows))
    print("\n" + table)
    if args.out:
        Path(args.out).write_text(
            f"# Chunking strategies — {args.task}\n\n{table}\n"
            f"```json\n{json.dumps(rows, indent=2)}\n```\n"
        )
        print(f"✓ wrote {args.out}")
    return 0


def _fold(rows: "list[dict]") -> "list[dict]":
    """Average the per-episode rows of each strategy into one row."""
    folded: "dict[str, dict]" = {}
    for row in rows:
        bucket = folded.setdefault(row["strategy"], {"strategy": row["strategy"]})
        bucket.setdefault("_rows", []).append(row)
    out = []
    for name, bucket in folded.items():
        group = bucket["_rows"]
        ratios = [r["seam_ratio"] for r in group if r["seam_ratio"] is not None]
        trips = [
            r["round_trip_ms_median"]
            for r in group
            if r["round_trip_ms_median"] is not None
        ]
        out.append(
            {
                "strategy": name,
                "success": all(r["success"] for r in group),
                "seam_ratio": float(np.mean(ratios)) if ratios else None,
                "held_fraction": float(np.mean([r["held_fraction"] for r in group])),
                "path_length": float(np.mean([r["path_length"] for r in group])),
                "round_trip_ms_median": float(np.mean(trips)) if trips else None,
            }
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
