"""Run a policy in the digital twin, with inference here or on another machine.

This is the BATCH half of the twin: many episodes on chosen seeds, scored, with
no operator in the loop. For one interactive rollout with the live view -- the
deployment procedure itself, rehearsed -- use ``tool/run_policy.py --sim``.

This is ``tool/run_policy.py`` with the bench swapped for the twin. The
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

    # the comparison this exists for: every strategy at every tuning
    venv/bin/python tool/run_policy_sim.py --server http://127.0.0.1:8765 \\
        --task handover --grid all \\
        --pace virtual --latency-ticks 18 --episodes 5

THE GRID. ``--grid`` takes ``all`` -- the built-in seventeen cells -- or a
``;``-separated list of ``strategy[:param=v1,v2]`` terms, and a cell is a
strategy TOGETHER with the numbers it reads, because the strategies are not
comparable as bare names. ``common/chunk_sweep.py`` holds the syntax, the grid
and the reason ``rtc`` is not in it. Every cell runs the same seeds, and the
table is ranked by success rate first and by how fast the successes finished
second -- which is the question the sweep exists to answer.

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

from common.chunk_metrics import summarise  # noqa: E402
from common.chunk_sweep import expand_grid  # noqa: E402
from common.chunk_sweep import SPEC_HELP, SweepSpecError, cell_label  # noqa: E402
from common.chunking import STRATEGIES  # noqa: E402
from common.policy_rig import TwinRig  # noqa: E402
from common.policy_rig import parse_camera_map as _parse_camera_map  # noqa: E402

#: The twin renders 'scene' where the rig records 'central'; a checkpoint
#: trained on rig data asks for the latter. Offered as the default so the
#: common case needs no flag, and overridable because a sim-trained policy
#: wants no rename at all.
DEFAULT_CAMERA_MAP = {"scene": "central"}


def parse_camera_map(text: "str | None") -> "dict[str, str]":
    """``scene=central,wrist_camera_left=left`` -> a rename map."""
    try:
        return _parse_camera_map(text, DEFAULT_CAMERA_MAP)
    except ValueError as exc:
        raise SystemExit(f"❌ {exc}")


def _refuse_mismatched_cameras(args, source) -> None:
    """Stop now if the twin's cameras are not the ones the policy asks for.

    The names are known before the first tick -- the host reports them at the
    handshake -- so a sweep that would fail on every episode of every cell can
    say so in its first second instead of raising a WireError partway through.
    The default rename bridges a RIG-trained checkpoint to the twin; a policy
    trained IN the twin wants no rename at all, and the two look identical on
    the command line.
    """
    wanted = getattr(source, "cameras", None)
    if wanted is None:
        return
    sent = sorted(
        _parse_camera_map(args.camera_map, DEFAULT_CAMERA_MAP).get(name, name)
        for name in args.camera_names
    )
    if sent == sorted(wanted):
        return
    raise SystemExit(
        f"❌ the twin would send {sent}, but the policy asks for "
        f"{sorted(wanted)}.\n"
        f"   Rename them with --camera-map, or pass --camera-map none if this "
        f"checkpoint was trained in the twin."
    )


def make_source(args, cell: dict, hz: float):
    """A local or remote action source, configured for one sweep cell.

    One source per cell rather than one reconfigured between them: a fresh
    handshake carries no queue, no in-flight request and no round-trip history
    from the tuning before it, and the prefetch threshold grows from the link
    this cell actually saw rather than the last one's.
    """
    if args.server:
        from common.policy_client import RemoteActionSource

        # The flags are the floor; the cell overrides whatever it names. A cell
        # only ever names parameters its own strategy reads, so a flag that does
        # not apply to this strategy is simply carried unused.
        knobs = {
            "blend_window": args.blend_window,
            "ramp_kind": args.ramp,
            "new_weight": args.new_weight,
            "execute_ratio": args.execute_ratio,
        }
        knobs.update({k: v for k, v in cell.items() if k != "strategy"})
        return RemoteActionSource(
            args.server,
            args.task_string,
            actions_per_chunk=args.actions_per_chunk,
            prefetch=args.prefetch,
            hz=hz,
            strategy=cell["strategy"],
            camera_map=parse_camera_map(args.camera_map),
            virtual_delay_ticks=(
                args.latency_ticks if args.pace == "virtual" else None
            ),
            **knobs,
        )
    from common.policy_client import LocalActionSource

    return LocalActionSource(args.checkpoint, args.device, args.task_string)


def rehandshake(source) -> None:
    """Put a source back to how it started, between two episodes.

    Thirty trials of one cell only mean thirty independent samples if nothing
    survives the boundary. ``reset()`` is the source's own word for that where
    it has one -- a fresh session with the host, so the observation window and
    the server's sampler state go too -- and dropping the queue is the most that
    can be done where it does not. A local policy keeps its action queue inside
    itself, so that gets reset as well where it is reachable.
    """
    reset = getattr(source, "reset", None)
    if callable(reset):
        reset()
    else:
        source.drain()
    policy = getattr(source, "policy", None)
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()


def run_episode(env, source, scenario, args, label: str, composer=None) -> dict:
    """One rollout. Returns the metrics row for this episode."""
    from tool.eval_sim_policy import _released

    env.reset(scenario)
    rehandshake(source)
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
    #: The tick index at which the task was first done AND let go of. This is
    #: the finishing speed the sweep exists to measure: two cells that both
    #: succeed are not equally good if one takes half as long to get there.
    ticks_to_success: "int | None" = None
    last_seq = 0
    dt = 1.0 / args.fps
    tick = -1
    wall0 = time.perf_counter()

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
            ticks_to_success = tick
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
    row["cell"] = label
    row["strategy"] = str(getattr(source, "strategy", label))
    # 0-based, so a successful episode has ticks == ticks_to_success + 1.
    row["ticks_to_success"] = ticks_to_success
    row["fps"] = float(args.fps)
    row["wall_s"] = round(time.perf_counter() - wall0, 2)
    row["place_err_mm"] = round(float(env.place_error() * 1e3), 2)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", choices=("single", "handover", "handover_split"), default="handover"
    )
    parser.add_argument("--checkpoint", help="Local inference from this directory")
    parser.add_argument("--server", help="Remote inference at this base URL")
    parser.add_argument(
        "--strategy",
        default="append",
        choices=STRATEGIES,
        help="How an arriving chunk joins the one executing (default: append)",
    )
    parser.add_argument(
        "--grid",
        help="Sweep these (strategy, tuning) cells on the same seeds, then rank "
        "them. Overrides --strategy. " + SPEC_HELP,
    )
    parser.add_argument(
        "--sweep",
        help="Deprecated spelling of --grid that takes a comma list of bare "
        "strategy names; each runs at the tuning the flags below give.",
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
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Control rate, driven into the twin as well as the client. PIN "
        "this to the rate the checkpoint under test was trained at — 30 for the "
        "original handover checkpoints, 25 for anything collected since — as a "
        "sweep run at another rate compares the strategies on a policy that is "
        "being stepped wrong",
    )
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
        "--execute-ratio",
        type=float,
        default=0.5,
        help="--strategy receding: fraction of each chunk to execute (0-1)",
    )
    parser.add_argument(
        "--camera-map",
        default=None,
        help="Rename cameras for the checkpoint, e.g. 'scene=central' "
        "(the default; pass 'none' for a sim-trained policy)",
    )
    parser.add_argument("--out", help="Write the metrics table here")
    parser.add_argument(
        "--video-dir",
        help="Write one composite rollout video per episode here, named for the "
        "cell and the seed (slow: it renders an extra overview per tick)",
    )
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.server):
        raise SystemExit("❌ pass exactly one of --checkpoint (local) or --server")
    if args.pace == "virtual" and not args.server:
        raise SystemExit("❌ --pace virtual only means something with --server")
    if args.pace == "virtual" and args.latency_ticks is None:
        raise SystemExit("❌ --pace virtual needs --latency-ticks N")
    if args.latency_ticks is not None and args.pace != "virtual":
        raise SystemExit("❌ --latency-ticks only applies to --pace virtual")
    if args.grid and args.sweep:
        raise SystemExit("❌ pass --grid or --sweep, not both")

    spec = args.grid
    if args.sweep:
        # One path, not two: the old comma list is the new syntax with the
        # separator swapped, and every cell then takes its tuning from the
        # flags exactly as it did before.
        print("⚠ --sweep is deprecated; use --grid")
        spec = ";".join(s.strip() for s in args.sweep.split(",") if s.strip())
    try:
        cells = expand_grid(
            spec or args.strategy,
            defaults={
                "execute_ratio": args.execute_ratio,
                "blend_window": args.blend_window,
                "new_weight": args.new_weight,
                "ramp_kind": args.ramp,
            },
        )
    except SweepSpecError as exc:
        raise SystemExit(f"❌ {exc}")
    if len(cells) > 1 and not args.server:
        # A local source has no splice: every cell would run the same policy and
        # the table would be one experiment repeated under N different labels.
        raise SystemExit("❌ a grid of more than one cell needs --server")

    from sim_datagen.env import CAMERAS, TASKS, PickPlaceTwinEnv
    from sim_datagen.seeds import EVAL_SEEDS, VAL_SEEDS
    from tool.eval_sim_policy import _make_script, _pick_device, _scenario_for_seed

    args.task_string = TASKS[args.task]
    args.camera_names = list(CAMERAS)
    args.device = _pick_device(args.device) if args.checkpoint else "remote"

    if args.seeds == "simple":
        default_seed = 0 if args.task == "single" else 14
        # `simple` repeats ONE scenario, so it measures a strategy against a
        # policy that has already seen it. `full` is the held-out pool.
        seed = default_seed if args.simple_seed is None else args.simple_seed
        seeds = [seed] * (args.episodes or 5)
    else:
        pool = list(VAL_SEEDS if args.seeds == "val" else EVAL_SEEDS)
        seeds = pool[: args.episodes] if args.episodes else pool

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(PickPlaceTwinEnv, args.task, args.fps)
    print(f"▶ {args.task}: {len(seeds)} episode(s) x {len(cells)} cell(s)")
    print(f"  cameras: {', '.join(args.camera_names)}   pace: {args.pace}")
    print(f"  grid: {', '.join(cell_label(c) for c in cells)}")

    rows: "list[dict]" = []
    for cell in cells:
        label = cell_label(cell)
        source = make_source(args, cell, args.fps)
        _refuse_mismatched_cameras(args, source)
        print(f"\n== {label} — {source.describe()}")
        for i, seed in enumerate(seeds):
            scenario = _scenario_for_seed(args.task, seed)
            script = _make_script(args.task, scenario, env)
            args.max_ticks = int(
                np.ceil(
                    (args.max_seconds * args.fps)
                    if args.max_seconds is not None
                    else 1.5 * script.duration * args.fps
                )
            )
            composer = _make_composer(args) if video_dir is not None else None
            row = run_episode(env, source, scenario, args, label, composer)
            row["seed"] = seed
            rows.append(row)
            if composer is not None and video_dir is not None:
                composer.save(video_dir / _video_name(label, seed, i))
                composer.close()
            ratio = row["seam_ratio"]
            reached = row["ticks_to_success"]
            print(
                f"  seed {seed:>6}  {'✓' if row['success'] else '·'}  "
                f"seam {'—' if ratio is None else f'{ratio:.2f}'}  "
                f"held {row['held_fraction']:.0%}  "
                f"{row['place_err_mm']:.0f} mm  "
                f"{'—' if reached is None else f'{reached / args.fps:.1f} s'}"
            )

    table = _table(_fold(rows))
    print("\n" + table)
    if args.out:
        Path(args.out).write_text(
            f"# Chunking strategies — {args.task}\n\n"
            f"grid: `{spec or args.strategy}` · {len(seeds)} episode(s) per cell · "
            f"seeds `{args.seeds}` · pace `{args.pace}`"
            + (
                f" · latency {args.latency_ticks} ticks\n\n"
                if args.pace == "virtual"
                else "\n\n"
            )
            + f"{table}\n"
            f"```json\n{json.dumps(rows, indent=2)}\n```\n"
        )
        print(f"✓ wrote {args.out}")
    return 0


def _make_env(env_cls, task: str, fps: float):
    """The twin, stepped at the rate the client is being paced at.

    The environment took its tick rate as a constant until recently and takes
    it as an argument now; asked for by keyword only where the constructor
    admits one, so this tool works either side of that change and never
    silently runs a 25 Hz twin under a 30 Hz client.
    """
    import inspect

    if "fps" in inspect.signature(env_cls).parameters:
        return env_cls(task, fps=fps)
    return env_cls(task)


def _make_composer(args):
    """The rollout-video composer ``eval_sim_policy`` uses, on the same frames."""
    from common.eval_video import EvalVideoComposer

    return EvalVideoComposer(int(round(args.fps)))


def _video_name(label: str, seed: int, index: int) -> str:
    """A cell label carries '/' and '@'; a filename may not carry the first."""
    return f"{label.replace('/', '-')}_seed{seed}_{index}.mp4"


def _median(values: "list[float]") -> "float | None":
    """Median of what is there, or None when nothing is. Pure, no numpy."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _fold(rows: "list[dict]") -> "list[dict]":
    """The per-episode rows of each cell -> one row per cell.

    ``success`` is a RATE, not a verdict: over thirty seeds a strategy that
    finishes twenty-nine of them is not the same as one that finishes none, and
    an ``all()`` over the episodes calls both of them False.

    ``ticks_to_success`` is folded as a MEDIAN over the episodes that succeeded,
    because a failure has no finishing time and counting it as the timeout would
    make the metric a second, worse-scaled success rate. A cell that never
    succeeded reports no time at all rather than a zero.
    """
    folded: "dict[str, list[dict]]" = {}
    for row in rows:
        folded.setdefault(str(row.get("cell") or row["strategy"]), []).append(row)
    out = []
    for label, group in folded.items():
        ratios = [r["seam_ratio"] for r in group if r["seam_ratio"] is not None]
        trips = [
            r["round_trip_ms_median"]
            for r in group
            if r.get("round_trip_ms_median") is not None
        ]
        wins = [r for r in group if r.get("success")]
        reached = [
            float(r["ticks_to_success"])
            for r in wins
            if r.get("ticks_to_success") is not None
        ]
        errs = [
            float(r["place_err_mm"]) for r in group if r.get("place_err_mm") is not None
        ]
        fps = float(group[0].get("fps") or 0.0)
        median_ticks = _median(reached)
        out.append(
            {
                "strategy": label,
                "cell": label,
                "base_strategy": group[0].get("strategy", label),
                "episodes": len(group),
                "successes": len(wins),
                "success": len(wins) / len(group),
                "ticks_to_success_median": median_ticks,
                "seconds_to_success_median": (
                    None if median_ticks is None or fps <= 0 else median_ticks / fps
                ),
                "seam_ratio": _median(ratios) if ratios else None,
                "held_fraction": float(
                    sum(r["held_fraction"] for r in group) / len(group)
                ),
                "path_length": float(sum(r["path_length"] for r in group) / len(group)),
                "round_trip_ms_median": (
                    float(sum(trips) / len(trips)) if trips else None
                ),
                "place_err_mm": float(sum(errs) / len(errs)) if errs else None,
                "wall_s": round(float(sum(r.get("wall_s", 0.0) for r in group)), 1),
            }
        )
    return sorted(out, key=_rank)


def _rank(row: dict):
    """Best first: most successes, then the quickest to get there.

    Seam ratio breaks the remaining ties. It is the reason the strategies were
    written, but it is not the reason to choose one: a cell that finishes the
    task more often, and sooner, wins over a cell with prettier joins.
    """
    reached = row.get("ticks_to_success_median")
    ratio = row.get("seam_ratio")
    return (
        -float(row.get("success") or 0.0),
        float("inf") if reached is None else float(reached),
        float("inf") if ratio is None else float(ratio),
        str(row.get("strategy", "")),
    )


def _table(folded: "list[dict]") -> str:
    """A markdown table of one folded row per cell, best first. Pure."""
    if not folded:
        return "_no runs_\n"
    header = (
        "| cell | success | median t→done | seam ratio | held | path "
        "| place err | rtt median | wall |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in sorted(folded, key=_rank):
        secs = row.get("seconds_to_success_median")
        ratio = row.get("seam_ratio")
        rtt = row.get("round_trip_ms_median")
        err = row.get("place_err_mm")
        lines.append(
            "| {name} | {ok:.0%} ({n}/{eps}) | {secs} | {ratio} | {held:.0%} "
            "| {path:.1f} | {err} | {rtt} | {wall:.0f} s |\n".format(
                name=row.get("strategy", "?"),
                ok=float(row.get("success") or 0.0),
                n=int(row.get("successes", 0)),
                eps=int(row.get("episodes", 0)),
                secs="—" if secs is None else f"{secs:.2f} s",
                ratio="—" if ratio is None else f"{ratio:.2f}",
                held=float(row.get("held_fraction", 0.0) or 0.0),
                path=float(row.get("path_length", 0.0) or 0.0),
                err="—" if err is None else f"{err:.0f} mm",
                rtt="—" if rtt is None else f"{rtt:.0f} ms",
                wall=float(row.get("wall_s", 0.0) or 0.0),
            )
        )
    return header + "".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
