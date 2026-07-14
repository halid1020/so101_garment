#!/usr/bin/env python3
"""Pick–handover–place benchmark for Quest teleop methods (MuJoCo).

One arm picks the payload cube, hands it over at the midline, the other
arm places it on the target. Runs N seeded, IK-feasibility-checked
scenarios per method and reports success rate, place error, and tracking
quality — the bimanual-coordination complement to run_benchmark.py's
tracking-only sweep.

Usage:
    python src/sim_benchmark/run_handover.py                      # 30 x all methods
    python src/sim_benchmark/run_handover.py --methods pink_relaxed scipy_ls
    python src/sim_benchmark/run_handover.py --view --methods scipy_ls --scenarios 0
    python src/sim_benchmark/run_handover.py --save out.json --plot plots/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
for _p in (str(_repo_root), str(_repo_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim_benchmark.constants import CONTROL_RATE_HZ, SIDES  # noqa: E402
from sim_benchmark.handover import (  # noqa: E402
    HandoverScript,
    Scenario,
    generate_scenarios,
)
from sim_benchmark.methods import (  # type: ignore[attr-defined]  # noqa: E402
    METHODS,
    MethodFactory,
)
from sim_benchmark.scene import DualArmSim  # noqa: E402

SUCCESS_RADIUS = 0.02  # m, XY distance payload-to-target that counts as success


def run_episode(
    sim: DualArmSim,
    method_factory: MethodFactory,
    scenario: Scenario,
    view: bool = False,
    gif_path: Path | None = None,
) -> dict:
    """Run one scenario with one method; return episode results."""
    method = method_factory(sim.model)
    q0 = sim.neutral_q()
    sim.reset(q0)
    method.reset(q0)
    sim.set_payload_pos(scenario.payload_pos)

    initial_poses = {side: sim.eef_pose(side) for side in SIDES}
    script = HandoverScript(scenario, initial_poses)

    dt = 1.0 / CONTROL_RATE_HZ
    n_substeps = max(1, round(dt / sim.model.opt.timestep))
    n_ticks = int(script.duration / dt)

    track_errs: list[float] = []
    solve_ms: list[float] = []
    payload_path: list[np.ndarray] = []
    held_by_placer = False
    # XY offset payload-minus-EE measured when the placer acquires the
    # object; the human aligns the *object* over the target, not their
    # hand, so the placer's targets are corrected by this offset.
    placer_offset: np.ndarray | None = None

    viewer_ctx = None
    if view:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)

    recorder = None
    if gif_path is not None:
        from sim_benchmark.gif import GifRecorder

        recorder = GifRecorder(sim.model, label=gif_path.stem)

    try:
        for k in range(n_ticks):
            t = k * dt
            targets = script.targets(t)
            grips = script.grips(t)
            if placer_offset is not None:
                pos, rot = targets[scenario.place_side]
                targets[scenario.place_side] = (
                    pos - np.array([placer_offset[0], placer_offset[1], 0.0]),
                    rot,
                )

            t0 = time.perf_counter()
            q_cmd = method.solve(targets, dt)
            solve_ms.append((time.perf_counter() - t0) * 1e3)

            sim.set_arm_targets(q_cmd)
            sim.set_target_markers(targets)

            # Mock gripper: keep trying to attach while commanded closed
            # (the EE may still be converging onto the payload), release
            # on open.
            for side in SIDES:
                if grips[side]:
                    sim.try_attach(side)
                else:
                    sim.release(side)
            if sim.attached_side == scenario.place_side:
                held_by_placer = True
                if placer_offset is None:
                    ee_pos, _ = sim.eef_pose(scenario.place_side)
                    placer_offset = sim.payload_pos() - ee_pos

            sim.step(n_substeps)

            payload_path.append(sim.payload_pos())
            for side in SIDES:
                pos, _ = sim.eef_pose(side)
                track_errs.append(float(np.linalg.norm(targets[side][0] - pos)))

            if recorder is not None:
                recorder.maybe_capture(sim.data, t)

            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                sleep = dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if recorder is not None and gif_path is not None:
            recorder.save(gif_path)
            recorder.close()
            print(f"  saved {gif_path}")

    final = sim.payload_pos()
    place_err = float(np.linalg.norm(final[:2] - scenario.target_pos[:2]))
    return {
        "scenario": scenario.index,
        "pick_side": scenario.pick_side,
        "success": bool(place_err < SUCCESS_RADIUS and sim.attached_side is None),
        "handover_ok": held_by_placer,
        "place_err_mm": place_err * 1e3,
        "track_err_mean_mm": float(np.mean(track_errs) * 1e3),
        "track_err_max_mm": float(np.max(track_errs) * 1e3),
        "solve_ms_mean": float(np.mean(solve_ms)),
        "duration_s": script.duration,
        "payload_path": np.asarray(payload_path),
    }


def aggregate(episodes: list[dict]) -> dict:
    place_errs = np.array([e["place_err_mm"] for e in episodes])
    return {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "handover_rate": float(np.mean([e["handover_ok"] for e in episodes])),
        "place_err_mean_mm": float(place_errs.mean()),
        "place_err_p95_mm": float(np.percentile(place_errs, 95)),
        "track_err_mean_mm": float(np.mean([e["track_err_mean_mm"] for e in episodes])),
        "solve_ms_mean": float(np.mean([e["solve_ms_mean"] for e in episodes])),
        "episodes": len(episodes),
    }


def print_table(summary: dict[str, dict]) -> None:
    cols = [
        ("success_rate", "success"),
        ("handover_rate", "handover"),
        ("place_err_mean_mm", "place_err(mm)"),
        ("place_err_p95_mm", "place_p95(mm)"),
        ("track_err_mean_mm", "track_err(mm)"),
        ("solve_ms_mean", "solve(ms)"),
    ]
    header = f"{'method':<14}" + "".join(f"{label:>15}" for _, label in cols)
    print("\n=== pick-handover-place ===")
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(f"{name:<14}" + "".join(f"{m[key]:>15.3f}" for key, _ in cols))


def plot_results(
    scenarios: list[Scenario],
    episodes: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Scenario-arrow success maps + payload paths for scenario 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(episodes)

    # --- success map: one panel per method, arrows pick -> target
    fig, axes = plt.subplots(
        1, len(methods), figsize=(3.4 * len(methods), 4.2), squeeze=False
    )
    for col, name in enumerate(methods):
        ax = axes[0][col]
        for ep, sc in zip(episodes[name], scenarios):
            color = "tab:green" if ep["success"] else "tab:red"
            ax.annotate(
                "",
                xy=sc.target_pos[:2],
                xytext=sc.payload_pos[:2],
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
            )
            ax.plot(*sc.payload_pos[:2], "o", color=color, ms=3)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.plot([0, 0], [0.15, -0.15], "ks", ms=5)  # arm bases
        rate = np.mean([e["success"] for e in episodes[name]])
        ax.set_title(f"{name}\nsuccess {rate:.0%}", fontsize=10)
        ax.set_xlim(0.05, 0.45)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        if col == 0:
            ax.set_ylabel("y (m)")
    fig.suptitle("Pick→target scenarios (green = placed within 2 cm)")
    fig.tight_layout()
    path = out_dir / "handover_success_map.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")

    # --- payload trajectory, scenario 0, all methods (top + side view)
    sc = scenarios[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    for name in methods:
        p = episodes[name][0]["payload_path"]
        axes[0].plot(p[:, 0], p[:, 1], lw=1.4, label=name)
        axes[1].plot(p[:, 0], p[:, 2], lw=1.4, label=name)
    axes[0].plot(*sc.payload_pos[:2], "k^", ms=9, label="start")
    axes[0].plot(*sc.target_pos[:2], "k*", ms=13, label="target")
    circ = plt.Circle(
        sc.target_pos[:2], SUCCESS_RADIUS, fill=False, color="k", ls="--", lw=0.8
    )
    axes[0].add_patch(circ)
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    fig.suptitle(
        f"Payload trajectory, scenario 0 ({sc.pick_side} picks): "
        "top view (xy) and side view (xz)"
    )
    fig.tight_layout()
    path = out_dir / "handover_payload_paths.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS)
    )
    parser.add_argument("--n-scenarios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=int,
        default=None,
        help="Run only these scenario indices",
    )
    parser.add_argument("--view", action="store_true", help="Live MuJoCo viewer")
    parser.add_argument("--save", type=str, default=None, help="Save results JSON")
    parser.add_argument("--plot", type=str, default=None, metavar="DIR")
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        metavar="DIR",
        help="Record an animated GIF per (scenario, method) to DIR",
    )
    args = parser.parse_args()

    print(f"Sampling {args.n_scenarios} feasible scenarios (seed {args.seed})...")
    scenarios = generate_scenarios(args.n_scenarios, seed=args.seed)
    if args.scenarios is not None:
        scenarios = [s for s in scenarios if s.index in args.scenarios]

    sim = DualArmSim()
    episodes: dict[str, list[dict]] = {}
    for name in args.methods:
        episodes[name] = []
        for sc in scenarios:
            gif_path = (
                Path(args.gif) / f"handover_s{sc.index:02d}_{name}.gif"
                if args.gif
                else None
            )
            ep = run_episode(sim, METHODS[name], sc, view=args.view, gif_path=gif_path)
            status = "ok " if ep["success"] else "FAIL"
            print(
                f"▶ {name:<13} #{sc.index:02d} [{status}] "
                f"place_err {ep['place_err_mm']:6.1f} mm",
                flush=True,
            )
            episodes[name].append(ep)

    summary = {name: aggregate(eps) for name, eps in episodes.items()}
    print_table(summary)

    if args.plot:
        plot_results(scenarios, episodes, Path(args.plot))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            name: {
                "summary": summary[name],
                "episodes": [
                    {k: v for k, v in ep.items() if k != "payload_path"} for ep in eps
                ],
            }
            for name, eps in episodes.items()
        }
        out.write_text(json.dumps(serializable, indent=2))
        print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
