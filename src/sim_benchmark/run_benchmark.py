#!/usr/bin/env python3
"""Benchmark Meta-Quest teleop methods on dual SO-101 arms in MuJoCo.

Sweeps mocked Quest hand trajectories (circles of several radii, table-plane
lines in several directions) across the registered teleop/IK methods and
reports tracking metrics, so the methods can be vetted in simulation before
touching the real robots.

Usage:
    python src/sim_benchmark/run_benchmark.py                     # full sweep
    python src/sim_benchmark/run_benchmark.py --methods pink_full mink
    python src/sim_benchmark/run_benchmark.py --view --methods dls \
        --trajectories circle_r5cm                            # live viewer
    python src/sim_benchmark/run_benchmark.py --save out.json
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

from sim_benchmark.constants import ARM_JOINTS, CONTROL_RATE_HZ, SIDES  # noqa: E402
from sim_benchmark.methods import (  # type: ignore[attr-defined]  # noqa: E402
    METHODS,
    MethodFactory,
)
from sim_benchmark.metrics import RunLog, compute_metrics  # noqa: E402
from sim_benchmark.mock_quest import MockTrajectory, default_suite  # noqa: E402
from sim_benchmark.scene import DualArmSim  # noqa: E402

METRIC_COLUMNS = [
    ("ik_err_mean_mm", "ik_err(mm)"),
    ("pos_err_mean_mm", "err_mean(mm)"),
    ("pos_err_p95_mm", "err_p95(mm)"),
    ("ori_err_mean_deg", "ori(deg)"),
    ("roll_lag_ms", "lag(ms)"),
    ("cmd_jerk_rms_rad_s3", "jerk_rms"),
    ("joint_vel_max_rad_s", "qd_max"),
    ("limit_margin_min_deg", "lim_margin(deg)"),
    ("solve_ms_mean", "solve(ms)"),
]


def run_episode(
    sim: DualArmSim,
    method_factory: MethodFactory,
    trajectory: MockTrajectory,
    view: bool = False,
    gif_path: Path | None = None,
) -> tuple[dict, RunLog]:
    """Run one (method, trajectory) episode; return (metrics, raw log)."""
    method = method_factory(sim.model)
    q0 = sim.neutral_q()
    sim.reset(q0)
    method.reset(q0)

    # Latch initial EE poses at "grip press", exactly like the real clutch.
    initial_poses = {side: sim.eef_pose(side) for side in SIDES}

    dt = 1.0 / CONTROL_RATE_HZ
    n_substeps = max(1, round(dt / sim.model.opt.timestep))
    n_ticks = int(trajectory.duration / dt)
    log = RunLog()

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
            targets = trajectory.targets(t, initial_poses)

            t0 = time.perf_counter()
            q_cmd = method.solve(targets, dt)
            solve_ms = (time.perf_counter() - t0) * 1e3

            sim.set_arm_targets(q_cmd)
            sim.set_target_markers(targets)
            sim.step(n_substeps)

            measured = {side: sim.eef_pose(side) for side in SIDES}
            log.add(
                t,
                targets,
                measured,
                sim.fk_eef_pos(q_cmd),
                q_cmd,
                sim.arm_q(),
                solve_ms,
            )

            if recorder is not None:
                recorder.maybe_capture(sim.data, t)

            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                # Real-time pacing so the motion is watchable.
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

    q_low = np.array([sim.model.joint(j).range[0] for j in ARM_JOINTS])
    q_high = np.array([sim.model.joint(j).range[1] for j in ARM_JOINTS])
    return compute_metrics(log, q_low, q_high), log


def plot_paths(logs: dict[str, dict[str, RunLog]], out_dir: Path) -> None:
    """Save top-view XY path plots (target vs commanded-FK vs measured)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for traj_name, by_method in logs.items():
        n = len(by_method)
        fig, axes = plt.subplots(
            2, n, figsize=(3.2 * n, 6.4), squeeze=False, sharex=True, sharey=True
        )
        for col, (method_name, log) in enumerate(by_method.items()):
            for row, side in enumerate(SIDES):
                ax = axes[row][col]
                tp = np.asarray(log.target_pos[side])
                ip = np.asarray(log.ik_pos[side])
                mp = np.asarray(log.measured_pos[side])
                ax.plot(tp[:, 0], tp[:, 1], "k--", lw=1.2, label="target")
                ax.plot(ip[:, 0], ip[:, 1], "C0", lw=1.0, label="IK cmd")
                ax.plot(mp[:, 0], mp[:, 1], "C3", lw=1.0, label="measured")
                ax.set_aspect("equal")
                if row == 0:
                    ax.set_title(method_name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{side} arm\ny (m)")
                if row == 1:
                    ax.set_xlabel("x (m)")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{traj_name} — top view (XY)")
        fig.tight_layout()
        path = out_dir / f"{traj_name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  saved {path}")


def print_table(results: dict[str, dict[str, dict]]) -> None:
    """results[trajectory][method] -> metrics."""
    for traj_name, by_method in results.items():
        print(f"\n=== {traj_name} ===")
        header = f"{'method':<14}" + "".join(
            f"{label:>16}" for _, label in METRIC_COLUMNS
        )
        print(header)
        print("-" * len(header))
        for method_name, m in by_method.items():
            row = f"{method_name:<14}" + "".join(
                f"{m[key]:>16.3f}" for key, _ in METRIC_COLUMNS
            )
            print(row)

    # Cross-trajectory summary: mean position error per method.
    print("\n=== summary (mean err_mean across trajectories, mm) ===")
    methods = list(next(iter(results.values())).keys())
    for method_name in methods:
        vals = [results[t][method_name]["pos_err_mean_mm"] for t in results]
        print(f"{method_name:<14}{np.mean(vals):>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS)
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=None,
        help="Trajectory names to run (default: all). See mock_quest.default_suite.",
    )
    parser.add_argument("--view", action="store_true", help="Live MuJoCo viewer")
    parser.add_argument("--save", type=str, default=None, help="Save metrics JSON")
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="DIR",
        help="Save top-view XY path plots (target vs IK vs measured) to DIR",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        metavar="DIR",
        help="Record an animated GIF per (trajectory, method) to DIR",
    )
    args = parser.parse_args()

    suite = default_suite()
    if args.trajectories:
        known = {t.name for t in suite}
        unknown = set(args.trajectories) - known
        if unknown:
            parser.error(
                f"Unknown trajectories {sorted(unknown)} (known: {sorted(known)})"
            )
        suite = [t for t in suite if t.name in args.trajectories]

    sim = DualArmSim()
    results: dict[str, dict[str, dict]] = {}
    logs: dict[str, dict[str, RunLog]] = {}
    for traj in suite:
        results[traj.name] = {}
        logs[traj.name] = {}
        for name in args.methods:
            print(f"▶ {traj.name} / {name} ...", flush=True)
            gif_path = Path(args.gif) / f"{traj.name}_{name}.gif" if args.gif else None
            metrics, log = run_episode(
                sim, METHODS[name], traj, view=args.view, gif_path=gif_path
            )
            results[traj.name][name] = metrics
            logs[traj.name][name] = log

    print_table(results)

    if args.plot:
        plot_paths(logs, Path(args.plot))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
