#!/usr/bin/env python3
"""Benchmark out-of-envelope (OOE) target policies on dual SO-101 arms.

Sweeps deliberately out-of-reach mock trajectories (radial overshoot, floor
swoop, boundary slide — see mock_quest.envelope_suite) across the four OOE
policies (warn / project / freeze / slow, common.workspace_envelope) and the
selected IK methods, and reports how each combination behaves while the
operator's hand is outside the reachable workspace and on re-entry.

Usage:
    python src/sim_benchmark/run_envelope.py                     # full sweep
    python src/sim_benchmark/run_envelope.py --oob-modes project freeze
    python src/sim_benchmark/run_envelope.py --methods pink_relaxed telegrip \
        --save outputs/envelope.json --plot outputs/envelope_plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
for _p in (str(_repo_root), str(_repo_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pinocchio as pin  # noqa: E402

from common.workspace_envelope import (  # noqa: E402
    OOE_POLICIES,
    ArmEnvelope,
    build_envelopes,
    make_policies,
)
from sim_benchmark.constants import (  # noqa: E402
    ARM_JOINTS,
    CONTROL_RATE_HZ,
    DUAL_URDF_PATH,
    SIDES,
)
from sim_benchmark.methods import METHODS, MethodFactory  # noqa: E402
from sim_benchmark.metrics import RunLog, compute_envelope_metrics  # noqa: E402
from sim_benchmark.mock_quest import MockTrajectory, envelope_suite  # noqa: E402
from sim_benchmark.scene import DualArmSim  # noqa: E402
from sim_benchmark.scene_rig import make_sim  # noqa: E402

METRIC_COLUMNS = [
    ("oob_time_s", "oob(s)"),
    ("pos_err_while_oob_mm", "err_oob(mm)"),
    ("raw_err_while_oob_mm", "raw_oob(mm)"),
    ("qd_max_oob_rad_s", "qd_oob"),
    ("recovery_time_s", "recover(s)"),
    ("cmd_jerk_rms_rad_s3", "jerk_rms"),
    ("pos_err_mean_mm", "err_mean(mm)"),
]

DEFAULT_METHODS = ["pink_relaxed", "pink_full", "telegrip"]


def _load_envelopes() -> dict[str, ArmEnvelope]:
    full = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
    gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
    model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))
    return build_envelopes(model)


def run_episode(
    sim: DualArmSim,
    method_factory: MethodFactory,
    trajectory: MockTrajectory,
    oob_mode: str,
    envelopes: dict[str, ArmEnvelope],
) -> tuple[dict, RunLog]:
    """Run one (method, trajectory, policy) episode; return (metrics, log)."""
    method = method_factory(sim.ik_model)
    q0 = sim.neutral_q()
    sim.reset(q0)
    method.reset(q0)
    policies = make_policies(oob_mode, envelopes)

    initial_poses = {side: sim.eef_pose(side) for side in SIDES}
    dt = 1.0 / CONTROL_RATE_HZ
    n_substeps = max(1, round(dt / sim.model.opt.timestep))
    n_ticks = int(trajectory.duration / dt)
    log = RunLog()

    for k in range(n_ticks):
        t = k * dt
        raw = trajectory.targets(t, initial_poses)
        clamped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_pos: dict[str, np.ndarray] = {}
        oob: dict[str, bool] = {}
        for side in SIDES:
            p_raw, rot = raw[side]
            p_out, status = policies[side].apply(p_raw, t)
            clamped[side] = (p_out, rot)
            raw_pos[side] = p_raw
            oob[side] = not status.inside

        t0 = time.perf_counter()
        q_cmd = method.solve(clamped, dt)
        solve_ms = (time.perf_counter() - t0) * 1e3

        sim.set_arm_targets(q_cmd)
        sim.set_target_markers(clamped)
        sim.step(n_substeps)

        measured = {side: sim.eef_pose(side) for side in SIDES}
        log.add(
            t,
            clamped,
            measured,
            sim.fk_eef_pos(q_cmd),
            q_cmd,
            sim.arm_q(),
            solve_ms,
            raw_targets=raw_pos,
            oob=oob,
        )

    q_low = np.array([sim.model.joint(j).range[0] for j in ARM_JOINTS])
    q_high = np.array([sim.model.joint(j).range[1] for j in ARM_JOINTS])
    return compute_envelope_metrics(log, q_low, q_high), log


def plot_episodes(
    logs: dict[str, dict[str, dict[str, RunLog]]],
    envelopes: dict[str, ArmEnvelope],
    out_dir: Path,
) -> None:
    """Per (trajectory, method): XY top view + radial distance vs time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    env = envelopes["right"]
    for traj_name, by_method in logs.items():
        for method_name, by_mode in by_method.items():
            n = len(by_mode)
            fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.8), squeeze=False)
            for col, (mode, log) in enumerate(by_mode.items()):
                rp = np.asarray(log.raw_target_pos["right"])
                tp = np.asarray(log.target_pos["right"])
                mp = np.asarray(log.measured_pos["right"])
                t = np.asarray(log.times)

                ax = axes[0][col]
                theta = np.linspace(-np.pi, np.pi, 200)
                for r_b in (env.r_min, env.r_max):
                    piv = env.pivot(np.array([0.3, -0.15, 0.1]))
                    ax.plot(
                        piv[0] + r_b * np.cos(theta),
                        piv[1] + r_b * np.sin(theta),
                        color="0.75",
                        lw=0.8,
                    )
                ax.plot(rp[:, 0], rp[:, 1], "k--", lw=1.1, label="raw target")
                ax.plot(tp[:, 0], tp[:, 1], "C2", lw=1.0, label="emitted")
                ax.plot(mp[:, 0], mp[:, 1], "C3", lw=1.0, label="measured")
                ax.set_aspect("equal")
                ax.set_title(f"{mode}", fontsize=10)
                if col == 0:
                    ax.set_ylabel("right arm — y (m)")
                    ax.legend(fontsize=7)

                ax = axes[1][col]
                r_raw = np.linalg.norm(
                    rp - np.array([env.pivot(p) for p in rp]), axis=1
                )
                r_meas = np.linalg.norm(
                    mp - np.array([env.pivot(p) for p in mp]), axis=1
                )
                ax.plot(t, r_raw, "k--", lw=1.1, label="raw r")
                ax.plot(t, r_meas, "C3", lw=1.0, label="measured r")
                ax.axhline(env.r_max, color="0.6", lw=0.8)
                ax.axhline(env.r_min, color="0.6", lw=0.8)
                ax.set_xlabel("t (s)")
                if col == 0:
                    ax.set_ylabel("radial dist (m)")
                    ax.legend(fontsize=7)
            fig.suptitle(f"{traj_name} — {method_name}")
            fig.tight_layout()
            path = out_dir / f"{traj_name}_{method_name}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            print(f"  saved {path}")


def print_table(results: dict[str, dict[str, dict[str, dict]]]) -> None:
    """results[trajectory][method][oob_mode] -> metrics."""
    for traj_name, by_method in results.items():
        for method_name, by_mode in by_method.items():
            print(f"\n=== {traj_name} / {method_name} ===")
            header = f"{'policy':<10}" + "".join(
                f"{label:>14}" for _, label in METRIC_COLUMNS
            )
            print(header)
            print("-" * len(header))
            for mode, m in by_mode.items():
                row = f"{mode:<10}" + "".join(
                    f"{m[key]:>14.3f}" for key, _ in METRIC_COLUMNS
                )
                print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS)
    )
    parser.add_argument(
        "--oob-modes",
        nargs="+",
        default=sorted(OOE_POLICIES),
        choices=sorted(OOE_POLICIES),
    )
    parser.add_argument("--trajectories", nargs="+", default=None)
    parser.add_argument(
        "--scene",
        choices=["rig", "plain"],
        default="rig",
        help="rig = collision-enabled printed-rig twin (default); plain = flat scene",
    )
    parser.add_argument("--save", type=str, default=None, help="Save metrics JSON")
    parser.add_argument("--plot", type=str, default=None, metavar="DIR")
    parser.add_argument(
        "--render-views",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help="Render 3-D reach-envelope figures (top/front/side/iso + section) "
        "to DIR and exit; DIR defaults to $SO101_OUTPUT_DIR/teleop_envelope_plots",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=0.0,
        help="Work-surface height in the IK frame for the envelope plots (m)",
    )
    args = parser.parse_args()

    if args.render_views is not None:
        from sim_benchmark.envelope_views import render_envelope_views

        out_root = os.environ.get("SO101_OUTPUT_DIR", "outputs")
        out_dir = args.render_views or str(Path(out_root) / "teleop_envelope_plots")
        paths = render_envelope_views(_load_envelopes(), out_dir, table_z=args.table_z)
        print("Rendered envelope views:")
        for p in paths:
            print(f"  {p}")
        return

    suite = envelope_suite()
    if args.trajectories:
        known = {t.name for t in suite}
        unknown = set(args.trajectories) - known
        if unknown:
            parser.error(
                f"Unknown trajectories {sorted(unknown)} (known: {sorted(known)})"
            )
        suite = [t for t in suite if t.name in args.trajectories]

    envelopes = _load_envelopes()
    sim = make_sim(args.scene)
    results: dict[str, dict[str, dict[str, dict]]] = {}
    logs: dict[str, dict[str, dict[str, RunLog]]] = {}
    for traj in suite:
        results[traj.name] = {}
        logs[traj.name] = {}
        for name in args.methods:
            results[traj.name][name] = {}
            logs[traj.name][name] = {}
            for mode in args.oob_modes:
                print(f"▶ {traj.name} / {name} / {mode} ...", flush=True)
                metrics, log = run_episode(sim, METHODS[name], traj, mode, envelopes)
                results[traj.name][name][mode] = metrics
                logs[traj.name][name][mode] = log

    print_table(results)

    if args.plot:
        plot_episodes(logs, envelopes, Path(args.plot))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
