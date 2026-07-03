"""Tracking-quality metrics for one (method, trajectory) benchmark run."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim_benchmark.constants import SIDES


@dataclass
class RunLog:
    """Per-tick log of one benchmark episode."""

    times: list[float] = field(default_factory=list)
    target_pos: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    measured_pos: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    ik_pos: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    q_cmd: list[np.ndarray] = field(default_factory=list)
    q_meas: list[np.ndarray] = field(default_factory=list)
    solve_ms: list[float] = field(default_factory=list)

    def add(
        self,
        t: float,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        measured: dict[str, tuple[np.ndarray, np.ndarray]],
        ik_pos: dict[str, np.ndarray],
        q_cmd: np.ndarray,
        q_meas: np.ndarray,
        solve_ms: float,
    ) -> None:
        self.times.append(t)
        for side in SIDES:
            self.target_pos[side].append(targets[side][0].copy())
            self.measured_pos[side].append(measured[side][0].copy())
            self.ik_pos[side].append(ik_pos[side].copy())
        self.q_cmd.append(q_cmd.copy())
        self.q_meas.append(q_meas.copy())
        self.solve_ms.append(solve_ms)


def compute_metrics(log: RunLog, q_low: np.ndarray, q_high: np.ndarray) -> dict:
    """Summarize a run: tracking error, smoothness, solver cost, limit hits."""
    t = np.asarray(log.times)
    dt = np.mean(np.diff(t))
    q_cmd = np.asarray(log.q_cmd)

    errors = []
    ik_errors = []
    for side in SIDES:
        tp = np.asarray(log.target_pos[side])
        mp = np.asarray(log.measured_pos[side])
        ip = np.asarray(log.ik_pos[side])
        errors.append(np.linalg.norm(tp - mp, axis=1))
        ik_errors.append(np.linalg.norm(tp - ip, axis=1))
    err = np.concatenate(errors)
    ik_err = np.concatenate(ik_errors)

    qd = np.gradient(q_cmd, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    jerk = np.gradient(qdd, dt, axis=0)

    margin = np.minimum(q_cmd - q_low, q_high - q_cmd)
    solve_ms = np.asarray(log.solve_ms)

    return {
        "pos_err_mean_mm": float(err.mean() * 1e3),
        "pos_err_p95_mm": float(np.percentile(err, 95) * 1e3),
        "pos_err_max_mm": float(err.max() * 1e3),
        "ik_err_mean_mm": float(ik_err.mean() * 1e3),
        "ik_err_p95_mm": float(np.percentile(ik_err, 95) * 1e3),
        "cmd_jerk_rms_rad_s3": float(np.sqrt((jerk**2).mean())),
        "joint_vel_max_rad_s": float(np.abs(qd).max()),
        "limit_margin_min_deg": float(np.rad2deg(margin.min())),
        "solve_ms_mean": float(solve_ms.mean()),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
        "ticks": int(len(t)),
    }
