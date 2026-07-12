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
    # Optional orientation logs (populated when the caller passes rotations).
    target_rot: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    measured_rot: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    # Optional envelope logs (populated by run_envelope.py): the RAW operator
    # target before the OOE policy, and whether it was outside the envelope.
    raw_target_pos: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )
    oob_flags: dict[str, list[bool]] = field(
        default_factory=lambda: {s: [] for s in SIDES}
    )

    def add(
        self,
        t: float,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        measured: dict[str, tuple[np.ndarray, np.ndarray]],
        ik_pos: dict[str, np.ndarray],
        q_cmd: np.ndarray,
        q_meas: np.ndarray,
        solve_ms: float,
        raw_targets: dict[str, np.ndarray] | None = None,
        oob: dict[str, bool] | None = None,
    ) -> None:
        self.times.append(t)
        for side in SIDES:
            self.target_pos[side].append(targets[side][0].copy())
            self.measured_pos[side].append(measured[side][0].copy())
            self.ik_pos[side].append(ik_pos[side].copy())
            self.target_rot[side].append(targets[side][1].copy())
            self.measured_rot[side].append(measured[side][1].copy())
            if raw_targets is not None:
                self.raw_target_pos[side].append(raw_targets[side].copy())
            if oob is not None:
                self.oob_flags[side].append(oob[side])
        self.q_cmd.append(q_cmd.copy())
        self.q_meas.append(q_meas.copy())
        self.solve_ms.append(solve_ms)


def _geodesic_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Per-tick geodesic angle (deg) between two stacks of 3x3 rotations."""
    tr = np.einsum("nij,nij->n", r_a, r_b)  # trace(r_a^T r_b)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _roll_angle(rots: np.ndarray) -> np.ndarray:
    """Signed roll of each rotation's local z about its tip axis (local x),
    measured against the horizontal reference (world z projected out of the
    tip). NaN where the tip is near-vertical."""
    tip = rots[:, :, 0]
    z_ax = rots[:, :, 2]
    u = np.zeros_like(tip)
    u[:, 2] = 1.0
    u = u - tip * tip[:, 2:3]
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = u / norm
    sin = np.einsum("ni,ni->n", np.cross(u, z_ax), tip)
    cos = np.einsum("ni,ni->n", u, z_ax)
    roll = np.arctan2(sin, cos)
    roll[np.abs(tip[:, 2]) >= 0.995] = np.nan
    return roll


def _roll_lag_ms(
    target_roll: np.ndarray, measured_roll: np.ndarray, dt: float
) -> float:
    """Phase lag (ms) of the measured roll behind the commanded roll via the
    peak of the cross-correlation; NaN when the roll signal is negligible."""
    ok = ~(np.isnan(target_roll) | np.isnan(measured_roll))
    a = target_roll[ok]
    b = measured_roll[ok]
    if len(a) < 20 or a.std() < np.deg2rad(2.0):
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(round(1.0 / dt))  # search +-1 s
    lags = np.arange(-max_lag, max_lag + 1)
    corr = [
        np.dot(a[max(0, -k) : len(a) - max(0, k)], b[max(0, k) : len(b) - max(0, -k)])
        for k in lags
    ]
    return float(lags[int(np.argmax(corr))] * dt * 1e3)


def compute_metrics(log: RunLog, q_low: np.ndarray, q_high: np.ndarray) -> dict:
    """Summarize a run: tracking error, smoothness, solver cost, limit hits."""
    t = np.asarray(log.times)
    dt = np.mean(np.diff(t))
    q_cmd = np.asarray(log.q_cmd)

    errors = []
    ik_errors = []
    ori_errors = []
    roll_lags = []
    for side in SIDES:
        tp = np.asarray(log.target_pos[side])
        mp = np.asarray(log.measured_pos[side])
        ip = np.asarray(log.ik_pos[side])
        errors.append(np.linalg.norm(tp - mp, axis=1))
        ik_errors.append(np.linalg.norm(tp - ip, axis=1))
        if log.target_rot[side]:
            tr = np.asarray(log.target_rot[side])
            mr = np.asarray(log.measured_rot[side])
            ori_errors.append(_geodesic_deg(tr, mr))
            roll_lags.append(_roll_lag_ms(_roll_angle(tr), _roll_angle(mr), dt))
    err = np.concatenate(errors)
    ik_err = np.concatenate(ik_errors)
    ori_err = np.concatenate(ori_errors) if ori_errors else np.array([np.nan])
    roll_lag = float(np.nanmean(roll_lags)) if roll_lags else float("nan")

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
        "ori_err_mean_deg": float(np.nanmean(ori_err)),
        "ori_err_p95_deg": float(np.nanpercentile(ori_err, 95)),
        "ori_err_max_deg": float(np.nanmax(ori_err)),
        "roll_lag_ms": roll_lag,
        "cmd_jerk_rms_rad_s3": float(np.sqrt((jerk**2).mean())),
        "joint_vel_max_rad_s": float(np.abs(qd).max()),
        "limit_margin_min_deg": float(np.rad2deg(margin.min())),
        "solve_ms_mean": float(solve_ms.mean()),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
        "ticks": int(len(t)),
    }


def compute_envelope_metrics(
    log: RunLog, q_low: np.ndarray, q_high: np.ndarray
) -> dict:
    """Metrics for an out-of-envelope episode (run_envelope.py).

    Requires raw_target_pos and oob_flags to be populated. Errors are split
    into the emitted-target error (how well the arm follows the policy's own
    command) and the raw-target error (how far behavior deviates from the
    operator's infeasible intent) while outside, plus re-entry recovery.
    """
    base = compute_metrics(log, q_low, q_high)
    t = np.asarray(log.times)
    dt = float(np.mean(np.diff(t)))
    q_cmd = np.asarray(log.q_cmd)
    qd = np.abs(np.gradient(q_cmd, dt, axis=0))

    oob_any = np.zeros(len(t), dtype=bool)
    emitted_oob: list[np.ndarray] = []
    raw_oob: list[np.ndarray] = []
    recovery_times: list[float] = []
    for side in SIDES:
        flags = np.asarray(log.oob_flags[side], dtype=bool)
        oob_any |= flags
        tp = np.asarray(log.target_pos[side])  # emitted (post-policy)
        rp = np.asarray(log.raw_target_pos[side])  # raw operator target
        mp = np.asarray(log.measured_pos[side])
        emitted_err = np.linalg.norm(tp - mp, axis=1)
        raw_err = np.linalg.norm(rp - mp, axis=1)
        if flags.any():
            emitted_oob.append(emitted_err[flags])
            raw_oob.append(raw_err[flags])
            # Recovery: after the LAST re-entry, first time the raw-target
            # error stays below 10 mm for 0.2 s.
            last_out = int(np.max(np.nonzero(flags)[0]))
            window = int(round(0.2 / dt))
            rec = float("nan")
            for k in range(last_out + 1, len(t) - window):
                if np.all(raw_err[k : k + window] < 0.010):
                    rec = t[k] - t[last_out]
                    break
            recovery_times.append(rec)

    out = dict(base)
    out.update(
        {
            "oob_time_s": float(oob_any.sum() * dt),
            "pos_err_while_oob_mm": (
                float(np.concatenate(emitted_oob).mean() * 1e3)
                if emitted_oob
                else float("nan")
            ),
            "raw_err_while_oob_mm": (
                float(np.concatenate(raw_oob).mean() * 1e3) if raw_oob else float("nan")
            ),
            "qd_max_oob_rad_s": (
                float(qd[oob_any].max()) if oob_any.any() else float("nan")
            ),
            "recovery_time_s": (
                float(np.nanmax(recovery_times)) if recovery_times else float("nan")
            ),
        }
    )
    return out
