#!/usr/bin/env python3
"""Export benchmark metrics JSON to LaTeX (booktabs) tables for the paper.

Handles both results shapes:
- run_benchmark.py:  results[trajectory][method] -> metrics
- run_envelope.py:   results[trajectory][method][oob_mode] -> metrics

Usage:
    python sim_benchmark/export_latex_tables.py \
        --input outputs/teleop_wrist_bench.json \
        --output paper/teleoperation/tables --prefix wrist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH_COLUMNS = [
    ("pos_err_mean_mm", r"$\bar{e}_{pos}$ (mm)"),
    ("pos_err_p95_mm", r"$e_{pos}^{95}$ (mm)"),
    ("ori_err_mean_deg", r"$\bar{e}_{ori}$ ($^\circ$)"),
    ("roll_lag_ms", r"lag (ms)"),
    ("cmd_jerk_rms_rad_s3", r"jerk (rad/s$^3$)"),
    ("joint_vel_max_rad_s", r"$\dot{q}_{max}$"),
    ("solve_ms_mean", r"solve (ms)"),
]

ENVELOPE_COLUMNS = [
    ("oob_time_s", r"$t_{out}$ (s)"),
    ("pos_err_while_oob_mm", r"$\bar{e}_{emit}$ (mm)"),
    ("raw_err_while_oob_mm", r"$\bar{e}_{raw}$ (mm)"),
    ("qd_max_oob_rad_s", r"$\dot{q}_{max}^{out}$"),
    ("recovery_time_s", r"$t_{rec}$ (s)"),
    ("cmd_jerk_rms_rad_s3", r"jerk (rad/s$^3$)"),
]


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "--"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def _escape(name: str) -> str:
    return name.replace("_", r"\_")


def _tabular(
    rows: list[tuple[str, dict]], columns: list[tuple[str, str]], row_label: str
) -> str:
    lines = [
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        row_label + " & " + " & ".join(label for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for name, metrics in rows:
        cells = " & ".join(_fmt(metrics.get(key, float("nan"))) for key, _ in columns)
        lines.append(f"{_escape(name)} & {cells}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def export(results: dict, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for traj_name, by_method in results.items():
        first = next(iter(by_method.values()))
        nested = isinstance(first, dict) and all(
            isinstance(v, dict) for v in first.values()
        )
        if nested:  # envelope shape: method -> mode -> metrics
            for method_name, by_mode in by_method.items():
                rows = list(by_mode.items())
                tex = _tabular(rows, ENVELOPE_COLUMNS, "policy")
                path = out_dir / f"{prefix}_{traj_name}_{method_name}.tex"
                path.write_text(tex)
                print(f"  wrote {path}")
        else:  # benchmark shape: method -> metrics
            rows = list(by_method.items())
            tex = _tabular(rows, BENCH_COLUMNS, "method")
            path = out_dir / f"{prefix}_{traj_name}.tex"
            path.write_text(tex)
            print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default="bench")
    args = parser.parse_args()
    results = json.loads(Path(args.input).read_text())
    export(results, Path(args.output), args.prefix)


if __name__ == "__main__":
    main()
