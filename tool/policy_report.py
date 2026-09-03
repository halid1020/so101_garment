#!/usr/bin/env python3
"""What the rollouts say, across every run on this machine.

A rollout leaves a directory behind (``common.policy_log``); a comparison
between two checkpoints needs many of them read together. This is that reader:
it walks ``$SO101_OUTPUT_DIR/policy_runs/`` and prints one row per run, then one
row per CHECKPOINT -- which is the number an ablation is actually reported from.

    venv/bin/python tool/policy_report.py                 # every run
    venv/bin/python tool/policy_report.py --runs 20       # the last 20
    venv/bin/python tool/policy_report.py --checkpoint act  # matching runs only
    venv/bin/python tool/policy_report.py --json           # the same, machine-readable

It reads and prints; it never deletes a run or edits a verdict.

Two conventions, both from ``policy_log``. A ``discard`` leaves the denominator
rather than counting against the policy -- an attempt nobody thinks should count
is not a failure -- so a run of three attempts with one discarded is scored out
of two. And where an attempt was judged more than once, the LAST verdict is the
one that counts, because looking again is the normal reason to judge twice.

A run nobody judged shows a rate of ``—`` rather than 0%: unscored is not the
same as failed, and printing zero there would quietly libel a policy that was
never assessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.policy_log import runs_root, score, trial_verdicts  # noqa: E402


def read_meta(run: Path) -> dict:
    try:
        return json.loads((run / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def trial_seconds(run: Path) -> "dict[int, float]":
    """How long each attempt lasted, from the tick log. Empty if there is none.

    A run killed before ``close()`` has no ``ticks.parquet`` at all -- the ticks
    are held in memory until then -- which is a normal way for one to end, so
    this reports nothing rather than failing.
    """
    path = run / "ticks.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["t", "trial"])
    except Exception:  # noqa: BLE001 -- a diagnostic must not die on one run
        return {}
    if frame.empty:
        return {}
    span = frame.groupby("trial")["t"].agg(["min", "max"])
    return {int(k): float(v) for k, v in (span["max"] - span["min"]).items()}


def describe_checkpoint(meta: dict) -> str:
    """A checkpoint path shortened to the part that identifies the run.

    The full path is a scratch directory a dozen segments deep and identical
    across every row but one. What tells two apart is the RUN directory --
    ``<dataset>__<cameras>`` -- and the policy under it, so that is what is
    printed and everything above it is dropped.
    """
    raw = str(meta.get("checkpoint") or "")
    if not raw:
        return meta.get("policy_type") or "(unrecorded)"
    parts = Path(raw).parts
    if "train" in parts:
        cut = parts.index("train")
        return "/".join(parts[max(0, cut - 1) : cut + 2])
    return Path(raw).name or raw


def collect(root: Path, limit: int = 0, match: str = "") -> "list[dict]":
    """One record per run directory, newest last (the stamp sorts that way)."""
    out: "list[dict]" = []
    for run in sorted(p for p in root.glob("*") if p.is_dir()):
        meta = read_meta(run)
        name = describe_checkpoint(meta)
        if match and match not in name and match not in run.name:
            continue
        seconds = trial_seconds(run)
        counted = {
            trial: v
            for trial, v in trial_verdicts(run).items()
            if v.get("outcome") != "discard"
        }
        durations = [seconds[t] for t in counted if t in seconds]
        out.append(
            {
                "run": run.name,
                "path": str(run),
                "checkpoint": name,
                "policy": meta.get("policy_type") or "",
                "task": meta.get("task") or "",
                "frames": bool(meta.get("frames")),
                "mean_trial_s": (sum(durations) / len(durations))
                if durations
                else None,
                **score(run),
            }
        )
    return out[-limit:] if limit else out


def _rate(record: dict) -> str:
    rate = record.get("rate")
    return "—" if rate is None else f"{rate * 100:.0f}%"


def _secs(value) -> str:
    return "—" if value is None else f"{value:.0f}s"


def _table(rows: "list[list[str]]", headers: "list[str]") -> str:
    widths = [
        max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def by_checkpoint(records: "list[dict]") -> "list[dict]":
    """Every run of one checkpoint pooled. This is the ablation's row."""
    pooled: "dict[str, dict]" = {}
    for record in records:
        agg = pooled.setdefault(
            record["checkpoint"],
            {
                "checkpoint": record["checkpoint"],
                "runs": 0,
                "trials": 0,
                "successes": 0,
                "discarded": 0,
            },
        )
        agg["runs"] += 1
        for key in ("trials", "successes", "discarded"):
            agg[key] += record[key]
    for agg in pooled.values():
        agg["rate"] = (agg["successes"] / agg["trials"]) if agg["trials"] else None
    return sorted(pooled.values(), key=lambda a: a["checkpoint"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", default=None, help="Run-log root (default: outputs/policy_runs)"
    )
    parser.add_argument("--runs", type=int, default=0, help="Only the last N runs")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Only runs whose checkpoint or stamp contains this",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else runs_root()
    if not root.is_dir():
        raise SystemExit(f"❌ no run logs at {root}")

    records = collect(root, limit=args.runs, match=args.checkpoint)
    if args.json:
        print(
            json.dumps(
                {"runs": records, "checkpoints": by_checkpoint(records)}, indent=2
            )
        )
        return

    if not records:
        raise SystemExit(
            f"no runs under {root}"
            + (f" matching {args.checkpoint!r}" if args.checkpoint else "")
        )

    print(f"\n📁 {root}  ({len(records)} run(s))\n")
    print(
        _table(
            [
                [
                    r["run"],
                    r["checkpoint"],
                    str(r["trials"]),
                    str(r["successes"]),
                    _rate(r),
                    str(r["discarded"]),
                    _secs(r["mean_trial_s"]),
                    "yes" if r["frames"] else "",
                ]
                for r in records
            ],
            ["run", "checkpoint", "trials", "wins", "rate", "void", "mean", "frames"],
        )
    )

    pooled = by_checkpoint(records)
    if len(pooled) > 1 or len(records) > 1:
        print("\n📊 pooled by checkpoint\n")
        print(
            _table(
                [
                    [
                        a["checkpoint"],
                        str(a["runs"]),
                        str(a["trials"]),
                        str(a["successes"]),
                        _rate(a),
                        str(a["discarded"]),
                    ]
                    for a in pooled
                ],
                ["checkpoint", "runs", "trials", "wins", "rate", "void"],
            )
        )
    unscored = sum(1 for r in records if r["trials"] == 0)
    if unscored:
        print(
            f"\nℹ️  {unscored} run(s) were never judged, and are counted nowhere "
            "above. Judge an attempt from the rollout page while it is in front "
            "of you."
        )
    print()


if __name__ == "__main__":
    main()
