"""Aggregate a long sim-VLA run directory into results.md + results.json.

Walks the layout ``test/system/long_vla_sim.sh`` writes::

    <run>/oracle_gate/<task>_<oracle>.json          dry-run oracle stats
    <run>/collect_stats/<mode>_<task>.json          collection stats
    <run>/<mode>/<task>/<policy>/val/step_*.json    per-checkpoint validation
    <run>/<mode>/<task>/<policy>/selected.json      the chosen checkpoint
    <run>/<mode>/<task>/<policy>/eval/results.json  final 30-seed evaluation

and writes ``results.md`` (human tables) and ``results.json`` (everything,
machine-readable) into the run directory. Missing cells (``--only`` runs,
skipped modes) are reported as absent rather than failing.

Usage: ``python -m sim_datagen.report <run_dir>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MODES = ["simple", "full"]
TASKS = ["single", "handover"]
POLICIES = ["act", "diffusion", "pi05"]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def collect(run_dir: Path) -> dict[str, Any]:
    """Gather every stats/val/eval JSON under ``run_dir`` into one dict."""
    out: dict[str, Any] = {"run_dir": str(run_dir), "cells": []}

    gate = {}
    for f in sorted((run_dir / "oracle_gate").glob("*.json")):
        gate[f.stem] = _load(f)
    out["oracle_gate"] = gate

    coll = {}
    for f in sorted((run_dir / "collect_stats").glob("*.json")):
        coll[f.stem] = _load(f)
    out["collection"] = coll

    for mode in MODES:
        for task in TASKS:
            for policy in POLICIES:
                cell_dir = run_dir / mode / task / policy
                if not cell_dir.is_dir():
                    continue
                cell: dict[str, Any] = {
                    "mode": mode,
                    "task": task,
                    "policy": policy,
                }
                cell["selected"] = _load(cell_dir / "selected.json")
                cell["val"] = {
                    f.stem: _load(f) for f in sorted((cell_dir / "val").glob("*.json"))
                }
                cell["eval"] = _load(cell_dir / "eval" / "results.json")
                out["cells"].append(cell)
    return out


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def render_markdown(data: dict[str, Any]) -> str:
    """Human-readable results.md from the collected dict."""
    lines = ["# Sim-VLA long-run results", "", f"Run: `{data['run_dir']}`", ""]

    if data["oracle_gate"]:
        lines += ["## Oracle gate (dry-run success rates)", ""]
        lines += ["| cell | oracle | success | attempts |", "|---|---|---|---|"]
        for name, g in data["oracle_gate"].items():
            if not g:
                continue
            lines.append(
                f"| {g.get('task')} | {g.get('oracle_mode')} | "
                f"{_fmt_pct(g.get('oracle_success_rate'))} | "
                f"{g.get('episode_attempts')} |"
            )
        lines.append("")

    if data["collection"]:
        lines += ["## Collection", ""]
        lines += [
            "| dataset | episodes | per-seed success | attempts |",
            "|---|---|---|---|",
        ]
        for name, c in data["collection"].items():
            if not c:
                continue
            lines.append(
                f"| {name} | {c.get('episodes_collected')} | "
                f"{_fmt_pct(c.get('per_seed_success_rate'))} | "
                f"{c.get('episode_attempts')} |"
            )
        lines.append("")

    cells = data["cells"]
    if cells:
        lines += ["## Final evaluation (selected checkpoint, EVAL seeds)", ""]
        header = "| mode | task | " + " | ".join(POLICIES) + " |"
        lines += [header, "|---|---|" + "---|" * len(POLICIES)]
        for mode in MODES:
            for task in TASKS:
                row = [mode, task]
                any_cell = False
                for policy in POLICIES:
                    cell = next(
                        (
                            c
                            for c in cells
                            if (c["mode"], c["task"], c["policy"])
                            == (mode, task, policy)
                        ),
                        None,
                    )
                    ev = cell and cell.get("eval")
                    if ev:
                        any_cell = True
                        row.append(
                            f"{_fmt_pct(ev['success_rate'])} "
                            f"({ev['place_err_mm_mean']:.0f} mm)"
                        )
                    else:
                        row.append("—")
                if any_cell:
                    lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("Cell format: success rate (mean place error).")
        lines.append("")

        lines += ["## Selected checkpoints (validation)", ""]
        lines += [
            "| mode | task | policy | step | val success |",
            "|---|---|---|---|---|",
        ]
        for c in cells:
            sel = c.get("selected")
            if not sel:
                continue
            lines.append(
                f"| {c['mode']} | {c['task']} | {c['policy']} | "
                f"{sel.get('step')} | {_fmt_pct(sel.get('val_success'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m sim_datagen.report <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    data = collect(run_dir)
    (run_dir / "results.json").write_text(json.dumps(data, indent=2))
    md = render_markdown(data)
    (run_dir / "results.md").write_text(md)
    print(md)
    print(f"📝 wrote {run_dir / 'results.md'} and results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
