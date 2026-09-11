#!/usr/bin/env python3
"""Fold one or more sweep journals into the tables the report and paper use.

A sweep writes a journal of scored episodes and a markdown table of its own
cells. That is not enough on its own for two reasons. A campaign is run in
stages -- a short pass over every cell, then a longer pass over the ones worth
resolving -- and the two must be read as one table, with the episode count
stated per row because it differs between them. And the paper needs the same
numbers as LaTeX, which nothing here produced.

So: journals in, one folded table out, in whichever of the two formats is
asked for. Later stages win, so re-running a cell replaces its earlier rows
rather than averaging two different sample sizes together.

    venv/bin/python tool/export_chunking_table.py \\
        outputs/chunking/stage_a.md.episodes.jsonl \\
        outputs/chunking/stage_b.md.episodes.jsonl \\
        --latex --caption "..." --label tab:chunking
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from actoris_harena.deploy.sweep_journal import load_rows, row_key  # noqa: E402

from tool.run_policy_sim import _fold, _table  # noqa: E402

#: How a cell label is spelled in prose. The machine-readable labels name a
#: strategy and its tuning; the paper names what the strategy DOES, because a
#: reader cannot be expected to know our vocabulary.
PROSE = {
    "sync": "hold for the reply",
    "append": "queue behind",
    "replace": "discard and replace",
    "receding": "execute a fraction",
    "blend": "cross-fade",
    "ensemble": "weighted average",
}


def merge(paths: "list[Path]") -> "list[dict]":
    """Every journal's episodes, later files overriding earlier ones per cell.

    A cell re-run at a larger sample size supersedes its short pass entirely:
    keeping both would fold ten episodes and thirty into one mean whose
    denominator is a fiction.
    """
    by_cell: "dict[str, dict[str, dict]]" = {}
    order: "list[str]" = []
    for path in paths:
        rows = load_rows(path)
        cells = {str(r.get("cell")) for r in rows if r.get("cell")}
        for cell in cells:
            if cell not in by_cell:
                order.append(cell)
            by_cell[cell] = {}
        for row in rows:
            cell = str(row.get("cell") or "")
            trial = row.get("trial")
            if not cell or trial is None:
                continue
            by_cell[cell][row_key(cell, int(trial))] = row
    out: "list[dict]" = []
    for cell in order:
        out.extend(by_cell.get(cell, {}).values())
    return out


def _fmt(value, spec: str, dash: str = "--") -> str:
    return dash if value is None else format(value, spec)


def latex_table(folded: "list[dict]", caption: str, label: str) -> str:
    """A booktabs table of the folded cells, ranked as the sweep ranks them."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Splice & Tuning & $n$ & Success & Median $t$ (s) & Seam & Held & "
        r"Path (deg) \\",
        r"\midrule",
    ]
    for row in folded:
        cell = str(row["cell"])
        base, _, tuning = cell.partition("@")
        lines.append(
            " & ".join(
                [
                    PROSE.get(base, base),
                    tuning.replace("_", r"\_") if tuning else "--",
                    str(row["episodes"]),
                    rf"{row['success']:.0%}".replace("%", r"\%"),
                    _fmt(row.get("seconds_to_success_median"), ".1f"),
                    _fmt(row.get("seam_ratio"), ".2f"),
                    rf"{row['held_fraction']:.0%}".replace("%", r"\%"),
                    f"{row['path_length']:.0f}",
                ]
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def latency_of(path: Path, rows: "list[dict]") -> "int | None":
    """The pinned delay a journal was measured at, in ticks.

    Prefer what the rows record. Journals written before the pacing was
    recorded fall back to the delay in their own file name, which is how the
    campaign named them; a journal that answers neither is realtime.
    """
    for row in rows:
        if row.get("latency_ticks") is not None:
            return int(row["latency_ticks"])
    if any(row.get("pace") == "realtime" for row in rows):
        return None
    found = re.search(r"lat(\d+)", path.name)
    return int(found.group(1)) if found else None


def latency_table(
    journals: "list[Path]", caption: str, label: str, metric: str = "success"
) -> str:
    """Strategies down the side, pinned delays across the top.

    The comparison the campaign is actually about: not which splice wins, but
    how each one holds up as the delay grows towards the chunk's own duration,
    beyond which an aligning splice has no rows left to execute.
    """
    by_lat: "dict[int, dict[str, dict]]" = {}
    for path in journals:
        rows = load_rows(path)
        lat = latency_of(path, rows)
        if lat is None:
            continue
        by_lat.setdefault(lat, {})
        for folded in _fold(rows):
            by_lat[lat][str(folded["cell"])] = folded
    lats = sorted(by_lat)
    cells: "list[str]" = []
    for lat in lats:
        for name in by_lat[lat]:
            if name not in cells:
                cells.append(name)

    def render(cell: str, lat: int) -> str:
        row = by_lat[lat].get(cell)
        if row is None:
            return "--"
        if metric == "success":
            return rf"{row['success']:.0%}".replace("%", r"\%")
        return rf"{row['held_fraction']:.0%}".replace("%", r"\%")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(lats) + "}",
        r"\toprule",
        "Splice & " + " & ".join(f"{ln} ticks" for ln in lats) + r" \\",
        r"\midrule",
    ]
    for cell in cells:
        base, _, tuning = cell.partition("@")
        name = PROSE.get(base, base) + (f" ({tuning})" if tuning else "")
        lines.append(
            name + " & " + " & ".join(render(cell, ln) for ln in lats) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("journals", nargs="+", type=Path)
    ap.add_argument("--latex", action="store_true", help="emit LaTeX, not markdown")
    ap.add_argument("--caption", default="Action-chunk splice strategies.")
    ap.add_argument("--label", default="tab:chunking")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--by-latency",
        action="store_true",
        help="strategies down the side, pinned delays across the top",
    )
    ap.add_argument("--metric", choices=("success", "held"), default="success")
    args = ap.parse_args()

    if args.by_latency:
        text = latency_table(args.journals, args.caption, args.label, args.metric)
    else:
        rows = merge(args.journals)
        if not rows:
            raise SystemExit("❌ no scored episodes in those journals")
        folded = _fold(rows)
        text = (
            latex_table(folded, args.caption, args.label)
            if args.latex
            else _table(folded)
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"✓ wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
