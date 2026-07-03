#!/usr/bin/env python3
"""Package the benchmark report into a shareable, self-contained zip.

Collects markdowns/teleop_benchmark_results.md (asset paths rewritten),
the plots, and the report-referenced GIFs into
outputs/teleop_benchmark_report/, renders a double-clickable report.html,
and zips the folder to outputs/teleop_benchmark_report.zip.

Run after regenerating results:
    venv/bin/python sim_benchmark/package_report.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import markdown  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "outputs" / "teleop_benchmark_plots"
GIFS_DIR = REPO_ROOT / "outputs" / "teleop_benchmark_gifs"
REPORT_MD = REPO_ROOT / "markdowns" / "teleop_benchmark_results.md"
STAGE = REPO_ROOT / "outputs" / "teleop_benchmark_report"

# GIFs embedded in the report (the 35 per-episode tracking GIFs stay out
# to keep the zip small).
REPORT_GIFS = [
    "compare_handover_s00.gif",
    "compare_circle_r5cm.gif",
    "compare_line_90deg.gif",
    *(
        f"handover_s00_{m}.gif"
        for m in ("pink_full", "pink_relaxed", "dls", "mink", "scipy_ls")
    ),
]

APPENDIX = """
---

## Appendix: static figures

**Handover success map** — pick→target arrows for all 30 scenarios, per
method (green = placed within 2 cm):

![success map](plots/handover_success_map.png)

**Payload trajectory, scenario 0** — top and side view, all methods:

![payload paths](plots/handover_payload_paths.png)

**Tracking paths, circle r = 5 cm** — target vs IK command vs measured,
per method and arm:

![circle paths](plots/circle_r5cm.png)

**Tracking paths, line 90°:**

![line90 paths](plots/line_90deg.png)

All remaining trajectory plots are in `plots/`.
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SO-101 Teleop Benchmark Results</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1080px; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.55; color: #1a1a1a; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; font-size: 0.92em; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
pre code {{ display: block; padding: 10px; overflow-x: auto; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
</style></head><body>
{body}
</body></html>"""


def main() -> None:
    missing = [
        p
        for p in (REPORT_MD, PLOTS_DIR, *(GIFS_DIR / g for g in REPORT_GIFS))
        if not p.exists()
    ]
    if missing:
        sys.exit(
            "Missing inputs (regenerate results/GIFs first):\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "plots").mkdir(parents=True)
    (STAGE / "gifs").mkdir()

    for png in PLOTS_DIR.glob("*.png"):
        shutil.copy(png, STAGE / "plots" / png.name)
    for name in REPORT_GIFS:
        shutil.copy(GIFS_DIR / name, STAGE / "gifs" / name)

    text = REPORT_MD.read_text()
    text = text.replace("../outputs/teleop_benchmark_gifs/", "gifs/")
    text = text.replace("`outputs/teleop_benchmark_plots/", "`plots/")
    text = text.replace("outputs/teleop_benchmark_gifs/", "gifs/")
    text += APPENDIX
    (STAGE / "README.md").write_text(text)

    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    (STAGE / "report.html").write_text(HTML_TEMPLATE.format(body=body))

    zip_path = shutil.make_archive(str(STAGE), "zip", STAGE.parent, STAGE.name)
    n_files = sum(1 for p in STAGE.rglob("*") if p.is_file())
    size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"Packaged {n_files} files -> {zip_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
