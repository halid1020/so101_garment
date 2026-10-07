#!/usr/bin/env python3
"""Training and held-out action error at every saved step of one run.

A thin loop over ``tool/eval_action_mse.py``: each checkpoint the run kept
(``--keep-weights`` in the driver keeps every step's weights) is scored with the
same protocol as the report's main table -- LeRobot's own split, every 25th
frame, a pinned sampler -- so the last point of a curve IS that table's number,
which is the check that the curve measures what the table does.

Each step's ``action_mse.json`` is written under ``--out/<step>/`` and reused on
a second run, so an interrupted job resumes. ``curve.json`` collects them, with
the held-out training LOSS from the run's log where the run computed one
(``--eval_steps``), since that is what the trainer itself could see.

    venv/bin/python tool/score_curve.py --run <run>/train/act \\
        --dataset <root> --out outputs/analysis/<date>/act-all/curve
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_LOSS = re.compile(r"step (\d+): eval_loss=([0-9.eE+-]+)")


def checkpoints(run: Path) -> "dict[int, Path]":
    """``{step: pretrained_model}`` for every step whose weights survive."""
    out = {}
    for path in sorted((run / "checkpoints").glob("[0-9]*/pretrained_model")):
        out[int(path.parent.name)] = path
    return out


def eval_losses(log: "Path | None") -> "dict[int, float]":
    if log is None or not log.exists():
        return {}
    return {
        int(m.group(1)): float(m.group(2))
        for m in EVAL_LOSS.finditer(log.read_text(errors="replace"))
    }


def summary(result: dict, half: str) -> dict:
    """RMSE over the whole plan and over its first ten steps, in degrees."""
    per_step = result[half]["mse_per_step"]
    first = per_step[:10]
    return {
        "rmse": result[half]["rmse"],
        "rmse_first10": (sum(first) / len(first)) ** 0.5,
        "frames": result[half]["frames"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="<run dir>/train/<policy>")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--log", default=None, help="train log, for eval_loss")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run = Path(args.run).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    found = checkpoints(run)
    if not found:
        raise SystemExit(f"❌ no checkpoints with weights under {run}")
    print(f"📈 {len(found)} step(s): {', '.join(str(s) for s in found)}")

    curve: "dict[str, dict]" = {}
    for step, model in found.items():
        step_dir = out / f"{step:06d}"
        result_path = step_dir / "action_mse.json"
        if not result_path.exists():
            command = [
                sys.executable,
                str(ROOT / "tool/eval_action_mse.py"),
                "--checkpoint", str(model),
                "--dataset", args.dataset,
                "--every", str(args.every),
                "--out", str(step_dir),
                "--name", f"curve-{step}",
            ]  # fmt: skip
            if args.device:
                command += ["--device", args.device]
            print(f"\n### step {step}", flush=True)
            subprocess.run(command, check=True)
        result = json.loads(result_path.read_text())
        curve[str(step)] = {
            "train": summary(result, "train"),
            "validation": summary(result, "validation"),
        }

    payload = {
        "run": str(run),
        "policy": run.name,
        "every": args.every,
        "steps": curve,
        "eval_loss": {
            str(k): v
            for k, v in eval_losses(Path(args.log) if args.log else None).items()
        },
    }
    (out / "curve.json").write_text(json.dumps(payload, indent=1))
    print(f"\n📝 {out / 'curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
