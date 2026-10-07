#!/usr/bin/env python
"""Where in each episode a gripper closes on the garment.

On this rig a gripper rests nearly closed (about 0.02), opens to 0.15-0.3 on
the approach, and closes again on the cloth -- fully, because the cloth is thin.
So a grasp is the COMMAND falling from open to closed: above ``--high`` and,
within ``--within`` frames, below ``--low``. The frame where it first drops
below ``--low`` is the onset.

For each onset the tool also prints the frame to start a prediction window
from, ``--lead`` frames earlier, so a world model that predicts a second ahead
is shown the open gripper and asked to predict the contact.

    venv/bin/python tool/find_grasps.py --dataset <root> --episodes 58-64
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def grasp_onsets(
    command: np.ndarray, high: float = 0.1, low: float = 0.05, within: int = 15
) -> "list[int]":
    """Frames where ``command`` falls from above ``high`` to below ``low``."""
    onsets = []
    for t in range(1, len(command)):
        if command[t] < low <= command[t - 1]:
            recent = command[max(0, t - within) : t]
            if recent.size and recent.max() > high:
                onsets.append(t)
    return onsets


def main() -> int:
    import pandas as pd

    from common.robot_schema import GRIPPER_COLUMNS
    from tool.eval_world_model import parse_range

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", default="58-64")
    parser.add_argument("--high", type=float, default=0.1)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--within", type=int, default=15)
    parser.add_argument("--lead", type=int, default=15)
    parser.add_argument("--json", default=None, help="also write the list here")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser()
    frames = pd.concat(
        pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))
    )
    names = {GRIPPER_COLUMNS[0]: "left", GRIPPER_COLUMNS[1]: "right"}
    found = []
    offset = 0  # where each episode starts once the episodes are loaded together
    for episode in parse_range(args.episodes):
        rows = frames[frames.episode_index == episode].sort_values("frame_index")
        actions = np.stack(rows["action"].values)
        for column in GRIPPER_COLUMNS:
            for onset in grasp_onsets(
                actions[:, column], args.high, args.low, args.within
            ):
                found.append(
                    {
                        "episode": episode,
                        "arm": names[column],
                        "column": column,
                        "onset": onset,
                        "start": max(0, onset - args.lead),
                        "loaded_index": offset + max(0, onset - args.lead),
                        "length": len(rows),
                    }
                )
        offset += len(rows)
    for item in found:
        print(
            f"episode {item['episode']:3d}  {item['arm']:5s} arm closes at frame "
            f"{item['onset']:4d}  -> start the window at {item['start']}"
        )
    print("--frames " + " ".join(str(i["loaded_index"]) for i in found))
    if args.json:
        Path(args.json).write_text(json.dumps(found, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
