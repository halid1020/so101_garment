"""What one autonomous rollout did, written down while it happens.

A failed grasp is over in a second and cannot be paused mid-air, so the run has
to leave enough behind to be examined afterwards: what the policy was shown,
what it planned from that, and what the arms actually did with the plan.

Two files, because the two rates are different. ``ticks.parquet`` has one row
per control tick -- the measured joints, the goal written, the mode, the queue
depth -- and is what a tracking plot is drawn from. ``chunks.jsonl`` has one
line per *plan*, appended the moment it lands, so a run that dies mid-episode
still leaves its plans on disk; a chunk is thirty-two actions and a state, and
the sequence numbers tie it back to the ticks that executed it.

Images are deliberately not stored: they are the largest part of an observation
by far, the cameras were already recorded during collection, and the questions
this log answers are about joints. Written by ``tool/run_policy.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

RUNS_SUBDIR = "policy_runs"
TICK_COLUMNS = ("t", "mode", "queue", "served", "state", "commanded")


def runs_root() -> Path:
    """Where run logs live: beside every other output of a session."""
    return (
        Path(os.environ.get("SO101_OUTPUT_DIR", "outputs")).expanduser() / RUNS_SUBDIR
    )


class RunLog:
    """One directory per rollout. Create it, feed it, close it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ticks: "list[dict]" = []
        self._chunks = open(self.root / "chunks.jsonl", "a", encoding="utf-8")
        self._last_seq = 0

    @classmethod
    def create(
        cls, task: str, source: str, hz: float, root: "Path | None" = None
    ) -> "RunLog":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log = cls(Path(root or runs_root()) / stamp)
        (log.root / "meta.json").write_text(
            json.dumps(
                {"task": task, "source": source, "hz": hz, "started": stamp}, indent=2
            )
        )
        return log

    def tick(
        self,
        t: float,
        state: np.ndarray,
        commanded: "np.ndarray | None",
        mode: str,
        queue: int,
    ) -> None:
        self._ticks.append(
            {
                "t": round(float(t), 4),
                "mode": mode,
                "queue": int(queue),
                "served": commanded is not None,
                "state": np.asarray(state, dtype=np.float32),
                "commanded": (
                    None if commanded is None else np.asarray(commanded, np.float32)
                ),
            }
        )

    def note_chunk(self, source) -> bool:
        """Append the source's latest plan, if it is one this log has not seen."""
        seq = int(getattr(source, "last_chunk_seq", 0) or 0)
        chunk = getattr(source, "last_chunk", None)
        if seq <= self._last_seq or chunk is None:
            return False
        self._last_seq = seq
        sent = getattr(source, "last_sent", lambda: None)()
        self._chunks.write(
            json.dumps(
                {
                    "seq": seq,
                    "at": round(float(getattr(source, "last_chunk_at", 0.0)), 4),
                    "state": (
                        None if sent is None else np.asarray(sent[0]).round(4).tolist()
                    ),
                    "actions": np.asarray(chunk, dtype=float).round(4).tolist(),
                    "round_trip_s": round(float(source.round_trip_s), 4),
                    "server_infer_s": round(
                        float(getattr(source, "server_infer_s", 0.0)), 4
                    ),
                }
            )
            + "\n"
        )
        self._chunks.flush()
        return True

    def close(self) -> Path:
        """Write the ticks and return the directory. Safe to call twice."""
        import pandas as pd

        if not self._chunks.closed:
            self._chunks.close()
        path = self.root / "ticks.parquet"
        frame = pd.DataFrame(self._ticks, columns=list(TICK_COLUMNS))
        frame.to_parquet(path, index=False)
        return self.root
