"""When in an episode a stream mattered, against what the hands were doing.

An attribution averaged over an episode answers almost nothing. Tactile cameras
cannot contribute while the grippers are still travelling towards the cloth, and
that they contribute little THERE says nothing about whether they matter at the
moment of contact. So the interesting figure is contribution against time, with
the episode segmented by what the grippers were doing -- and the segmentation
has to come from the data rather than from a stopwatch.

The gripper channels are the segmentation. They are columns 5 and 11 of the 12-D
action, from the recorder's own layout ``[left5 deg, left_grip, right5 deg,
right_grip]``, and across these datasets they span a few tenths of open fraction
and never reach either end -- so a phase boundary is a change in that narrow
band, never an absolute threshold, which would put every frame in one phase.

Pure: numbers in, labels out. No policy, no torch.
"""

from __future__ import annotations

import numpy as np

GRIPPER_COLUMNS = (5, 11)

#: The phases, in the order they occur. ``closing``/``opening`` are transitions
#: and are usually short; ``holding`` is a closed gripper that is not changing,
#: which is where a tactile camera should earn its place.
PHASES = ("reaching", "closing", "holding", "opening", "released")


def gripper_trace(actions: np.ndarray, columns=GRIPPER_COLUMNS) -> np.ndarray:
    """The gripper channels over an episode: ``(frames, 2)``."""
    actions = np.asarray(actions, dtype=np.float64)
    valid = [c for c in columns if c < actions.shape[1]]
    return actions[:, valid]


def segment(
    actions: np.ndarray,
    columns=GRIPPER_COLUMNS,
    closed_quantile: float = 0.4,
    move_quantile: float = 0.75,
) -> "list[str]":
    """One phase label per frame, from the grippers alone.

    Both thresholds are QUANTILES of this episode's own values, not constants.
    An absolute "closed below 0.05" would label an entire dataset one phase,
    because these grippers work in a band whose position depends on the object
    and on the calibration of the day.

    A frame is *closing* or *opening* when the gripper is moving faster than
    ``move_quantile`` of this episode's speeds, and the sign says which. It is
    *holding* when closed and not moving, *reaching* when open before the first
    close, and *released* when open after the last one.
    """
    trace = gripper_trace(actions, columns)
    if trace.size == 0:
        return []
    frames = trace.shape[0]
    if frames < 3:
        return ["reaching"] * frames

    # The tighter of the two hands drives the phase: a one-armed fold still has
    # a contact moment, and averaging would smear it into nothing.
    tightest = trace.min(axis=1)
    speed = np.gradient(tightest)
    closed_at = np.quantile(tightest, closed_quantile)
    moving_at = np.quantile(np.abs(speed), move_quantile)
    # A perfectly still trace makes every threshold zero and every frame
    # "moving"; require a real change before calling anything a transition.
    if moving_at <= 0 or np.allclose(np.abs(speed).max(), 0.0):
        return ["reaching"] * frames

    closed = tightest <= closed_at
    labels: "list[str]" = []
    for index in range(frames):
        if abs(speed[index]) >= moving_at:
            labels.append("closing" if speed[index] < 0 else "opening")
        elif closed[index]:
            labels.append("holding")
        else:
            labels.append("reaching")

    # Everything open AFTER the last closed frame is "released", not another
    # approach: the difference matters, because it is the half of the episode a
    # tactile camera has nothing left to feel.
    closed_indices = np.flatnonzero(closed)
    if closed_indices.size:
        for index in range(int(closed_indices[-1]) + 1, frames):
            if labels[index] == "reaching":
                labels[index] = "released"
    return labels


def spans_of(labels: "list[str]") -> "list[tuple[str, int, int]]":
    """Contiguous runs as ``(phase, start, stop)``, half-open. For shading a plot."""
    out: "list[tuple[str, int, int]]" = []
    for index, label in enumerate(labels):
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1], index + 1)
        else:
            out.append((label, index, index + 1))
    return out


def by_phase(labels: "list[str]", values: "list[float]") -> "dict[str, float]":
    """Mean of ``values`` within each phase. The table under the time plot."""
    out: "dict[str, float]" = {}
    for phase in PHASES:
        picked = [v for label, v in zip(labels, values) if label == phase]
        if picked:
            out[phase] = float(np.mean(picked))
    return out
