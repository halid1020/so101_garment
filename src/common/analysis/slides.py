"""Slide-ready output: videos, plots and tables from an attribution run.

A study is not a talk. What persuades on a slide is a picture that moves in step
with the thing it explains, and a table small enough to read from the back of a
room -- neither of which falls out of ``attribution.json`` unaided.

Everything here reads a finished result. Nothing infers, nothing loads a
checkpoint, so a figure or a video can be redrawn from the numbers without a GPU
and without the dataset.

Videos are encoded with **PyAV**, not the ffmpeg command line: a compute node
has the library and not the binary, which is the same reason
``recording/dataset_view.py`` encodes its composites that way. H.264 rather than
the AV1 the datasets use -- a slide deck is opened by whatever is on the laptop
in the room, and PowerPoint and Keynote both decode H.264 and neither reliably
decodes AV1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Slides are read from a distance, so nothing here is drawn at a size that
#: works on a monitor and vanishes on a projector.
SLIDE_DPI = 150
SLIDE_FONT = 13


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": SLIDE_FONT,
            "axes.titlesize": SLIDE_FONT + 2,
            "axes.labelsize": SLIDE_FONT,
            "legend.fontsize": SLIDE_FONT - 2,
            "figure.facecolor": "white",
        }
    )
    return plt


# ── tables ───────────────────────────────────────────────────────────────────


def markdown_table(headers: "list[str]", rows: "list[list[str]]") -> str:
    widths = [
        max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))
    ]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([line, rule, *body])


def csv_table(headers: "list[str]", rows: "list[list[str]]") -> str:
    return "\n".join([",".join(headers), *[",".join(str(c) for c in r) for r in rows]])


def stream_series(episode: dict, stream: str, key: str = "share") -> "list[float]":
    """One stream's occlusion metric across an episode's analysed frames."""
    return [
        float(frame["occlusion"]["streams"][stream][key])
        for frame in episode["frames"]
        if "occlusion" in frame
    ]


def frame_phases(episode: dict) -> "list[str]":
    """The phase each ANALYSED frame stands for -- the dominant one in its window.

    Not the label of that exact frame. The analysis samples one frame in N, and
    the transitions are SHORT: a gripper closes in well under fifteen frames, so
    reading the phase at the sampled index alone misses `closing` and `opening`
    almost entirely and reports an episode as nothing but reaching and holding.
    Each sampled frame stands for the stretch up to the next one, so the phase it
    is labelled with is the one covering most of that stretch.
    """
    labels = episode.get("phases") or []
    frames = episode.get("frames") or []
    if not labels or not frames:
        return []
    indices = [int(f["index"]) for f in frames]
    step = (indices[1] - indices[0]) if len(indices) > 1 else 1
    out: "list[str]" = []
    for position, index in enumerate(indices):
        stop = indices[position + 1] if position + 1 < len(indices) else index + step
        window = labels[min(index, len(labels) - 1) : min(stop, len(labels))]
        if not window:
            window = [labels[min(index, len(labels) - 1)]]
        out.append(max(set(window), key=window.count))
    return out


def by_phase_table(
    payload: dict, streams: "list[str]"
) -> "tuple[list[str], list[list[str]]]":
    """Mean share per stream per gripper phase, pooled over every episode.

    The table the question was asked to produce: it is where "tactile matters at
    contact and not during the approach" is either true or not.
    """
    from common.analysis.phases import PHASES

    buckets: "dict[str, dict[str, list[float]]]" = {s: {} for s in streams}
    for episode in payload["episodes"].values():
        labels = frame_phases(episode)
        if not labels:
            continue
        for stream in streams:
            values = stream_series(episode, stream)
            for label, value in zip(labels, values):
                buckets[stream].setdefault(label, []).append(value)

    present = [p for p in PHASES if any(p in buckets[s] for s in streams)]
    headers = ["stream", *present, "overall"]
    rows = []
    for stream in streams:
        cells = []
        for phase in present:
            picked = buckets[stream].get(phase) or []
            cells.append(f"{np.mean(picked):.1%}" if picked else "—")
        every = [v for values in buckets[stream].values() for v in values]
        cells.append(f"{np.mean(every):.1%}" if every else "—")
        rows.append([stream, *cells])
    return headers, rows


def method_table(payload: dict) -> "tuple[list[str], list[list[str]]]":
    """Every method's mean share per camera, side by side, with its agreement.

    Presented together on purpose. Two of these three are inferences about the
    policy and one is an intervention on it, and a reader who is shown only the
    agreeing pair learns the wrong lesson about the third.
    """
    from common.analysis.gradients import rank_agreement

    cameras = payload["cameras"]
    pooled: "dict[str, dict[str, float]]" = {}
    for method, path in (
        ("occlusion", ("occlusion", "streams", "share")),
        ("integrated gradients", ("ig", "shares")),
        ("attention (raw)", ("attention", "mean")),
        ("attention (vs uniform)", ("attention", "deviation")),
    ):
        values: "dict[str, list[float]]" = {c: [] for c in cameras}
        for episode in payload["episodes"].values():
            for frame in episode["frames"]:
                if path[0] not in frame:
                    continue
                block = frame[path[0]]
                for camera in cameras:
                    if path[0] == "occlusion":
                        if camera in block["streams"]:
                            values[camera].append(block["streams"][camera]["share"])
                    elif camera in block[path[1]]:
                        values[camera].append(block[path[1]][camera])
        if any(values[c] for c in cameras):
            pooled[method] = {
                c: float(np.mean(values[c])) if values[c] else 0.0 for c in cameras
            }

    reference = pooled.get("occlusion", {})
    headers = ["method", *cameras, "agreement with occlusion"]
    rows = []
    for method, shares in pooled.items():
        agreement = (
            "—"
            if method == "occlusion"
            else f"{rank_agreement(shares, reference):+.2f}"
        )
        rows.append([method, *[f"{shares[c]:.1%}" for c in cameras], agreement])
    return headers, rows


def headline_numbers(payload: dict, tactile_prefix: str = "") -> "dict[str, Any]":
    """The three or four numbers a slide actually carries."""
    cameras = payload["cameras"]
    tactile = [c for c in cameras if "gripper" in c]
    vision = [c for c in cameras if c not in tactile]
    per_frame_tactile: "list[float]" = []
    for episode in payload["episodes"].values():
        series = [stream_series(episode, c) for c in tactile]
        if series and series[0]:
            per_frame_tactile.extend(np.sum(series, axis=0).tolist())
    return {
        "cameras": cameras,
        "tactile": tactile,
        "vision": vision,
        "tactile_mean": float(np.mean(per_frame_tactile)) if per_frame_tactile else 0.0,
        "tactile_peak": float(np.max(per_frame_tactile)) if per_frame_tactile else 0.0,
        "tactile_min": float(np.min(per_frame_tactile)) if per_frame_tactile else 0.0,
        "frames": len(per_frame_tactile),
        "episodes": len(payload["episodes"]),
    }


# ── plots ────────────────────────────────────────────────────────────────────


def phase_bars(path, headers, rows, title="") -> str:
    """Per-stream share grouped by gripper phase. The talk's summary figure."""
    plt = _plt()
    phases = headers[1:-1]
    streams = [r[0] for r in rows]
    values = np.array(
        [
            [float(c.rstrip("%")) / 100 if c != "—" else 0.0 for c in r[1:-1]]
            for r in rows
        ]
    )
    fig, axis = plt.subplots(figsize=(11, 5.2))
    width = 0.8 / max(len(streams), 1)
    from common.analysis.report import colours

    ink = colours(streams)
    for index, stream in enumerate(streams):
        offset = (index - (len(streams) - 1) / 2) * width
        axis.bar(
            [i + offset for i in range(len(phases))],
            values[index],
            width=width,
            label=stream,
            color=ink[stream],
        )
    axis.set_xticks(range(len(phases)))
    axis.set_xticklabels(phases)
    axis.set_ylabel("share of the plan's movement")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, lw=0.6)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=SLIDE_DPI)
    plt.close(fig)
    return str(path)


def vision_vs_touch(path, payload, title="") -> str:
    """One line for vision, one for touch, over the episode. The clearest slide.

    Five curves is a study; two is a talk. The fingertips are summed because the
    question on a slide is whether touch matters at all and when, not which of
    four fingers mattered most.
    """
    plt = _plt()
    numbers = headline_numbers(payload)
    fig, axis = plt.subplots(figsize=(11, 5.0))
    from common.analysis.phases import spans_of
    from common.analysis.report import PHASE_INK

    for name, data in payload["episodes"].items():
        frames = [f["index"] for f in data["frames"]]
        touch = np.sum([stream_series(data, c) for c in numbers["tactile"]], axis=0)
        sight = np.sum([stream_series(data, c) for c in numbers["vision"]], axis=0)
        labels = frame_phases(data)
        if labels:
            seen = set()
            for phase, start, stop in spans_of(labels):
                ink = PHASE_INK.get(phase, "#00000000")
                if ink.endswith("00"):
                    continue
                axis.axvspan(
                    frames[start],
                    frames[min(stop, len(frames)) - 1],
                    color=ink,
                    lw=0,
                    label=phase if phase not in seen else None,
                )
                seen.add(phase)
        axis.plot(frames, sight, color="#1f77b4", lw=2.4, label="vision (central)")
        axis.plot(frames, touch, color="#d62728", lw=2.4, label="touch (4 fingertips)")
        break  # one episode: a slide shows an example, a table shows the pooling
    axis.set_xlabel("frame")
    axis.set_ylabel("share of the plan's movement")
    axis.set_title(title)
    axis.grid(alpha=0.25, lw=0.6)
    axis.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=SLIDE_DPI)
    plt.close(fig)
    return str(path)


# ── video ────────────────────────────────────────────────────────────────────


def encode(path, frames, fps: float = 6.0) -> str:
    """A list of RGB uint8 arrays -> one H.264 file. PyAV, no ffmpeg binary.

    Every frame must be the same size, and both dimensions must be even -- H.264
    encodes in 16x16 macroblocks over a 4:2:0 chroma plane, and an odd dimension
    is rejected by the encoder rather than padded.
    """
    import av

    if not frames:
        return ""
    height, width = frames[0].shape[:2]
    height, width = height - (height % 2), width - (width % 2)
    out: "Any" = av.open(str(path), mode="w")
    try:
        stream = out.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "slow"}
        for frame in frames:
            canvas = np.ascontiguousarray(frame[:height, :width, :3])
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, "rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return str(path)


def figure_to_array(fig) -> np.ndarray:
    """A matplotlib figure as RGB pixels, without going through a file."""
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return buffer[:, :, :3].copy()
