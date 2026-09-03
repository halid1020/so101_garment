#!/usr/bin/env python3
"""Turn an attribution run into things you can put on a slide.

Reads the `attribution.json` written by `tool/analyse_policy_inputs.py` and
produces a folder of videos, plots and tables about what each input stream
contributed. It loads no checkpoint: the numbers are already decided, and this
only presents them -- so it can be re-run to restyle a figure without a GPU.

    venv/bin/python tool/analysis_slides.py --in outputs/analysis/act-all

Videos need the frames as well as the numbers, so pass whichever source the
analysis used:

    venv/bin/python tool/analysis_slides.py --in outputs/analysis/act-all \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile
    venv/bin/python tool/analysis_slides.py --in outputs/analysis/rollout \\
        --run outputs/policy_runs/<stamp>

What it writes:

  contribution.mp4   the episode playing beside the curve that tracks it, so
                     a stream's rise can be watched against what the hands are
                     doing at that moment
  gradcam.mp4        the five cameras with their Grad-CAM overlays, moving
  vision_vs_touch.png / by_phase.png / methods.png
  tables.md          the same numbers as Markdown, and one .csv each
  SLIDES.md          the running order, with the headline sentences already
                     written out

Encoded H.264 with PyAV -- a compute node has the library and not the ffmpeg
binary, and a projector laptop decodes H.264 and may not decode AV1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.analysis import slides  # noqa: E402
from common.analysis.phases import spans_of  # noqa: E402
from common.analysis.report import PHASE_INK  # noqa: E402


def load(path: Path) -> dict:
    payload = json.loads((path / "attribution.json").read_text())
    if not payload.get("episodes"):
        raise SystemExit(f"❌ {path} has no analysed episodes in it")
    return payload


def source_for(args, payload):
    """The frames the video needs, or None if this run is numbers only."""
    if args.dataset:
        from common.analysis.sources import DatasetSource

        first = next(iter(payload["episodes"]))
        indices = [f["index"] for f in payload["episodes"][first]["frames"]]
        every = max(1, (indices[1] - indices[0]) if len(indices) > 1 else 1)
        return DatasetSource(args.dataset, first, every, payload["cameras"])
    if args.run:
        from common.analysis.sources import RunSource

        return RunSource(args.run, payload["cameras"])
    return None


def frames_of(source, episode: int, wanted: "list[int]") -> "dict[int, dict]":
    """The observations the analysis actually used, by index."""
    keep = set(wanted)
    out: "dict[int, dict]" = {}
    for index, _state, images, _truth in source.observations(episode):
        if index in keep:
            out[int(index)] = images
        if len(out) == len(keep):
            break
    return out


def contribution_video(out_path, payload, source, episode_name, fps) -> str:
    """The scene playing above the curve that explains it, frame by frame.

    One picture that moves in step with one curve. A still of either alone
    invites the question the other answers -- 'when was that?' and 'what was
    happening?' -- and answering it live is what makes the point land.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = payload["episodes"][episode_name]
    indices = [f["index"] for f in episode["frames"]]
    if not indices:
        return ""
    numbers = slides.headline_numbers(payload)
    touch = np.sum(
        [slides.stream_series(episode, c) for c in numbers["tactile"]], axis=0
    )
    sight = np.sum(
        [slides.stream_series(episode, c) for c in numbers["vision"]], axis=0
    )
    labels = slides.frame_phases(episode)
    pictures = frames_of(source, int(episode_name), indices)
    if not pictures:
        return ""

    rendered = []
    for step, index in enumerate(indices):
        if index not in pictures:
            continue
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        grid = fig.add_gridspec(2, len(payload["cameras"]), height_ratios=[1.35, 1])
        for column, camera in enumerate(payload["cameras"]):
            axis = fig.add_subplot(grid[0, column])
            axis.imshow(pictures[index][camera])
            axis.set_title(camera.replace("_", " "), fontsize=10)
            axis.axis("off")
        axis = fig.add_subplot(grid[1, :])
        if labels:
            seen = set()
            for phase, start, stop in spans_of(labels):
                ink = PHASE_INK.get(phase, "#00000000")
                if ink.endswith("00"):
                    continue
                axis.axvspan(
                    indices[start],
                    indices[min(stop, len(indices)) - 1],
                    color=ink,
                    lw=0,
                    label=phase if phase not in seen else None,
                )
                seen.add(phase)
        axis.plot(indices, sight, color="#1f77b4", lw=2.2, label="vision (central)")
        axis.plot(indices, touch, color="#d62728", lw=2.2, label="touch (4 fingertips)")
        axis.axvline(index, color="#111", lw=1.6)
        axis.plot([index], [touch[step]], "o", color="#d62728", ms=9)
        axis.plot([index], [sight[step]], "o", color="#1f77b4", ms=9)
        axis.set_xlim(min(indices), max(indices))
        axis.set_ylim(0, max(1e-3, float(max(sight.max(), touch.max())) * 1.15))
        axis.set_xlabel("frame")
        axis.set_ylabel("share of the plan")
        axis.grid(alpha=0.25, lw=0.6)
        axis.legend(loc="upper right", fontsize=10, framealpha=0.92)
        phase = labels[step] if labels else ""
        fig.suptitle(
            f"What the ACT policy's plan depended on — episode {episode_name}, "
            f"frame {index}" + (f"  ·  {phase}" if phase else ""),
            fontsize=14,
        )
        fig.tight_layout()
        rendered.append(slides.figure_to_array(fig))
        plt.close(fig)
    return slides.encode(out_path, rendered, fps)


def gradcam_video(out_path, payload, source, episode_name, fps) -> str:
    """The five cameras with their Grad-CAM overlays, moving through the episode."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = payload["episodes"][episode_name]
    have = [f for f in episode["frames"] if "gradcam" in f]
    if not have:
        return ""
    indices = [f["index"] for f in have]
    pictures = frames_of(source, int(episode_name), indices)
    rendered = []
    for record in have:
        index = record["index"]
        if index not in pictures:
            continue
        fig, axes = plt.subplots(
            1, len(payload["cameras"]), figsize=(12.8, 3.6), dpi=100
        )
        axes = np.atleast_1d(axes)
        for axis, camera in zip(axes, payload["cameras"]):
            frame = pictures[index][camera]
            axis.imshow(frame)
            heat = np.asarray(record["gradcam"].get(camera, []), dtype=float)
            if heat.size:
                axis.imshow(
                    cv2.resize(heat, (frame.shape[1], frame.shape[0])),
                    cmap="jet",
                    alpha=0.45,
                )
            axis.set_title(camera.replace("_", " "), fontsize=9)
            axis.axis("off")
        fig.suptitle(f"Grad-CAM — where the plan came from, frame {index}", fontsize=13)
        fig.tight_layout()
        rendered.append(slides.figure_to_array(fig))
        plt.close(fig)
    return slides.encode(out_path, rendered, fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in",
        dest="source_dir",
        required=True,
        help="A directory written by tool/analyse_policy_inputs.py",
    )
    parser.add_argument("--dataset", help="The dataset it analysed (for the videos)")
    parser.add_argument("--run", help="The rollout it analysed (for the videos)")
    parser.add_argument(
        "--out", default=None, help="Where to write (default: <in>/slides)"
    )
    parser.add_argument("--fps", type=float, default=6.0, help="Video frame rate")
    parser.add_argument("--episode", default=None, help="Which episode the videos show")
    parser.add_argument("--no-video", action="store_true", help="Plots and tables only")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    payload = load(source_dir)
    out_dir = Path(args.out).expanduser() if args.out else source_dir / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = payload["cameras"]
    streams = payload["streams"]
    numbers = slides.headline_numbers(payload)
    print(
        f"📊 {payload['policy']} · {len(cameras)} camera(s) · "
        f"{numbers['episodes']} episode(s), {numbers['frames']} analysed frame(s)"
    )

    # -- tables ---------------------------------------------------------
    phase_headers, phase_rows = slides.by_phase_table(payload, streams)
    method_headers, method_rows = slides.method_table(payload)
    (out_dir / "table_by_phase.csv").write_text(
        slides.csv_table(phase_headers, phase_rows)
    )
    (out_dir / "table_methods.csv").write_text(
        slides.csv_table(method_headers, method_rows)
    )
    tables = (
        "# Tables\n\n## What each stream contributed, by gripper phase\n\n"
        + slides.markdown_table(phase_headers, phase_rows)
        + "\n\nShare of how far the plan moves when that stream is replaced "
        f"(baseline `{payload['baseline']}`, {payload['direction']}). "
        "Phases are segmented from the recorded gripper channels.\n\n"
        "## Do the methods agree?\n\n"
        + slides.markdown_table(method_headers, method_rows)
        + "\n\nAgreement is the Spearman correlation of the per-camera ranking "
        "against occlusion, which is the only one of these that intervenes on "
        "the policy rather than inferring about it.\n"
    )
    (out_dir / "tables.md").write_text(tables)
    print(f"📋 {out_dir / 'tables.md'}")
    print()
    print(slides.markdown_table(phase_headers, phase_rows))
    print()
    print(slides.markdown_table(method_headers, method_rows))

    # -- plots ----------------------------------------------------------
    written = [
        slides.vision_vs_touch(
            out_dir / "vision_vs_touch.png",
            payload,
            title="Vision and touch over one episode (ACT, five cameras)",
        ),
        slides.phase_bars(
            out_dir / "by_phase.png",
            phase_headers,
            phase_rows,
            title="What each input stream contributed, by gripper phase",
        ),
    ]

    # -- videos ---------------------------------------------------------
    episode_name = args.episode or next(iter(payload["episodes"]))
    if not args.no_video:
        source = source_for(args, payload)
        if source is None:
            print(
                "\nℹ️  no --dataset or --run given, so no video: the frames are "
                "not in attribution.json (they are the largest part of an "
                "observation and are deliberately not stored there)"
            )
        else:
            print(f"\n🎬 rendering from {source.describe()}")
            written.append(
                contribution_video(
                    out_dir / "contribution.mp4",
                    payload,
                    source,
                    episode_name,
                    args.fps,
                )
            )
            written.append(
                gradcam_video(
                    out_dir / "gradcam.mp4", payload, source, episode_name, args.fps
                )
            )

    # -- the running order ----------------------------------------------
    (out_dir / "SLIDES.md").write_text(running_order(payload, numbers, out_dir, tables))
    print()
    for path in written:
        if path:
            print(f"🖼️  {path}")
    print(f"📝 {out_dir / 'SLIDES.md'}")


def running_order(payload, numbers, out_dir: Path, tables: str) -> str:
    """A deck's worth of order and wording, so the numbers arrive with a claim."""
    tactile = ", ".join(numbers["tactile"])
    return f"""# What each input stream contributes — running order

Policy: **{payload['policy']}** on {len(payload['cameras'])} cameras.
Source: {payload['source']}
Measured over {numbers['episodes']} episode(s), {numbers['frames']} frames.
Streams removed against the `{payload['baseline']}` baseline, `{payload['direction']}`.

---

## 1. The question

Two questions, and they are not the same one.

* What *could* a policy learn from each stream? — train one per camera set and
  compare. Costs GPU-days.
* What does *this* trained policy *use*? — measured from the weights, in minutes.

They can disagree, and the disagreement is the result.

## 2. How it is measured

Replace one stream, re-plan, and see how far the plan moved. That is behaviour,
not a claim about behaviour — so it is the ground truth everything else is
scored against. There is no neutral image, so the substitute is named on every
figure.

## 3. Vision and touch over an episode — `vision_vs_touch.png`, `contribution.mp4`

The tactile cameras contribute **{numbers['tactile_min']:.0%}** at their
quietest and **{numbers['tactile_peak']:.0%}** at their peak, averaging
**{numbers['tactile_mean']:.0%}**. The video is the same curve with the scene
above it, so the rise can be watched against what the hands are doing.

Fingertips: {tactile}

## 4. By phase — `by_phase.png`, `table_by_phase.csv`

Phases come from the recorded gripper channels, with thresholds drawn from each
episode's own range rather than fixed — these grippers work in a narrow band
whose position depends on the object and the day.

## 5. Where in the frame — `gradcam.mp4`

Grad-CAM on each camera's own trunk, moving through the episode.

## 6. Do the methods agree? — `table_methods.csv`

Integrated gradients and occlusion agree. **Attention does not**, and that is
worth a slide of its own: each camera owns the same number of tokens, so its
attention mass is nearly uniform by construction and ranks the cameras against
measured behaviour almost in reverse. Reporting attention as importance would
have inverted the finding.

## 7. What it does not say

This is what the trained policy uses, not what a policy could learn without a
stream. The camera-ablation training rows answer that, and the two belong on
the same slide when both are in.

---

{tables}
"""


if __name__ == "__main__":
    main()
