#!/usr/bin/env python3
"""Turn an attribution run into things you can put on a slide.

Reads the `attribution.json` written by `tool/analyse_policy_inputs.py` and
produces a folder of videos, plots and tables about what each input stream
contributed. It loads no checkpoint: the numbers are already decided, and this
only presents them -- so it can be re-run to restyle a figure without a GPU.

    venv/bin/python tool/analysis_slides.py --in outputs/analysis/<day>/act-all

Videos need the frames as well as the numbers, so pass whichever source the
analysis used:

    venv/bin/python tool/analysis_slides.py --in outputs/analysis/<day>/act-all \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile
    venv/bin/python tool/analysis_slides.py --in outputs/analysis/<day>/rollout \\
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
            f"What the {payload['policy']} policy's plan depended on — episode "
            f"{episode_name}, frame {index}" + (f"  ·  {phase}" if phase else ""),
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


def compare(dirs: "list[str]", out_dir: Path) -> None:
    """One page putting several decks side by side, caveats and all.

    Separate from the per-deck slides because it answers a different question --
    *do these policies use the rig the same way?* -- and because the honest
    answer needs the differences between the decks printed next to the numbers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = [load(Path(d).expanduser().resolve()) for d in dirs]
    headers, rows = slides.compare_table(payloads)
    caveats = slides.compare_caveats(payloads)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "compare.csv").write_text(slides.csv_table(headers, rows))

    names = [r[0] for r in rows]
    fig, axis = plt.subplots(figsize=(1.6 * len(names) + 4, 4.4))
    width = 0.8 / max(len(payloads), 1)
    for column, payload in enumerate(payloads):
        shares = slides.pooled_shares(payload)
        heights = [
            shares.get(n, 0.0) if n in payload["streams"] else 0.0 for n in names
        ]
        axis.bar(
            [i + column * width for i in range(len(names))],
            heights,
            width=width,
            label=f"{payload['policy']} ({len(payload['cameras'])} cam)",
        )
    axis.set_xticks([i + 0.4 - width / 2 for i in range(len(names))])
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_ylabel("mean share of the plan (within its own deck)")
    axis.set_title("What each policy's plan depended on")
    axis.legend()
    axis.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "compare.png", dpi=150)
    plt.close(fig)

    body = "\n".join(f"* {note}" for note in caveats)
    (out_dir / "COMPARE.md").write_text(
        "# Do these policies use the rig the same way?\n\n"
        + slides.markdown_table(headers, rows)
        + "\n\n![](compare.png)\n\n## Read this before quoting a number\n\n"
        + body
        + "\n"
    )
    print(slides.markdown_table(headers, rows))
    print()
    for note in caveats:
        print(f"⚠️  {note}")
    print(f"📝 {out_dir / 'COMPARE.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in",
        dest="source_dir",
        default=None,
        help="A directory written by tool/analyse_policy_inputs.py",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Two or more such directories, put side by side on one page "
        "instead of building a deck for one",
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

    if args.compare:
        if len(args.compare) < 2:
            raise SystemExit("❌ --compare needs at least two analysis directories")
        default = Path(args.compare[0]).expanduser().resolve().parent / "compare"
        compare(args.compare, Path(args.out).expanduser() if args.out else default)
        return
    if not args.source_dir:
        raise SystemExit("❌ pass --in <analysis dir>, or --compare <dir> <dir> ...")

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
            title=(
                f"Vision and touch over one episode "
                f"({payload['policy']}, {len(cameras)} cameras)"
            ),
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
    (out_dir / "SLIDES.md").write_text(
        running_order(payload, numbers, out_dir, tables, method_rows)
    )
    print()
    for path in written:
        if path:
            print(f"🖼️  {path}")
    print(f"📝 {out_dir / 'SLIDES.md'}")


def gradcam_section(payload) -> str:
    """Section 5, which exists only if there is a Grad-CAM to show.

    A token model (pi0.5, the flow-matching policies) has no convolutional
    feature map to weight, and `analyse_policy_inputs.py` records that in the
    payload rather than raising. Printing the heading anyway would leave a deck
    promising a figure that was never drawn.
    """
    drawn = any(
        "gradcam" in frame
        for episode in payload["episodes"].values()
        for frame in episode["frames"]
    )
    if drawn:
        return (
            "## 5. Where in the frame — `gradcam.mp4`\n\n"
            "Grad-CAM on each camera's own trunk, moving through the episode.\n"
        )
    why = next(
        (
            frame["gradcam_unavailable"]
            for episode in payload["episodes"].values()
            for frame in episode["frames"]
            if "gradcam_unavailable" in frame
        ),
        "it was not among the methods this run was asked for",
    )
    return (
        "## 5. Where in the frame — not available for this policy\n\n"
        f"No Grad-CAM: {why}\n"
    )


def agreement_section(method_rows) -> str:
    """Section 6, stating what these methods did, not what ACT's did once.

    The template used to assert that integrated gradients agrees and attention
    does not. That was MEASURED on one ACT checkpoint; on a policy with no
    attention to report, or a checkpoint where the numbers come out otherwise,
    printing it would put a finding on a slide that nobody made.
    """
    lines = []
    reference_spread = next(
        (row[-2] for row in method_rows if row[0] == "occlusion"), "—"
    )
    for row in method_rows:
        method, spread, agreement = row[0], row[-2], row[-1]
        if method == "occlusion" or agreement == "—":
            continue
        detail = f", spread **{spread}**" if spread != "—" else ""
        lines.append(f"* {method}: rank agreement **{agreement}**{detail}")
    if not lines:
        return (
            "## 6. Do the methods agree? — `table_methods.csv`\n\n"
            "Only occlusion ran on this policy, so there is nothing to compare it "
            "against. That is a property of the architecture, not a gap in the "
            "run: see the note beside the table.\n"
        )
    body = "\n".join(lines)
    # Only say this where there IS an attention row. On a policy that has no
    # decoder cross-attention to report -- diffusion, pi0.5 -- the paragraph
    # would be describing a method the deck never ran, which is the habit this
    # whole file was rewritten to break.
    attention_note = (
        "\nThat is exactly what raw attention does here, and why the *vs "
        "uniform* row is the one to quote: each camera owns the same number of "
        "tokens, so raw attention mass says more about token count than about "
        "the policy.\n"
        if any(row[0].startswith("attention") for row in method_rows)
        else ""
    )
    return (
        "## 6. Do the methods agree? — `table_methods.csv`\n\n"
        "Spearman rank agreement against occlusion, which is the behavioural "
        "ground truth here, and the spread between the largest and smallest "
        "camera share each method gives (occlusion's own spread is "
        f"**{reference_spread}**):\n\n"
        f"{body}\n\n"
        "**Read the two columns together.** A near-uniform method can still sort "
        "the cameras the same way and so score a high agreement while carrying "
        "almost no signal.\n" + attention_note
    )


def running_order(payload, numbers, out_dir: Path, tables: str, method_rows) -> str:
    """A deck's worth of order and wording, so the numbers arrive with a claim."""
    tactile = ", ".join(numbers["tactile"]) or "none in this camera set"
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

Fingertips ({len(numbers['tactile'])} of {len(payload['cameras'])} cameras): {tactile}

## 4. By phase — `by_phase.png`, `table_by_phase.csv`

Phases come from the recorded gripper channels, with thresholds drawn from each
episode's own range rather than fixed — these grippers work in a narrow band
whose position depends on the object and the day.

{gradcam_section(payload)}
{agreement_section(method_rows)}
## 7. What it does not say

This is what the trained policy uses, not what a policy could learn without a
stream. The camera-ablation training rows answer that, and the two belong on
the same slide when both are in.

---

{tables}
"""


if __name__ == "__main__":
    main()
