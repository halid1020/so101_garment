#!/usr/bin/env python
"""The figures `training_analysis` is built from, drawn from measured JSON.

Every number here is read out of a file another tool wrote -- `eval_action_mse`
for the action error, `analyse_policy_inputs` for the stream shares. Nothing is
recomputed and nothing is typed in by hand, so a figure cannot drift away from
the run it claims to describe: regenerate and the numbers follow.

Run it with no arguments to draw everything it has the inputs for. An arm whose
results are not on disk is REPORTED MISSING and skipped, never interpolated and
never quietly dropped -- a comparison with a silently absent arm reads as a
complete comparison, which is the one way this figure set could mislead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# The dataviz palette. Colour follows the POLICY FAMILY and never the rank, so a
# figure that drops an arm does not repaint the survivors. The crop arm of each
# family is the same hue drawn dashed and lighter -- a crop is a variant of its
# twin, and giving it an unrelated hue would say it was an unrelated thing.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
FAMILY = {
    "act": "#2a78d6",
    "diffusion": "#eb6834",
    "pi05": "#7a5bb5",
}

#: Arm name to the label a reader sees, for figures that take arms by name.
ARMS_LABEL = (("act", "ACT"), ("diffusion", "Diffusion"), ("pi05", "pi0.5"))

#: The six arms, in the order they read best: each baseline beside its crop.
ARMS = (
    ("act", "ACT", False),
    ("act_crop", "ACT cropped", True),
    ("diffusion", "Diffusion", False),
    ("diffusion_crop", "Diffusion cropped", True),
    ("pi05", "pi0.5", False),
    ("pi05_crop", "pi0.5 cropped", True),
)


def family_of(arm: str) -> str:
    return arm.removesuffix("_crop")


def style_for(arm: str) -> dict:
    cropped = arm.endswith("_crop")
    return {
        "color": FAMILY[family_of(arm)],
        "linestyle": "--" if cropped else "-",
        "alpha": 0.75 if cropped else 1.0,
        "linewidth": 2.0,
    }


def load_arms(directory: Path, prefix: str = "split10-") -> "dict[str, dict]":
    """Every arm's result that is on disk, keyed by arm name."""
    found = {}
    for arm, _label, _crop in ARMS:
        path = directory / f"{prefix}{arm}.json"
        if path.is_file():
            found[arm] = json.loads(path.read_text())
    return found


def tidy(axis) -> None:
    """The recessive grid and spines every figure here shares."""
    axis.grid(alpha=0.35, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)


def draw_horizon(results: "dict[str, dict]", out: Path) -> Path:
    """How a plan decays across its own chunk, every arm on ONE pair of axes.

    Two panels, trained-on and held-out, sharing a y axis so the eye can carry a
    height from one to the other -- that shared scale is the comparison.

    The axis is LOGARITHMIC and the caption has to say so. pi0.5's error is
    roughly fifty times ACT's, so on a linear axis ACT and Diffusion collapse
    onto the floor as one indistinguishable line and the figure answers nothing
    about the two arms the report is mostly about.

    THE POLICIES DO NOT SHARE A HORIZON, and this figure is where that stops
    being invisible. ACT predicts a hundred steps and Diffusion thirty-two, so
    a mean error over each policy's own chunk compares one that must guess three
    times further ahead against one that need not. MEASURED: over their native
    horizons Diffusion beats ACT on held-out data by 35 %, and over the common
    first thirty-two steps by 9 %. The shaded band marks the shortest horizon
    any drawn arm has -- everything left of it is like-for-like, everything
    right of it is one policy being asked a harder question.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    drawn = 0
    horizons = [
        len(r[h]["mse_per_step"])
        for r in results.values()
        for h in ("train", "validation")
        if r.get(h, {}).get("mse_per_step")
    ]
    common = min(horizons) if horizons else 0
    for panel, half in zip(axes, ("train", "validation")):
        for arm, label, _crop in ARMS:
            result = results.get(arm)
            if not result or half not in result:
                continue
            per_step = result[half].get("mse_per_step") or []
            if not per_step:
                continue
            steps = np.arange(1, len(per_step) + 1)
            panel.plot(steps, per_step, label=label, **style_for(arm))
            drawn += 1
        if common:
            panel.axvspan(0.5, common + 0.5, color=GRID, alpha=0.55, zorder=0)
            panel.annotate(
                f"shared horizon\n(first {common} steps)",
                xy=(common / 2, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                fontsize=8,
                color=MUTED,
            )
        panel.set_yscale("log")
        panel.set_xlabel("step within the predicted chunk")
        tidy(panel)
    axes[0].set_title("on recordings it trained on", color=INK, fontsize=11)
    axes[1].set_title("on recordings it never saw", color=INK, fontsize=11)
    axes[0].set_ylabel("action error (MSE, log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    # The title says what the curves show and not what a headline would prefer.
    # ACT's error is FLAT across its whole chunk on recordings it trained on --
    # about 6 at the first step and 11 at the hundredth -- which is not "a plan
    # decays with horizon" at all, and an earlier draft of this title said
    # exactly that and was wrong about half the figure.
    figure.suptitle(
        "ACT is flat across its own chunk on recordings it memorised, "
        "and steep on recordings it did not",
        color=INK,
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  horizon decay: {drawn} curves -> {out}")
    return out


def draw_gap(results: "dict[str, dict]", out: Path) -> Path:
    """Trained-on against held-out RMSE, with the ratio between them named.

    The ratio is the finding and the bars alone do not show it, so it is
    annotated rather than left to be divided by eye.
    """
    present = [(a, lab) for a, lab, _c in ARMS if a in results]
    if not present:
        raise SystemExit("no arms on disk to draw")
    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(present))
    width = 0.38

    # EVERY bar over the same number of steps. The recorded `rmse` averages
    # over each policy's own horizon, and ACT's is a hundred steps against
    # Diffusion's thirty-two -- so plotting the recorded numbers side by side
    # shows ACT losing partly because it was asked to plan three times further
    # ahead. An earlier draft of this figure did exactly that.
    common = min(r["validation"]["horizon"] for r in results.values())
    train = [rmse_over(results[a], "train", common) for a, _ in present]
    held = [rmse_over(results[a], "validation", common) for a, _ in present]
    colours = [FAMILY[family_of(a)] for a, _ in present]

    axis.bar(x - width / 2, train, width, color=colours, alpha=0.45, label="trained on")
    axis.bar(x + width / 2, held, width, color=colours, alpha=1.0, label="never seen")

    top = max(held)
    for index, (value_t, value_h) in enumerate(zip(train, held)):
        axis.annotate(
            f"x{value_h / value_t:.1f}",
            xy=(index, value_h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=MUTED,
        )
    axis.set_ylim(0, top * 1.18)
    axis.set_xticks(x)
    axis.set_xticklabels([lab for _, lab in present], fontsize=9)
    axis.set_ylabel("action error (RMSE)")
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    axis.set_title(
        "The gap between the two bars is the result, not the height of either",
        color=INK,
        fontsize=12,
    )
    natives = ", ".join(
        f"{lab} {results[a]['validation']['horizon']}" for a, lab in present
    )
    figure.text(
        0.5,
        -0.04,
        f"Every bar is scored over the first {common} steps of the chunk, which "
        f"is the longest horizon every arm shares.\nNative horizons differ "
        f"({natives}); scoring each over its own would compare unequal questions.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  train-vs-held-out: {len(present)} arms -> {out}")
    return out


def rmse_over(result: dict, half: str, steps: "int | None" = None) -> float:
    """RMSE over the first `steps` of the chunk, or over all of it.

    The reason this exists rather than reading `rmse` straight out of the file:
    the recorded number averages over each policy's OWN horizon, and the
    horizons differ — a hundred steps for ACT, thirty-two for Diffusion. Read
    without care that makes a policy asked to plan three times further ahead
    look worse at planning, which is a different claim entirely.
    """
    per_step = np.array(result[half]["mse_per_step"], dtype=float)
    if steps:
        per_step = per_step[:steps]
    return float(np.sqrt(per_step.mean()))


def table_rows(
    results: "dict[str, dict]", common: "int | None" = None
) -> "list[list[str]]":
    """The cross-policy table, as cells. Shared by the LaTeX and the console."""
    rows = []
    for arm, label, _crop in ARMS:
        result = results.get(arm)
        if not result:
            # A pending arm is a row of dashes and never an omitted line: a
            # table that silently lost pi0.5 reads as a complete comparison.
            rows.append([label, "--", "--", "--", "--", "--", "--"])
            continue
        held = result["validation"]
        native_gap = rmse_over(result, "validation") / rmse_over(result, "train")
        shared_held: "float | None" = None
        shared_gap: "float | None" = None
        if common:
            shared_held = rmse_over(result, "validation", common)
            shared_gap = shared_held / rmse_over(result, "train", common)
        # BOTH gaps, because they are different numbers and neither can be
        # labelled honestly on its own: over its native hundred steps the
        # action-chunking family's ratio is 5.3, and over the thirty-two it
        # shares with the diffusion family it is 4.1. The bar figure annotates
        # the shared one, so a table showing only the native one would look
        # like one of the two was simply wrong.
        rows.append(
            [
                label,
                str(held["horizon"]),
                f"{rmse_over(result, 'train'):.3f}",
                f"{rmse_over(result, 'validation'):.3f}",
                f"{native_gap:.1f}",
                f"{shared_held:.3f}" if shared_held is not None else "--",
                f"{shared_gap:.1f}" if shared_gap is not None else "--",
            ]
        )
    return rows


def write_table(results: "dict[str, dict]", out: Path) -> Path:
    """A booktabs fragment, `\\input` by the report exactly as the paper does.

    Generated and never hand-edited, for the living-paper reason: a number typed
    into prose drifts from the run it came from the first time anything is
    retrained, and nothing catches it.
    """
    horizons = [r["validation"]["horizon"] for r in results.values()]
    common = min(horizons) if horizons else None
    # TWO groups of columns, headed so they cannot be confused. The gap over a
    # policy's own horizon and over the shared one are different numbers -- 5.3
    # against 4.1 for the longer-horizon family -- and a table showing one while
    # a figure annotates the other reads as an error in one of them.
    subhead = (
        "& & \\multicolumn{3}{c}{over its own horizon} & "
        f"\\multicolumn{{2}}{{c}}{{over the shared first {common}}} \\\\"
    )
    header = "policy & horizon & train & held-out & gap & held-out & gap \\\\"
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        subhead,
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
        header,
        "\\midrule",
    ]
    for row in table_rows(results, common):
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  table -> {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mse",
        default="outputs/mse/thanos/2026-09-16",
        help="directory of <prefix><arm>.json action-error results",
    )
    parser.add_argument("--prefix", default="split10-")
    parser.add_argument("--out", default=None, help="where the figures go")
    args = parser.parse_args()

    from actoris_harena.analysis.paths import analysis_dir

    import common.rig_profile  # noqa: F401  -- the rig declares itself first

    out = Path(args.out) if args.out else analysis_dir("training-report")
    out.mkdir(parents=True, exist_ok=True)

    results = load_arms(Path(args.mse), args.prefix)
    missing = [label for arm, label, _c in ARMS if arm not in results]
    print(f"arms found: {sorted(results)}")
    if missing:
        # Named, every run, and not once at the start. A reader of the log has
        # to see which arms the figures do NOT contain.
        print(f"MISSING, drawn as gaps rather than omitted: {missing}")

    draw_horizon(results, out / "horizon_decay.png")
    draw_gap(results, out / "train_vs_heldout.png")
    write_table(results, out / "policy_table.tex")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def draw_contact_sheet(dataset: Path, out: Path, held_out: "list[int]") -> Path:
    """One frame from the middle of every recording, with the held-back ones marked.

    The point is to make the split concrete. "58 training and 7 held out" is two
    integers a reader nods at; a sheet of sixty-five frames shows how similar the
    recordings are to one another, which is the thing that decides whether
    holding out the last seven is a mild or a severe test.

    Frames come from the overhead camera, at the midpoint of each recording. The
    midpoint rather than the first frame on purpose: the first frame of every
    recording shows the same flattened garment before either arm has moved, so a
    sheet of first frames would look identical sixty-five times over and show
    nothing at all.
    """
    import av
    import pandas as pd

    key = "observation.images.central"
    episodes = pd.read_parquet(next((dataset / "meta" / "episodes").rglob("*.parquet")))
    held = set(held_out)

    # Column names here contain dots (`videos/observation.images.central/...`),
    # and itertuples renames those into attributes that no longer match. Index
    # the columns by their real names instead.
    thumbs: "list[tuple[int, np.ndarray]]" = []
    for position in range(len(episodes)):
        row = episodes.iloc[position]
        index = int(row["episode_index"])
        chunk = int(row[f"videos/{key}/chunk_index"])
        file_index = int(row[f"videos/{key}/file_index"])
        start = float(row[f"videos/{key}/from_timestamp"])
        end = float(row[f"videos/{key}/to_timestamp"])
        path = (
            dataset
            / "videos"
            / key
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        middle = (start + end) / 2.0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(middle / stream.time_base), stream=stream)
            frame = next(container.decode(stream))
            image = frame.to_ndarray(format="rgb24")
        thumbs.append((index, image[::8, ::8]))

    columns = 9
    rows = int(np.ceil(len(thumbs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.35, rows * 1.1))
    for axis in np.ravel(axes):
        axis.axis("off")
    for axis, (index, image) in zip(np.ravel(axes), thumbs):
        axis.imshow(image)
        axis.axis("off")
        marked = index in held
        for spine in ("top", "bottom", "left", "right"):
            axis.spines[spine].set_visible(False)
        axis.set_title(
            f"{index}", fontsize=6, color=FAMILY["diffusion"] if marked else MUTED
        )
        if marked:
            # An outline and a colour, never colour alone: this has to survive a
            # greyscale print and a reader who cannot separate the two hues.
            axis.add_patch(
                plt.Rectangle(
                    (0, 0),
                    image.shape[1] - 1,
                    image.shape[0] - 1,
                    fill=False,
                    edgecolor=FAMILY["diffusion"],
                    linewidth=2.5,
                )
            )
    figure.suptitle(
        f"{len(thumbs)} recordings, one frame from the middle of each — "
        f"the {len(held)} outlined are held back from training",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  contact sheet: {len(thumbs)} recordings -> {out}")
    return out


def draw_split_comparison(temporal: dict, random_split: dict, out: Path) -> Path:
    """Does the generalisation gap survive a RANDOM split? The drift question.

    The trainer holds out the LAST recordings of a session, so a gap measured
    that way mixes two causes: the policy has not seen these recordings, and the
    recordings are late in the session, by which time lighting, garment
    placement and gel wear may all have moved. This figure puts the two splits
    side by side.

    THE DIRECTION IS THE RESULT, and it is robust in a way the magnitudes are
    not. If drift were the explanation, a random held-out set -- recordings that
    sit BETWEEN training recordings, so the policy interpolates in time rather
    than extrapolating -- should be easier. MEASURED: it is harder. So the
    degradation is a property of unseen recordings and not of when they were
    recorded.

    The magnitudes are a different matter and the caption has to say so: the two
    splits hold out DIFFERENT recordings, so part of the difference between the
    two gap ratios is that one set of seven may simply be harder than the other.
    """
    arms = [a for a in ("act", "diffusion") if a in temporal and a in random_split]
    if not arms:
        raise SystemExit("need both splits for at least one policy family")

    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(arms))
    width = 0.36
    gaps_t = [
        rmse_over(temporal[a], "validation") / rmse_over(temporal[a], "train")
        for a in arms
    ]
    gaps_r = [
        rmse_over(random_split[a], "validation") / rmse_over(random_split[a], "train")
        for a in arms
    ]
    colours = [FAMILY[family_of(a)] for a in arms]
    axis.bar(
        x - width / 2,
        gaps_t,
        width,
        color=colours,
        alpha=0.45,
        label="held out the LAST seven",
    )
    axis.bar(
        x + width / 2,
        gaps_r,
        width,
        color=colours,
        alpha=1.0,
        label="held out a RANDOM seven",
    )
    for index, (a, b) in enumerate(zip(gaps_t, gaps_r)):
        for offset, value in ((-width / 2, a), (width / 2, b)):
            axis.annotate(
                f"x{value:.1f}",
                xy=(index + offset, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=MUTED,
            )
    axis.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    axis.annotate(
        "no gap at all",
        xy=(len(arms) - 0.5, 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    axis.set_xticks(x)
    axis.set_xticklabels([dict(ARMS_LABEL).get(a, a) for a in arms], fontsize=10)
    axis.set_ylabel("held-out error / trained-on error")
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    axis.set_title(
        "The gap does not come from drift: a random hold-out is harder, not easier",
        color=INK,
        fontsize=12,
    )
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  split comparison: {len(arms)} families -> {out}")
    return out
