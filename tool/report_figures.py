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
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    drawn = 0
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
    figure.suptitle(
        "A plan is most wrong at its far end, and the gap widens on unseen data",
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
    figure, axis = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(present))
    width = 0.38

    train = [results[a]["train"]["rmse"] for a, _ in present]
    held = [results[a]["validation"]["rmse"] for a, _ in present]
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
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  train-vs-held-out: {len(present)} arms -> {out}")
    return out


def table_rows(results: "dict[str, dict]") -> "list[list[str]]":
    """The cross-policy table, as cells. Shared by the LaTeX and the console."""
    rows = []
    for arm, label, _crop in ARMS:
        result = results.get(arm)
        if not result:
            rows.append([label, "--", "--", "--", "--", "--", "--"])
            continue
        train, held = result["train"], result["validation"]
        rows.append(
            [
                label,
                f"{train['rmse']:.3f}",
                f"{held['rmse']:.3f}",
                f"{held['rmse'] / train['rmse']:.1f}",
                f"{held['first_step_mse']:.1f}",
                f"{held['last_step_mse']:.1f}",
                f"{held['mse_grippers']:.5f}",
            ]
        )
    return rows


def write_table(results: "dict[str, dict]", out: Path) -> Path:
    """A booktabs fragment, `\\input` by the report exactly as the paper does.

    Generated and never hand-edited, for the living-paper reason: a number typed
    into prose drifts from the run it came from the first time anything is
    retrained, and nothing catches it.
    """
    header = (
        "policy & train RMSE & held-out RMSE & gap & first step & last step "
        "& grippers \\\\"
    )
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in table_rows(results):
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
