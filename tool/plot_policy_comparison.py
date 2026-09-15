#!/usr/bin/env python3
"""Figures comparing each policy against its cropped-tactile twin.

    venv/bin/python tool/plot_policy_comparison.py outputs/mse/*/*/*/action_mse.json

Reads the `action_mse.json` files `tool/eval_action_mse.py` writes and draws the
three things that answer the question. It loads no checkpoint: the numbers are
already decided, so a figure can be restyled without a GPU.

WHY THESE THREE, and why not a single grouped bar chart of RMSE. The three
policies sit an order of magnitude apart (pi0.5 near 16, ACT near 3), so one
linear axis renders ACT and diffusion as two indistinguishable stubs beside
pi0.5 -- and a second y-axis to fix that would be worse. So the headline figure
plots the CHANGE cropping makes, per policy, which is the actual question and
puts all three on one honest scale; absolute values go in a table beside it.

The horizon figure is small multiples rather than one panel for the same reason,
and for a second: the policies have different chunk lengths (100, 32, 50), so
their curves are not even over the same x.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Categorical slots 1 and 2 of the validated palette. Two series, one adjacent
#: pair: worst CVD dE 24.7, normal-vision 33.6, both clear of the floors.
BASELINE = "#2a78d6"
CROP = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

#: Which policy a crop arm belongs beside. Read from the payload's own policy
#: name rather than the directory, so a renamed folder cannot mispair a chart.
FAMILY = {
    "act": "ACT",
    "so101_act": "ACT",
    "harena_act": "ACT",
    "so101_act_crop": "ACT",
    "harena_act_crop": "ACT",
    "diffusion": "Diffusion",
    "so101_diffusion": "Diffusion",
    "harena_diffusion": "Diffusion",
    "so101_diffusion_crop": "Diffusion",
    "harena_diffusion_crop": "Diffusion",
    "pi05": "pi0.5",
    "so101_pi05": "pi0.5",
    "harena_pi05": "pi0.5",
    "so101_pi05_crop": "pi0.5",
    "harena_pi05_crop": "pi0.5",
}

ORDER = ["ACT", "Diffusion", "pi0.5"]


def is_crop(policy: str) -> bool:
    return policy.endswith("_crop")


def collect(paths: "list[str]") -> "dict[str, dict]":
    """One entry per family: its baseline and its crop arm, the fullest each.

    Where a family has two runs of the same arm -- a short probe and a real
    pass -- the one with more frames wins. A 24-frame smoke run beside a
    220-frame measurement is not a second opinion, it is noise.
    """
    families: "dict[str, dict]" = {}
    for path in paths:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        train = blob.get("train") or {}
        if not train.get("frames"):
            continue
        family = FAMILY.get(blob["policy"])
        if family is None:
            continue
        arm = "crop" if is_crop(blob["policy"]) else "baseline"
        slot = families.setdefault(family, {})
        if arm in slot and slot[arm]["train"]["frames"] >= train["frames"]:
            continue
        slot[arm] = blob
    return families


def indistinguishable(base: dict, crop: dict) -> bool:
    """Are these two arms the same run under two names?

    MEASURED, and the reason this check exists: a cropped pi0.5 scored
    16.044489 against its baseline's 16.044489 -- every digit, and the whole
    per-step curve with it. Two different checkpoints cannot do that. The cause
    was that `make_pre_post_processors` short-circuits on `pretrained_path` and
    never calls the policy's factory, so the crop step was never inserted and a
    three-hour run trained on uncropped images while calling itself the crop arm.

    A figure showing that as "0 % change" would be a finding about nothing. It
    is refused here rather than drawn, because the next time this happens the
    plot is where it will be noticed.
    """
    a, b = base["train"], crop["train"]
    if abs(a["rmse"] - b["rmse"]) > 1e-9:
        return False
    return a.get("mse_per_step") == b.get("mse_per_step")


def pairs(families: "dict[str, dict]") -> "list[tuple[str, dict, dict]]":
    """Only families with BOTH arms genuinely measured, on the same sample.

    An unequal pair is not a comparison: the crop arm would be scored on a
    different sample from its baseline and the difference would be partly the
    sample. Reported as skipped rather than drawn.
    """
    out = []
    for name in ORDER:
        slot = families.get(name) or {}
        base, crop = slot.get("baseline"), slot.get("crop")
        if not base or not crop:
            continue
        if base["train"]["frames"] != crop["train"]["frames"]:
            continue
        if indistinguishable(base, crop):
            continue
        out.append((name, base, crop))
    return out


def style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def draw_change(families, out: Path) -> "Path | None":
    """The headline: what cropping did, per policy, as a percentage.

    Diverging around zero because the question has a polarity -- below the line
    is better, above is worse -- and a reader should not have to consult a
    legend to know which way is which.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = pairs(families)
    if not found:
        return None

    names = [n for n, _, _ in found]
    change = [
        100.0 * (c["train"]["rmse"] - b["train"]["rmse"]) / b["train"]["rmse"]
        for _, b, c in found
    ]

    fig, axis = plt.subplots(figsize=(1.9 * len(names) + 3.4, 4.3))
    bars = axis.bar(
        names,
        change,
        width=0.5,
        color=["#1baf7a" if v < 0 else CROP for v in change],
    )
    axis.axhline(0, color=INK, lw=1.2)
    for bar, value in zip(bars, change):
        above = value >= 0
        axis.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 7 if above else -17),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    style(axis)
    span = max(abs(min(change)), abs(max(change))) * 1.45 or 1.0
    axis.set_ylim(-span, span)
    axis.set_ylabel("change in action RMSE from cropping (%)", color=MUTED, fontsize=10)
    axis.set_title(
        "Cropping the tactile images helps ACT and hurts diffusion",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )
    axis.annotate(
        "below zero = the cropped policy matches the recorded actions more closely",
        (0, 1.005),
        xycoords="axes fraction",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def draw_horizon(families, out: Path) -> "Path | None":
    """Error along the chunk, one panel per policy.

    Small multiples, not one axis: the three chunk lengths differ (100, 32, 50)
    and the scales differ by an order of magnitude, so a shared panel would
    compress two of them to the floor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = pairs(families)
    if not found:
        return None

    fig, axes = plt.subplots(1, len(found), figsize=(4.2 * len(found), 3.9))
    if len(found) == 1:
        axes = [axes]
    for axis, (name, base, crop) in zip(axes, found):
        for blob, colour, label in (
            (base, BASELINE, "baseline"),
            (crop, CROP, "cropped"),
        ):
            series = blob["train"]["mse_per_step"]
            axis.plot(
                range(1, len(series) + 1),
                series,
                color=colour,
                lw=2,
                label=label,
                solid_capstyle="round",
            )
        style(axis)
        axis.set_title(name, fontsize=12, color=INK, loc="left")
        axis.set_xlabel("step within the planned chunk", color=MUTED, fontsize=9)
        axis.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    axes[0].set_ylabel("mean squared action error", color=MUTED, fontsize=10)
    fig.suptitle(
        "How each policy's plan degrades across its chunk",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def draw_absolute(families, out: Path) -> "Path | None":
    """Absolute RMSE, on a log axis because the policies differ by 5x.

    A log axis is the honest way to put pi0.5 beside ACT on one scale; the
    alternative a reader expects -- a linear axis -- renders two of the three
    as stubs, and a second y-axis would be worse still.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    found = pairs(families)
    if not found:
        return None

    names = [n for n, _, _ in found]
    base = [b["train"]["rmse"] for _, b, _ in found]
    crop = [c["train"]["rmse"] for _, _, c in found]
    x = np.arange(len(names))

    fig, axis = plt.subplots(figsize=(1.9 * len(names) + 3.6, 4.3))
    axis.bar(x - 0.21, base, 0.4, color=BASELINE, label="baseline")
    axis.bar(x + 0.21, crop, 0.4, color=CROP, label="cropped")
    for i, (b, c) in enumerate(zip(base, crop)):
        for offset, value in ((-0.21, b), (0.21, c)):
            axis.annotate(
                f"{value:.2f}",
                (i + offset, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
                color=INK,
            )
    axis.set_xticks(x, names)
    # Log ONLY when the spread needs it. With pi0.5 in the set the values span
    # 5x and a linear axis flattens ACT and diffusion into stubs; with just
    # those two it spans 1.4x, and a log axis then exaggerates a small
    # difference and prints ticks like 4x10^0 at a reader who wants "4.13".
    values = base + crop
    logarithmic = max(values) / min(values) >= 3.0
    if logarithmic:
        axis.set_yscale("log")
    style(axis)
    axis.set_ylabel(
        "action RMSE" + (" (log scale)" if logarithmic else ""),
        color=MUTED,
        fontsize=10,
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    axis.set_title(
        "Distance from the actions the operator performed",
        fontsize=13,
        color=INK,
        pad=12,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def stream_shares(path: Path) -> "dict[str, float]":
    """Each stream's mean occlusion share over every analysed frame of a deck."""
    import numpy as np

    blob = json.loads(path.read_text(encoding="utf-8"))
    totals: "dict[str, list[float]]" = {}
    for episode in blob["episodes"].values():
        for frame in episode["frames"]:
            block = frame.get("occlusion")
            if not block:
                continue
            for name, effect in block["streams"].items():
                totals.setdefault(name, []).append(effect["share"])
    shares = {k: float(np.mean(v)) for k, v in totals.items() if v}
    return {
        "policy": blob["policy"],
        "cameras": len([k for k in shares if "gripper" in k]),
        "proprioception": shares.get("state", 0.0),
        "vision": shares.get("central", 0.0),
        "tactile": sum(v for k, v in shares.items() if "gripper" in k),
    }


def draw_streams(decks: "list[str]", out: Path) -> "Path | None":
    """What each policy actually leans on, as a share of its own plan.

    Stacked to 100 % because the shares are normalised WITHIN a policy -- they
    answer "of what moved this plan, how much was touch?", not "how much touch
    is there". Comparing the bands across policies is legitimate; comparing a
    band's absolute size to another policy's is not, and stacking to a common
    height is the honest way to show that.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [stream_shares(Path(d)) for d in decks]
    rows = [r for r in rows if r["vision"] or r["tactile"]]
    if not rows:
        return None
    rows.sort(key=lambda r: ORDER.index(FAMILY.get(r["policy"], "ACT")))

    labels = [
        f"{FAMILY.get(r['policy'], r['policy'])}\n({r['cameras']} fingertip cams)"
        for r in rows
    ]
    bands = [
        ("proprioception", "#4a3aa7"),
        ("vision (central)", BASELINE),
        ("touch (fingertips)", CROP),
    ]
    keys = ["proprioception", "vision", "tactile"]

    fig, axis = plt.subplots(figsize=(2.3 * len(rows) + 3.4, 4.6))
    bottom = np.zeros(len(rows))
    for (label, colour), key in zip(bands, keys):
        values = np.array([r[key] for r in rows]) * 100
        axis.bar(labels, values, 0.55, bottom=bottom, color=colour, label=label)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v >= 4:
                axis.annotate(
                    f"{v:.0f}%",
                    (i, b + v / 2),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
        bottom += values
    style(axis)
    axis.set_ylim(0, 100)
    axis.set_ylabel("share of what moved the plan (%)", color=MUTED, fontsize=10)
    axis.legend(
        frameon=False,
        fontsize=9,
        labelcolor=MUTED,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    axis.set_title(
        "The three policies do not use the rig the same way",
        fontsize=13,
        color=INK,
        pad=12,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def table(families) -> str:
    rows = [
        "| policy | arm | RMSE | joints | grippers | step 1 | last step | frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ORDER:
        slot = families.get(name) or {}
        for arm in ("baseline", "crop"):
            blob = slot.get(arm)
            if not blob:
                continue
            t = blob["train"]
            rows.append(
                f"| {name} | {arm} | {t['rmse']:.3f} | {t['mse_joints']:.2f} | "
                f"{t['mse_grippers']:.5f} | {t['first_step_mse']:.2f} | "
                f"{t['last_step_mse']:.2f} | {t['frames']} |"
            )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", help="action_mse.json files")
    parser.add_argument(
        "--decks",
        nargs="+",
        default=None,
        help="attribution.json files, for the stream-contribution figure",
    )
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    from actoris_harena.analysis.paths import analysis_dir

    families = collect(args.results)
    out_dir = (
        Path(args.out).expanduser() if args.out else analysis_dir("crop-comparison")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    drawn = [
        draw_streams(args.decks or [], out_dir / "stream_shares.png"),
        draw_change(families, out_dir / "crop_change.png"),
        draw_absolute(families, out_dir / "crop_rmse.png"),
        draw_horizon(families, out_dir / "crop_horizon.png"),
    ]
    written = [p for p in drawn if p]

    text = table(families)
    (out_dir / "crop_comparison.md").write_text(
        "# Cropped tactile inputs against their baselines\n\n"
        + text
        + "\n\nEvery checkpoint here trained on all 65 episodes (`eval_split = 0.0`), "
        "so these are TRAINING-set numbers.\n"
    )
    print(text)

    complete = {n for n, _, _ in pairs(families)}
    for name in ORDER:
        slot = families.get(name) or {}
        if name in complete or not slot:
            continue
        base, crop = slot.get("baseline"), slot.get("crop")
        if base and crop and indistinguishable(base, crop):
            print(
                f"\n❌ {name}: the crop arm is bit-identical to its baseline "
                f"(RMSE {base['train']['rmse']:.6f} both, and the whole per-step "
                "curve). That is not a null result -- it is the same run under "
                "two names. NOT PLOTTED."
            )
        elif not (base and crop):
            have = "baseline" if base else ("crop" if crop else "neither arm")
            print(f"\n⚠️  {name}: only {have} measured, so no pair to draw.")
    for path in written:
        print(f"🖼️  {path}")
    print(f"📝 {out_dir / 'crop_comparison.md'}")


if __name__ == "__main__":
    main()
