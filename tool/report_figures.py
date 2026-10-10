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
    # The two world models: new entities, so new hues, placed well clear of the
    # three policy families above and never reused for them.
    "dreamzero": "#17917a",
    "fastwam": "#c2407e",
    # Flow matching: a policy, but neither a port nor a world model.
    "flowmatch": "#8a6d1d",
}

#: Every model drawn on the all-model horizon figure: key, label, marker. The
#: uncropped arm only -- the crop has its own figure and table.
ALL_MODELS = (
    ("act", "ACT", None),
    ("diffusion", "Diffusion", None),
    ("pi05", "pi0.5", None),
    ("flowmatch", "Flow matching", None),
    ("dreamzero", "DreamZero", "o"),
    ("fastwam", "FastWAM", "s"),
)

#: Commands are recorded at 30 Hz, so one planned step is a thirtieth of a second
#: for every model -- which is what lets their curves share a time axis.
ACTION_HZ = 30.0

#: World models: key, label, marker. The marker is the second encoding, so a
#: reader never needs the colour alone to tell the two apart.
WORLD_MODELS = (("dreamzero", "DreamZero", "o"), ("fastwam", "FastWAM", "s"))

#: Seconds from the last OBSERVED frame to horizon step 1, and between steps.
#: Read off the trained configs, at 30 fps. DreamZero: context frames at -48 and
#: -24, predictions at 0, 24, ... 120 (chunk_size 48, two latent frames per
#: chunk). FastWAM: observes frame 0 and predicts 4, 8, ... 32. The two
#: horizons barely overlap -- FastWAM's whole horizon fits inside DreamZero's
#: first step -- so steps are never compared by index, only by time.
#: A camera name a reader should see instead of the dataset key.
CAMERA_LABEL = {"central": "overhead", "tactile_quad": "fingertips, composite"}

HORIZON_SECONDS = {"dreamzero": (24 / 30, 24 / 30), "fastwam": (4 / 30, 4 / 30)}

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


def draw_horizon_all(results: "dict[str, dict]", out: Path, shared: int = 10) -> Path:
    """Error against how far ahead, for every model, on one linear scale.

    Root-mean-square error per planned step, against time ahead in seconds, so
    models that plan ten steps and a hundred sit on one axis without either
    being stretched. Linear on purpose -- a reader compares heights directly --
    and both panels share it. The shaded band is the stretch every model plans.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    drawn = []
    for panel, half, title in zip(
        axes,
        ("validation", "train"),
        ("on the 7 held-out recordings", "on the 58 training recordings"),
    ):
        for key, label, marker in ALL_MODELS:
            result = results.get(key)
            per_step = (result or {}).get(half, {}).get("mse_per_step") or []
            if not per_step:
                continue
            seconds = np.arange(1, len(per_step) + 1) / ACTION_HZ
            panel.plot(
                seconds,
                np.sqrt(per_step),
                label=label,
                color=FAMILY[key],
                linewidth=2.0,
                marker=marker,
                markersize=4,
                markevery=max(1, len(per_step) // 8),
            )
            if half == "validation":
                drawn.append(label)
        panel.axvspan(0, shared / ACTION_HZ, color=GRID, alpha=0.55, zorder=0)
        panel.set_xlim(0, None)
        panel.set_xlabel("time ahead (s)")
        panel.set_title(title, color=INK, fontsize=10)
        tidy(panel)
    axes[0].set_ylabel("action error (RMSE)")
    axes[0].annotate(
        f"first {shared} steps",
        xy=(shared / ACTION_HZ / 2, 0.97),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    missing = [label for _k, label, _m in ALL_MODELS if label not in drawn]
    print(
        f"  horizon, all models: {drawn}" + (f"  MISSING {missing}" if missing else "")
    )
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


TACTILE = (
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)


def tactile_share_series(run: dict, episode: int) -> "tuple[list[float], list[float]]":
    """Seconds into the recording, and the share the tactile cameras carried.

    The four fingertip cameras are summed because the question is what TOUCH
    contributed, not which finger. Time is measured from the start of the
    recording rather than from the start of the dataset, so the axis is a
    duration a reader can compare against the video.
    """
    frames = (((run.get("episodes") or {}).get(str(episode)) or {}).get("frames")) or []
    if not frames:
        return [], []
    start = frames[0]["index"]
    seconds, shares = [], []
    for frame in frames:
        streams = ((frame.get("occlusion") or {}).get("streams")) or {}
        if not streams:
            continue
        seconds.append((frame["index"] - start) / 30.0)
        shares.append(sum(float(streams[c]["share"]) for c in TACTILE if c in streams))
    return seconds, shares


def draw_framewise_shares(
    attribution: "Path | list[Path]", out: Path, episode: int = 58
) -> Path:
    """What touch contributed, moment by moment, on a recording nobody trained on.

    The pooled figures report a mean over the episode, and a mean is the one
    summary that cannot answer the question asked of it here: a channel that is
    ignored for most of a recording and decisive for a second of it has a small
    mean and a large moment. This draws the moment.
    """
    figure, axis = plt.subplots(figsize=(9.0, 3.6))
    drawn = 0
    roots = [attribution] if isinstance(attribution, Path) else list(attribution)
    for arm, label, _crop in ARMS:
        paths = [
            r / f"heldout-{arm}" / "attribution.json"
            for r in roots
            if (r / f"heldout-{arm}" / "attribution.json").is_file()
        ]
        if len(paths) > 1:
            raise SystemExit(f"❌ heldout-{arm} found in more than one directory")
        if not paths:
            continue
        path = paths[0]
        seconds, shares = tactile_share_series(json.loads(path.read_text()), episode)
        if not seconds:
            continue
        axis.plot(seconds, shares, label=label, **style_for(arm))
        drawn += 1
    axis.set_xlabel("seconds into the recording")
    axis.set_ylabel("share carried by touch")
    axis.set_ylim(bottom=0)
    axis.set_title(
        "Touch is not read evenly: its share moves several-fold within one recording",
        color=INK,
        fontsize=12,
    )
    axis.legend(frameon=False, fontsize=9, ncol=2)
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  framewise tactile share: {drawn} arms -> {out}")
    return out


def prediction_margins(payload: dict) -> "dict[str, list[float]]":
    """Per camera, PSNR minus the held-last-frame PSNR at every horizon step.

    The margin and not the PSNR, because holding scores highly on a camera that
    barely moves: a model is only predicting where it beats doing nothing.
    """
    return {
        camera: [
            round(m - h, 4) for m, h in zip(values["psnr"], values["psnr_baseline"])
        ]
        for camera, values in payload.get("per_camera", {}).items()
    }


def fingertip_margin(per_camera: "dict[str, list[float]]") -> "list[float] | None":
    """One fingertip series: the composite if the model saw one, else the mean.

    FastWAM reads the four fingertips as a single 2x2 composite and is scored on
    it; DreamZero reads them separately. The mean of four margins is the closest
    like-for-like, and the caption says that is what it is.
    """
    if "tactile_quad" in per_camera:
        return per_camera["tactile_quad"]
    series = [v for k, v in per_camera.items() if "gripper" in k]
    if not series:
        return None
    return [float(np.mean(step)) for step in zip(*series)]


def draw_prediction(results: "dict[str, dict]", out: Path) -> "Path | None":
    """Each world model's margin over holding, against seconds ahead."""
    margins = {key: prediction_margins(r) for key, r in results.items() if r}
    if not margins:
        print("  prediction: no world-model results; figure skipped")
        return None
    panels = (
        ("overhead camera", lambda m: m.get("central")),
        ("fingertips", fingertip_margin),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for axis, (title, pick) in zip(axes, panels):
        axis.axhline(0, color=MUTED, linewidth=1.0)
        for key, label, marker in WORLD_MODELS:
            series = pick(margins.get(key, {})) if key in margins else None
            if not series:
                continue
            first, spacing = HORIZON_SECONDS[key]
            seconds = [first + i * spacing for i in range(len(series))]
            axis.plot(
                seconds,
                series,
                color=FAMILY[key],
                marker=marker,
                markersize=5,
                linewidth=2.0,
                label=label,
            )
        axis.set_title(title, fontsize=11, color=INK)
        axis.set_xlabel(
            "seconds after the last observed frame", fontsize=9, color=MUTED
        )
        tidy(axis)
    axes[0].set_ylabel("PSNR above holding (dB)", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.suptitle(
        "Above the line, the model predicts better than repeating the last frame",
        color=INK,
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  prediction margin: {sorted(margins)} -> {out}")
    return out


def write_prediction_table(results: "dict[str, dict]", out: Path) -> "Path | None":
    """Per camera and model: PSNR and SSIM beside holding, and steps won.

    A step counts as won only by a clear margin (0.1 dB); a step the model
    ties with holding is not a step on which it predicted anything.
    """
    labels = {key: label for key, label, _m in WORLD_MODELS}
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{PSNR (dB)} & \multicolumn{2}{c}{SSIM} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"camera & model & model & held & model & held & steps won \\",
        r"\midrule",
    ]
    rows = 0
    for key, payload in results.items():
        for camera, v in (payload or {}).get("per_camera", {}).items():
            won = sum(1 for m, h in zip(v["psnr"], v["psnr_baseline"]) if m - h > 0.1)
            lines.append(
                f"{CAMERA_LABEL.get(camera, camera.replace('_', ' '))} & "
                f"{labels.get(key, key)} & "
                f"{np.mean(v['psnr']):.1f} & {np.mean(v['psnr_baseline']):.1f} & "
                f"{np.mean(v['ssim']):.3f} & {np.mean(v['ssim_baseline']):.3f} & "
                f"{won} of {len(v['psnr'])} \\\\"
            )
            rows += 1
    if not rows:
        return None
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  prediction table: {rows} rows -> {out}")
    return out


def write_world_model_table(
    policies: "dict[str, dict]", world: "dict[str, dict]", out: Path
) -> "Path | None":
    """Every family, baselines only, over the steps ALL of them plan.

    FastWAM plans ten actions, so ten is the only horizon every row shares; a
    wider one would silently drop it, and each family's own horizon would
    compare unequal questions -- the reason the policy table has a
    shared-horizon column at all.
    """
    candidates = [(label, policies.get(arm)) for arm, label in ARMS_LABEL]
    candidates += [(label, world.get(key)) for key, label, _m in WORLD_MODELS]
    rows: "list[tuple[str, dict]]" = [(lab, r) for lab, r in candidates if r]
    if not rows:
        return None
    common = min(len(r["train"]["mse_per_step"]) for _l, r in rows)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"& & \multicolumn{{3}}{{c}}{{over the first {common} steps}} \\",
        r"\cmidrule(lr){3-5}",
        r"model & horizon & train & held-out & gap \\",
        r"\midrule",
    ]
    for label, r in rows:
        train = rmse_over(r, "train", common)
        held = rmse_over(r, "validation", common)
        lines.append(
            f"{label} & {r['train']['horizon']} & {train:.3f} & {held:.3f} & "
            f"{held / train:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  world-model table: {len(rows)} rows over {common} steps -> {out}")
    return out


def write_floor_table(directory: Path, out: Path, steps: int = 10) -> "Path | None":
    """The sampler's own spread: held-out error under three seeds, per model.

    ``directory`` holds ``<model>-seed<N>.json`` results, each scoring ONLY the
    held-out recordings (so their block is labelled train). A difference
    between two models smaller than this spread is not a finding.
    """
    labels = dict(ARMS_LABEL) | {k: lab for k, lab, _m in WORLD_MODELS}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"& \multicolumn{{2}}{{c}}{{own horizon}} & \multicolumn{{2}}{{c}}{{first {steps} steps}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"model & lowest & highest & lowest & highest \\",
        r"\midrule",
    ]
    rows = 0
    for model in [k for k, _l in ARMS_LABEL] + [k for k, _l, _m in WORLD_MODELS]:
        paths = sorted(directory.glob(f"{model}-seed*.json"))
        if len(paths) < 2:
            continue
        blocks = [json.loads(p.read_text())["train"] for p in paths]
        own = [float(np.sqrt(np.mean(b["mse_per_step"]))) for b in blocks]
        near = [float(np.sqrt(np.mean(b["mse_per_step"][:steps]))) for b in blocks]
        lines.append(
            f"{labels.get(model, model)} ({len(paths)} seeds) & {min(own):.2f} & "
            f"{max(own):.2f} & {min(near):.2f} & {max(near):.2f} \\\\"
        )
        rows += 1
    if not rows:
        return None
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  sampler floor: {rows} models -> {out}")
    return out


#: The three ACT arms that separate the rim from the stretch (Stage 9).
STRETCH_ARMS = (
    ("act", "not cropped"),
    ("act_crop", "cropped and stretched back"),
    ("act_crop_noresize", "cropped, not stretched"),
)


def write_stretch_table(results: "dict[str, dict]", out: Path) -> "Path | None":
    """Held-out and trained-on error for the three ACT arms, at three horizons.

    The crop removed the gel rim AND stretched what remained back to full
    height; the third arm removes the rim alone. All three share one recorded
    configuration -- seed, batch, steps, split -- and differ in that one flag.
    """
    rows = [(label, results.get(arm)) for arm, label in STRETCH_ARMS]
    if sum(1 for _l, r in rows if r) < 3:
        return None
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{held-out} & trained-on \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
        r"ACT arm & first 10 & first 32 & all 100 & all 100 \\",
        r"\midrule",
    ]
    for label, r in rows:
        assert r is not None
        lines.append(
            f"{label} & {rmse_over(r, 'validation', 10):.2f} & "
            f"{rmse_over(r, 'validation', 32):.2f} & {rmse_over(r, 'validation'):.2f} & "
            f"{rmse_over(r, 'train'):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  stretch table -> {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mse",
        nargs="+",
        default=["outputs/mse/thanos/2026-09-16", "outputs/mse/thanos/2026-09-25"],
        help="directories of <prefix><arm>.json action-error results; an arm "
        "found in two is an error, so a rerun cannot silently shadow the original",
    )
    parser.add_argument("--prefix", default="split10-")
    parser.add_argument(
        "--attribution",
        nargs="+",
        default=["outputs/analysis/2026-09-18", "outputs/analysis/2026-09-25"],
        help="directories of heldout-<arm>/attribution.json runs",
    )
    parser.add_argument("--episode", type=int, default=58)
    parser.add_argument(
        "--prediction",
        nargs="*",
        default=[
            "dreamzero=outputs/analysis/2026-09-27/dreamzero-prediction/prediction.json",
            "fastwam=outputs/analysis/2026-09-27/fastwam-prediction/prediction.json",
        ],
        help="model=path pairs of world-model prediction.json results",
    )
    parser.add_argument(
        "--floor",
        nargs="*",
        default=["outputs/mse/floor"],
        help="directory of <model>-seed<N>.json held-out rescorings",
    )
    parser.add_argument(
        "--world-mse",
        nargs="*",
        default=[
            "dreamzero=outputs/mse/thanos/2026-09-25/split10-dreamzero.json",
            "fastwam=outputs/mse/viking/2026-09-25/split10-fastwam.json",
        ],
        help="model=path pairs of world-model action_mse.json results",
    )
    parser.add_argument(
        "--results-root",
        default="outputs/mse",
        help="every <machine>/<date>/<prefix><arm>.json under it feeds the "
        "all-model, crop and sensor tables and the all-model horizon figure",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="the dataset directory, for the trajectory strip and the rim "
        "evidence; both are skipped without it",
    )
    parser.add_argument(
        "--wm-prediction",
        nargs="*",
        default=[
            "dreamzero=outputs/analysis/2026-09-27/dreamzero-prediction/prediction.json",
            "fastwam=outputs/analysis/2026-09-27/fastwam-prediction/prediction.json",
        ],
        help="model=path pairs scored per sensor with reconstruction",
    )
    parser.add_argument(
        "--gradcam-evidence",
        default="outputs/analysis/2026-09-07/act-all/attribution.json",
        help="the ACT attribution whose Grad-CAM first showed edge weight",
    )
    parser.add_argument(
        "--gradcam-frame",
        type=lambda text: tuple(int(v) for v in text.split(":")),
        default=(0, 180),
        help="EPISODE:FRAME of the Grad-CAM evidence figure (default 0:180)",
    )
    parser.add_argument(
        "--h2h",
        nargs=2,
        default=None,
        metavar=("FASTWAM_NPZ", "DREAMZERO_NPZ"),
        help="filmstrips of the two world action models from the same seen frame",
    )
    parser.add_argument("--out", default=None, help="where the figures go")
    args = parser.parse_args()

    from actoris_harena.analysis.paths import analysis_dir

    import common.rig_profile  # noqa: F401  -- the rig declares itself first

    out = Path(args.out) if args.out else analysis_dir("training-report")
    out.mkdir(parents=True, exist_ok=True)

    results: "dict[str, dict]" = {}
    for directory in args.mse:
        found = load_arms(Path(directory), args.prefix)
        clash = sorted(set(found) & set(results))
        if clash:
            raise SystemExit(
                f"❌ {', '.join(clash)} found in more than one --mse directory"
            )
        results.update(found)
    missing = [label for arm, label, _c in ARMS if arm not in results]
    print(f"arms found: {sorted(results)}")
    if missing:
        # Named, every run, and not once at the start. A reader of the log has
        # to see which arms the figures do NOT contain.
        print(f"MISSING, drawn as gaps rather than omitted: {missing}")

    draw_horizon(results, out / "horizon_decay.png")
    draw_gap(results, out / "train_vs_heldout.png")
    write_table(results, out / "policy_table.tex")
    attributions = [Path(a) for a in args.attribution if Path(a).is_dir()]
    if attributions:
        draw_framewise_shares(attributions, out / "framewise_shares.png", args.episode)
    else:
        print(f"no attribution runs at {args.attribution}; framewise figure skipped")
    predictions = {}
    for pair in args.prediction:
        key, _, path = pair.partition("=")
        if Path(path).is_file():
            predictions[key] = json.loads(Path(path).read_text())
        else:
            print(f"  prediction: no {key} result at {path}")
    draw_prediction(predictions, out / "prediction_margin.png")
    write_prediction_table(predictions, out / "prediction_table.tex")
    world = {}
    for pair in args.world_mse:
        key, _, path = pair.partition("=")
        if Path(path).is_file():
            world[key] = json.loads(Path(path).read_text())
        else:
            print(f"  world-model action error: no {key} result at {path}")
    write_world_model_table(results, world, out / "world_model_table.tex")
    stretch = dict(results)
    for directory in args.mse:
        path = Path(directory) / f"{args.prefix}act_crop_noresize.json"
        if path.is_file():
            stretch["act_crop_noresize"] = json.loads(path.read_text())
    write_stretch_table(stretch, out / "stretch_table.tex")
    for directory in args.floor:
        write_floor_table(Path(directory), out / "floor_table.tex")

    everything = load_results(Path(args.results_root), args.prefix)
    missing_rows = [label for arm, label in ACTION_ROWS if arm not in everything]
    if missing_rows:
        print(f"MISSING from the action table, drawn as dashes: {missing_rows}")
    draw_horizon_all(everything, out / "horizon_all.png")
    write_action_table(everything, out / "action_table.tex")
    write_crop_table(everything, out / "crop_table.tex")
    write_sensor_table(everything, out / "sensor_table.tex")
    revised = {}
    for pair in args.wm_prediction:
        key, _, path = pair.partition("=")
        if Path(path).is_file():
            revised[key] = json.loads(Path(path).read_text())
        else:
            print(f"  per-sensor prediction: no {key} result at {path}")
    write_wm_table(revised, out / "wm_table.tex")
    draw_rim_attention([Path(a) for a in args.attribution], out / "rim_attention.png")
    decks = [(label, Path(path)) for label, path in PATCH_DECKS]
    if args.dataset:
        draw_patch_overlays(
            decks,
            Path(args.dataset).expanduser(),
            args.episode,
            0.45,
            out / "patch_maps.png",
        )
    else:
        draw_patch_maps(decks, out / "patch_maps.png")
    draw_stream_shares_all(
        [(label, Path(path)) for label, path in STREAM_DECKS], out / "stream_shares.png"
    )
    if args.h2h:
        from tool.eval_world_model import load_filmstrip

        fw_path, dz_path = (Path(p) for p in args.h2h)
        if fw_path.is_file() and dz_path.is_file():
            draw_filmstrip_h2h(
                load_filmstrip(fw_path),
                load_filmstrip(dz_path),
                out / "filmstrip_h2h.png",
            )
        else:
            print(f"  head-to-head filmstrip: missing {fw_path} or {dz_path}")
    if args.dataset:
        dataset = Path(args.dataset).expanduser()
        draw_trajectory_strip(dataset, args.episode, out / "trajectory_strip.png")
        draw_rim_evidence(
            dataset, list(range(65)), out / "rim_evidence.png", example=args.episode
        )
        draw_model_inputs(dataset, args.episode, 0.45, out / "model_inputs.png")
        draw_gradcam_evidence(
            Path(args.gradcam_evidence),
            dataset,
            args.gradcam_frame[0],
            args.gradcam_frame[1],
            out / "gradcam_evidence.png",
        )
    print(f"\nwrote {out}")
    return 0


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
    # Placed INSIDE the axes, not at their right edge: at `len(arms) - 0.5` the
    # text sat past the last bar and rendered outside the visible area, which is
    # the sort of thing that only shows up on looking at the picture.
    axis.annotate(
        "no gap at all (held-out error = trained-on error)",
        xy=(-0.42, 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
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


def episode_frames(
    dataset: Path, key: str, episode: int, fractions: "list[float]"
) -> "list[tuple[float, np.ndarray]]":
    """Frames of one camera at the given fractions of one recording's length.

    Returns ``(seconds from the recording's start, RGB frame)`` pairs, decoded
    by seeking into the recording's slice of the shared video file -- the same
    route the contact sheet takes, for the same reason: the metadata names each
    recording's span inside a file that holds several.
    """
    import av
    import pandas as pd

    table = pd.concat(
        pd.read_parquet(p)
        for p in sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    )
    row = table[table["episode_index"] == episode].iloc[0]
    chunk = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    start = float(row[f"videos/{key}/from_timestamp"])
    end = float(row[f"videos/{key}/to_timestamp"])
    path = (
        dataset / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    )
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for fraction in fractions:
            wanted = start + fraction * (end - start - 1.0 / 30)
            container.seek(int(wanted / stream.time_base), stream=stream)
            frame = None
            for candidate in container.decode(stream):
                frame = candidate
                if float(candidate.pts * stream.time_base) >= wanted - 1e-3:
                    break
            assert frame is not None
            out.append((wanted - start, frame.to_ndarray(format="rgb24")))
    return out


def draw_trajectory_strip(
    dataset: Path, episode: int, out: Path, count: int = 20
) -> Path:
    """One recording as the overhead camera saw it, ``count`` evenly spaced frames.

    Replaces a sheet of one frame per recording, which showed sixty-five near
    copies of one picture: what a reader needs first is what ONE demonstration
    looks like from start to finish.
    """
    frames = episode_frames(
        dataset, "observation.images.central", episode, list(np.linspace(0, 1, count))
    )
    columns = 5
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 1.65))
    for axis in np.ravel(axes):
        axis.axis("off")
    for axis, (seconds, image) in zip(np.ravel(axes), frames):
        axis.imshow(image)
        axis.set_title(f"{seconds:.1f} s", fontsize=8, color=MUTED, pad=2)
    figure.tight_layout(pad=0.3)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  trajectory strip: recording {episode}, {count} frames -> {out}")
    return out


def rim_profiles(
    dataset: Path, episodes: "list[int]"
) -> "dict[str, dict[str, np.ndarray]]":
    """Mean brightness per row and per column of each fingertip, before contact.

    The first frame of each recording, where both grippers are still open and
    nothing has touched a gel, so any structure at the edge is the sensor's own
    and not contact. Each profile is divided by the mean of its middle fifth,
    so ``1.1`` reads as "ten per cent brighter than the centre of the gel".
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    out: "dict[str, dict[str, np.ndarray]]" = {}
    for camera in TACTILE_CAMERAS:
        key = f"observation.images.{camera}"
        stack = np.stack(
            [
                episode_frames(dataset, key, e, [0.0])[0][1].mean(axis=2)
                for e in episodes
            ]
        ).astype(float)
        mean = stack.mean(axis=0)
        rows, cols = mean.mean(axis=1), mean.mean(axis=0)

        def relative(profile):
            n = len(profile)
            middle = profile[int(n * 0.4) : int(np.ceil(n * 0.6))].mean()
            return profile / middle

        out[camera] = {"rows": relative(rows), "cols": relative(cols), "image": mean}
    return out


#: The ridge crop's kept columns, as fractions of the width (rows 0.1-0.9).
RIDGE_CROP = (0.36, 0.90)


def draw_rim_evidence(
    dataset: Path, episodes: "list[int]", out: Path, example: int = 0
) -> "tuple[Path, dict[str, dict[str, float]]]":
    """The tactile rim, and the two crops that remove it, on every fingertip.

    Top row: one pre-contact frame per sensor with the three crop boxes drawn
    on it -- dashed, the rows-only crop (top and bottom tenth); solid, the
    four-edge crop (a tenth off every edge); dash-dot, the ridge crop -- and
    the gel's ridge marked where the column profile peaks. Bottom row: the brightness of each
    row and each column relative to the gel's centre, averaged over the first
    frame of every listed recording, with the crop lines marked. Returns the
    edge brightness per camera so the text quotes the figure's own numbers.
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    profiles = rim_profiles(dataset, episodes)
    edges: "dict[str, dict[str, float]]" = {}
    figure, axes = plt.subplots(2, 4, figsize=(11, 4.6), height_ratios=(1.15, 1))
    for column, camera in enumerate(TACTILE_CAMERAS):
        image = episode_frames(dataset, f"observation.images.{camera}", example, [0.0])[
            0
        ][1]
        height, width = image.shape[:2]
        top = axes[0, column]
        top.imshow(image)
        top.axis("off")
        top.set_title(
            f"{camera.replace('_', ' ')} ({SENSOR_SHORT[camera]})",
            fontsize=8,
            color=INK,
        )
        top.add_patch(
            plt.Rectangle(
                (0, 0.1 * height),
                width - 1,
                0.8 * height,
                fill=False,
                edgecolor="#f5f5f5",
                linestyle="--",
                linewidth=1.4,
            )
        )
        top.add_patch(
            plt.Rectangle(
                (0.1 * width, 0.1 * height),
                0.8 * width,
                0.8 * height,
                fill=False,
                edgecolor="#ffd23f",
                linewidth=1.6,
            )
        )
        # The ridge crop: the four-edge box with its left edge moved past the
        # ridge (RIDGE_CROP), drawn in cyan.
        left, right = RIDGE_CROP
        top.add_patch(
            plt.Rectangle(
                (left * width, 0.1 * height),
                (right - left) * width,
                0.8 * height,
                fill=False,
                edgecolor="#3fd0ff",
                linewidth=1.6,
                linestyle="-.",
            )
        )
        # The ridge itself, located from this sensor's own column profile:
        # the brightest column, which is not at the rim.
        cols = profiles[camera]["cols"]
        ridge = int(np.argmax(cols)) / len(cols)
        top.annotate(
            "ridge",
            xy=(ridge * width, 0.55 * height),
            xytext=(ridge * width + 0.12 * width, 0.8 * height),
            color="#ff4fa3",
            fontsize=8,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#ff4fa3", "lw": 1.5},
        )
        bottom = axes[1, column]
        rows, cols = profiles[camera]["rows"], profiles[camera]["cols"]
        bottom.plot(
            np.linspace(0, 1, len(rows)),
            rows,
            color=FAMILY["act"],
            lw=2,
            label="rows (top to bottom)",
        )
        bottom.plot(
            np.linspace(0, 1, len(cols)),
            cols,
            color=FAMILY["diffusion"],
            lw=2,
            label="columns (left to right)",
        )
        for x in (0.1, 0.9):
            bottom.axvline(x, color=MUTED, lw=1, ls=":")
        bottom.axvline(RIDGE_CROP[0], color="#3fd0ff", lw=1.2, ls="-.")
        bottom.annotate(
            "ridge",
            xy=(ridge, cols.max()),
            xytext=(ridge + 0.12, cols.max()),
            color="#ff4fa3",
            fontsize=7,
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#ff4fa3", "lw": 1.0},
        )
        bottom.axhline(1.0, color=GRID, lw=1)
        bottom.set_xlabel("position across the image", fontsize=8)
        if column == 0:
            bottom.set_ylabel("brightness / centre", fontsize=8)
        bottom.tick_params(labelsize=7)
        tidy(bottom)
        edges[camera] = {
            "rows": float(
                max(rows[: len(rows) // 10].max(), rows[-len(rows) // 10 :].max())
            ),
            "cols": float(
                max(cols[: len(cols) // 10].max(), cols[-len(cols) // 10 :].max())
            ),
        }
    axes[1, 0].legend(fontsize=7, frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  rim evidence: {len(episodes)} first frames per sensor -> {out}")
    for camera, value in edges.items():
        print(
            f"    {camera:<26} brightest outer tenth: rows x{value['rows']:.3f}"
            f"  cols x{value['cols']:.3f}"
        )
    return out, edges


#: Result files whose name marks them void. Kept on disk as the record of a
#: fault, and never drawn.
VOID_MARKERS = ("superseded", "INVALID")


def load_results(root: Path, prefix: str = "split10-") -> "dict[str, dict]":
    """Every action-error result under ``<root>/<machine>/<date>/``, by arm.

    One arm found twice is an error rather than a choice: a rescoring that
    silently shadowed the original is how a table ends up describing a run
    nobody can point to.
    """
    found: "dict[str, dict]" = {}
    where: "dict[str, Path]" = {}
    for path in sorted(root.glob(f"*/*/{prefix}*.json")):
        arm = path.stem[len(prefix) :]
        if any(marker in arm for marker in VOID_MARKERS):
            continue
        if arm in found:
            raise SystemExit(f"❌ {arm} is in both {where[arm]} and {path}")
        found[arm] = json.loads(path.read_text())
        where[arm] = path
    return found


#: Rows of the all-model action table: arm key, label.
ACTION_ROWS = (
    ("act", "ACT"),
    ("diffusion", "Diffusion"),
    ("pi05", "pi0.5"),
    ("pi05_long", "pi0.5, three passes"),
    ("pi05_lorawide", "pi0.5, wide LoRA"),
    ("flowmatch", "Flow matching"),
    ("dreamzero", "DreamZero"),
    ("fastwam", "FastWAM"),
)

#: What each model reads, in the table's shorthand: O the overhead camera, nT n
#: fingertips, q the twelve joint positions, a the past commands. Keyed by the
#: family, so every pi0.5 variant reads the same.
MODEL_INPUTS = {
    "act": "O+4T+q",
    "diffusion": "O+4T+q",
    "pi05": "O+2T+q",
    "flowmatch": "O+4T+q",
    "dreamzero": "O+4T+q+a",
    "fastwam": "O+4T+q",
}


def inputs_of(arm: str) -> str:
    """The input shorthand for an arm key such as ``pi05_lorawide``."""
    for family in sorted(MODEL_INPUTS, key=len, reverse=True):
        if arm == family or arm.startswith(family + "_"):
            return MODEL_INPUTS[family]
    raise KeyError(arm)


def bold_lowest(rows: "list[list[str]]", columns: "list[int]") -> "list[list[str]]":
    """Set the lowest number in each named column in bold.

    Lower is better for every error these tables show. A cell that is not a
    number (a dash, ``n/a``) takes no part, and a tie bolds every holder.
    """
    out = [list(row) for row in rows]
    for column in columns:
        values: "list[float | None]" = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (ValueError, IndexError):
                values.append(None)
        numbers = [v for v in values if v is not None]
        if not numbers:
            continue
        best = min(numbers)
        for i, value in enumerate(values):
            if value == best:
                out[i][column] = rf"\textbf{{{rows[i][column]}}}"
    return out


def _cells(result: "dict | None", steps: "int | None" = None) -> "list[str]":
    """Train, test and gap over ``steps`` (or the whole plan), as table cells."""
    if not result:
        return ["--", "--", "--"]
    train = rmse_over(result, "train", steps)
    test = rmse_over(result, "validation", steps)
    return [f"{train:.2f}", f"{test:.2f}", f"{test / train:.1f}"]


def write_action_table(results: "dict[str, dict]", out: Path, common: int = 10) -> Path:
    """Every model, over its own plan and over the steps they all plan.

    A missing model is a row of dashes, never an omitted line: a table that lost
    a row silently reads as a complete comparison.
    """
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        rf"& & & \multicolumn{{3}}{{c}}{{over its own plan}} & "
        rf"\multicolumn{{2}}{{c}}{{over the first {common} steps}} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"model & inputs & plan & train & test & gap & test & gap \\",
        r"\midrule",
    ]
    rows = []
    for arm, label in ACTION_ROWS:
        result = results.get(arm)
        plan = str(result["validation"]["horizon"]) if result else "--"
        own = _cells(result)
        near = _cells(result, common)
        rows.append([label, inputs_of(arm), plan, *own, near[1], near[2]])
    # Train, test and the shared-steps test; never the gap, where small is not
    # good -- pi0.5's is small because it never fitted its training set.
    lines += [" & ".join(row) + r" \\" for row in bold_lowest(rows, [3, 4, 6])]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  action table -> {out}")
    return out


#: Crop table: model label, then the arm key for each column. None means that
#: crop does not exist for the model, which is not the same as pending.
CROP_ROWS = (
    ("ACT", "act", "act_crop", "act_crop_noresize", "act_crop_edges", "act_crop_ridge"),
    (
        "Diffusion",
        "diffusion",
        "diffusion_crop",
        None,
        "diffusion_crop_edges",
        "diffusion_crop_ridge",
    ),
    ("pi0.5", "pi05", "pi05_crop", None, "pi05_crop_edges", "pi05_crop_ridge"),
    (
        "DreamZero",
        "dreamzero",
        None,
        None,
        "dreamzero_crop_edges",
        "dreamzero_crop_ridge",
    ),
    ("FastWAM", "fastwam", None, None, "fastwam_crop_edges", "fastwam_crop_ridge"),
)


def write_crop_table(results: "dict[str, dict]", out: Path) -> Path:
    """Test RMSE over each model's own plan, uncropped and under each crop."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"model & no crop & rows & rows, unstretched & four edges & ridge \\",
        r"\midrule",
    ]
    for label, *arms in CROP_ROWS:
        cells = []
        for arm in arms:
            if arm is None:
                cells.append("n/a")
            elif arm in results:
                cells.append(f"{rmse_over(results[arm], 'validation'):.2f}")
            else:
                cells.append("--")
        # Bold within a row: the question is which crop suits THIS model. The
        # row is bolded as a one-column table, so the minimum runs along it.
        row = [r[0] for r in bold_lowest([[c] for c in cells], [0])]
        lines.append(" & ".join([label, *row]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  crop table -> {out}")
    return out


SENSOR_ROWS = (
    ("act_central", "overhead"),
    ("act_central_2tactile", "overhead + one fingertip per arm"),
    ("act", "overhead + all four fingertips"),
)


def write_sensor_table(results: "dict[str, dict]", out: Path, common: int = 10) -> Path:
    """ACT on three camera sets, all with proprioception."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{over its 100-step plan} & "
        rf"\multicolumn{{2}}{{c}}{{first {common} steps}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"cameras & train & test & gap & test & gap \\",
        r"\midrule",
    ]
    rows = []
    for arm, label in SENSOR_ROWS:
        result = results.get(arm)
        near = _cells(result, common)
        rows.append([label, *_cells(result), near[1], near[2]])
    lines += [" & ".join(row) + r" \\" for row in bold_lowest(rows, [1, 2, 4])]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  sensor table -> {out}")
    return out


#: The fingertip sensors by arm and finger, as the report abbreviates them.
SENSOR_SHORT = {
    "left_arm_left_gripper": "LLG",
    "left_arm_right_gripper": "LRG",
    "right_arm_left_gripper": "RLG",
    "right_arm_right_gripper": "RRG",
}

#: Sensor rows of the world-model table, in the order a reader expects.
WM_SENSORS = (("central", "overhead"),) + tuple(
    (camera, SENSOR_SHORT[camera]) for camera in TACTILE
)


def bold_wm_best(rows: "list[list[str]]") -> "list[list[str]]":
    """Bold the better of prediction and repeat-last in each row (higher PSNR),
    and, per sensor, the model with the larger margin at each instant.

    Reconstruction is left alone: it is a ceiling, not a competitor, and each
    model reconstructs at its own resolution.
    """
    out = [list(row) for row in rows]

    def bold(i: int, j: int) -> None:
        out[i][j] = rf"\textbf{{{rows[i][j]}}}"

    for i, row in enumerate(rows):
        prediction, repeat = float(row[3]), float(row[4])
        if prediction >= repeat:
            bold(i, 3)
        if repeat >= prediction:
            bold(i, 4)
    for sensor in {row[0] for row in rows}:
        members = [i for i, row in enumerate(rows) if row[0] == sensor]
        for column in (5, 6):
            best = max(float(rows[i][column]) for i in members)
            for i in members:
                if float(rows[i][column]) == best:
                    bold(i, column)
    return out


def write_wm_table(
    predictions: "dict[str, dict]", out: Path, at: float = 0.8
) -> "Path | None":
    """Reconstruction, prediction and repeat-last per sensor, both world models.

    The margin is read at ``at`` seconds ahead -- the one instant both models
    predict -- and at the last step of each model's own horizon.
    """
    rows = []
    for key, label, _marker in WORLD_MODELS:
        payload = predictions.get(key)
        if not payload:
            continue
        first, spacing = HORIZON_SECONDS[key]
        for camera, camera_label in WM_SENSORS:
            values = payload["per_camera"].get(camera)
            if not values:
                continue
            model = np.array(values["psnr"])
            held = np.array(values["psnr_baseline"])
            times = first + spacing * np.arange(len(model))
            index = int(np.argmin(np.abs(times - at)))
            recon = values.get("recon_psnr")
            rows.append(
                [
                    camera_label,
                    label,
                    f"{recon[0]:.1f}" if recon else "--",
                    f"{model.mean():.1f}",
                    f"{held.mean():.1f}",
                    f"{model[index] - held[index]:+.1f}",
                    f"{model[-1] - held[-1]:+.1f}",
                ]
            )
    if not rows:
        # A table the report \input's must exist, so an unscored state is a
        # visible placeholder rather than a build error or a silent gap.
        out.write_text(
            "\\begin{tabular}{l}\\pending{world models not yet scored per sensor}\\end{tabular}\n"
        )
        print(f"  world-model sensor table: nothing scored yet -> placeholder {out}")
        return out
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"& & & & & \multicolumn{2}{c}{margin} \\",
        r"\cmidrule(lr){6-7}",
        rf"sensor & model & reconstruction & prediction & repeat last & "
        rf"at \SI{{{at}}}{{\second}} & at end \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in bold_wm_best(rows)]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"  world-model sensor table: {len(rows)} rows -> {out}")
    return out


#: The rim figure's arms: family, label, then (arm key, crop label) per bar.
RIM_ARMS = (
    (
        "act",
        "ACT",
        (("act", "no crop"), ("act_crop", "rows"), ("act_crop_edges", "four edges")),
    ),
    (
        "diffusion",
        "Diffusion",
        (
            ("diffusion", "no crop"),
            ("diffusion_crop", "rows"),
            ("diffusion_crop_edges", "four edges"),
        ),
    ),
    (
        "wam",
        "World action models (no crop)",
        (("fastwam", "FastWAM"), ("dreamzero", "DreamZero")),
    ),
)
RIM_BANDS = (0.10, 0.15, 0.20)

#: Grad-CAM runs of the world action models, through their frozen autoencoders
#: (`analysis.gradients.autoencoder_grad_cam`). Not ``heldout-gradcam-<arm>``
#: runs: they were made later, by `analyse_policy_inputs.py` directly.
WAM_GRADCAM = {
    "fastwam": Path("outputs/analysis/2026-10-09/gradcam-fastwam/attribution.json"),
    "dreamzero": Path("outputs/analysis/2026-10-10/gradcam-dreamzero/attribution.json"),
}


def draw_rim_attention(directories: "list[Path]", out: Path) -> "Path | None":
    """Grad-CAM weight in an edge band of the tactile maps, against a flat map.

    Read from the held-out Grad-CAM runs (``heldout-gradcam-<arm>``). The band is
    a fraction of the image, so maps of different resolution are comparable;
    ``tool/rim_attention.py`` is the measure and says why. A value of one is
    what a map with no preference for position would give.
    """
    from tool.rim_attention import pooled

    def find(arm: str) -> "Path | None":
        if arm in WAM_GRADCAM:
            return WAM_GRADCAM[arm] if WAM_GRADCAM[arm].is_file() else None
        hits = [d / f"heldout-gradcam-{arm}" / "attribution.json" for d in directories]
        hits = [h for h in hits if h.is_file()]
        return hits[-1] if hits else None

    figure, axes = plt.subplots(1, len(RIM_ARMS), figsize=(14, 3.6), sharey=True)
    styles = ({"alpha": 1.0}, {"alpha": 0.55}, {"alpha": 0.55, "hatch": "///"})
    drawn = []
    for axis, (family, title, bars) in zip(axes, RIM_ARMS):
        xs = np.arange(len(RIM_BANDS))
        width = 0.26
        for offset, ((arm, label), style) in enumerate(zip(bars, styles)):
            if family == "wam":  # two models side by side, neither a crop
                style = {"alpha": 1.0}
            path = find(arm)
            if path is None:
                continue
            values = []
            for band in RIM_BANDS:
                observed, flat, _n = pooled(path, band)
                values.append(observed / flat)
            position = xs + (offset - (len(bars) - 1) / 2) * width
            axis.bar(
                position,
                values,
                width * 0.92,
                color=FAMILY[family if family in FAMILY else family_of(arm)],
                edgecolor="white",
                label=label,
                zorder=3,
                **style,
            )
            for x, v in zip(position, values):
                axis.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
            drawn.append(arm)
        axis.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=2)
        axis.set_xticks(xs)
        axis.set_xticklabels([f"{int(b * 100)}%" for b in RIM_BANDS])
        axis.set_xlabel("edge band, as a fraction of the image", fontsize=9)
        axis.set_title(title, color=INK, fontsize=10, loc="left")
        axis.legend(frameon=False, fontsize=8, loc="lower right")
        tidy(axis)
    axes[0].set_ylabel("edge weight / flat map")
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    missing = [a for _f, _t, bars in RIM_ARMS for a, _l in bars if a not in drawn]
    print(f"  rim attention: {drawn}" + (f"  MISSING {missing}" if missing else ""))
    return out


def edge_ratio(cam: np.ndarray, band: float = 0.10) -> float:
    """Weight in the outer ``band`` of a map, over the share its area would get.

    1 is a map that ignores position; 2 puts twice its area's share on the edge.
    """
    cam = np.asarray(cam, dtype=float)
    height, width = cam.shape
    rows, cols = max(1, round(band * height)), max(1, round(band * width))
    mask = np.zeros(cam.shape, bool)
    mask[:rows], mask[-rows:], mask[:, :cols], mask[:, -cols:] = True, True, True, True
    total = cam.sum()
    return float(cam[mask].sum() / total / mask.mean()) if total > 0 else float("nan")


def draw_gradcam_evidence(
    attribution: Path, dataset: Path, episode: int, index: int, out: Path
) -> "Path | None":
    """The Grad-CAM picture that raised the rim question, with its edge band.

    The four fingertip maps of one frame, over the frame itself, with the outer
    tenth outlined and each sensor's edge ratio (:func:`edge_ratio`) in its
    title, so the claim "ACT weights the edge" is a number on the figure.
    """
    import pandas as pd
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    payload = json.loads(attribution.read_text())
    frames = payload["episodes"][str(episode)]["frames"]
    frame = next((f for f in frames if f.get("index") == index), None)
    if frame is None or "gradcam" not in frame:
        print(f"  gradcam evidence: no Grad-CAM at episode {episode} frame {index}")
        return None
    table = pd.concat(
        pd.read_parquet(p)
        for p in sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    )
    length = int(table[table["episode_index"] == episode].iloc[0]["length"])
    figure, axes = plt.subplots(1, 4, figsize=(11, 2.3))
    for axis, camera in zip(axes, TACTILE_CAMERAS):
        image = episode_frames(
            dataset, f"observation.images.{camera}", episode, [index / (length - 1)]
        )[0][1]
        cam = np.asarray(frame["gradcam"][camera], dtype=float)
        height, width = image.shape[:2]
        axis.imshow(image)
        axis.imshow(
            cam,
            cmap="jet",
            alpha=0.45,
            extent=(0, width, height, 0),
            interpolation="bilinear",
        )
        axis.add_patch(
            plt.Rectangle(
                (0.1 * width, 0.1 * height),
                0.8 * width,
                0.8 * height,
                fill=False,
                edgecolor="white",
                linestyle="--",
                linewidth=1.3,
            )
        )
        axis.set_title(
            f"{SENSOR_SHORT[camera]}: edge band x{edge_ratio(cam):.1f}", fontsize=8
        )
        axis.axis("off")
    figure.tight_layout(pad=0.3)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  gradcam evidence: episode {episode} frame {index} -> {out}")
    return out


def _resized(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def _letterboxed(image: np.ndarray, size: int) -> np.ndarray:
    """pi0.5's ``resize_with_pad``: keep the aspect ratio, pad the rest black."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    inner = _resized(image, round(height * scale), round(width * scale))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - inner.shape[0]) // 2, (size - inner.shape[1]) // 2
    canvas[top : top + inner.shape[0], left : left + inner.shape[1]] = inner
    return canvas


def draw_model_inputs(dataset: Path, episode: int, fraction: float, out: Path) -> Path:
    """What each model's image input looks like, from one moment of one recording.

    Rebuilt with the models' own layouts: ACT, Diffusion and flow matching read
    the five frames separately; pi0.5 letterboxes three of them to 224 x 224;
    DreamZero squashes all five into the cells of a 3 x 3 grid of one 224 x 224
    frame (spare cells black); FastWAM squashes the four fingertips into the
    quadrants of one image and sets it beside the overhead view, each 224 x 224.
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    names = ["central", *TACTILE_CAMERAS]
    frames = {
        n: episode_frames(dataset, f"observation.images.{n}", episode, [fraction])[0][1]
        for n in names
    }
    cell = 224 // 3
    tiled = np.zeros((224, 224, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        row, col = divmod(i, 3)
        tiled[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = _resized(
            frames[n], cell, cell
        )
    quad = np.zeros((224, 224, 3), dtype=np.uint8)
    for i, n in enumerate(TACTILE_CAMERAS):
        row, col = divmod(i, 2)
        quad[row * 112 : (row + 1) * 112, col * 112 : (col + 1) * 112] = _resized(
            frames[n], 112, 112
        )
    fastwam = np.concatenate([_resized(frames["central"], 224, 224), quad], axis=1)
    pi05 = np.concatenate(
        [
            _letterboxed(frames[n], 224)
            for n in ("central", "left_arm_left_gripper", "right_arm_left_gripper")
        ],
        axis=1,
    )

    figure = plt.figure(figsize=(11, 4.9))
    grid = figure.add_gridspec(2, 10, height_ratios=(1, 1.45), hspace=0.35)
    for i, n in enumerate(names):
        axis = figure.add_subplot(grid[0, 2 * i : 2 * i + 2])
        axis.imshow(frames[n])
        axis.set_title(
            "overhead" if n == "central" else SENSOR_SHORT[n], fontsize=8, color=INK
        )
        axis.axis("off")
    figure.text(
        0.5,
        0.93,
        "(a) the five cameras, as ACT, Diffusion and flow matching read them "
        "(each separately)",
        ha="center",
        fontsize=9,
    )
    panels = (
        (grid[1, 0:3], tiled, "(b) DreamZero: one tiled frame"),
        (grid[1, 3:7], fastwam, "(c) FastWAM: overhead | fingertip composite"),
        (grid[1, 7:10], pi05, "(d) pi0.5: overhead, LLG, RLG, padded"),
    )
    for spec, image, title in panels:
        axis = figure.add_subplot(spec)
        axis.imshow(image)
        axis.set_title(title, fontsize=9, color=INK)
        axis.axis("off")
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  model inputs: recording {episode} at {fraction:.2f} -> {out}")
    return out


#: Seconds after the last seen frame of each predicted frame. FastWAM predicts
#: one frame every four at 30 Hz; DreamZero one every 24, the first 0.8 s out.
FASTWAM_TIMES = tuple((i + 1) * 4 / 30 for i in range(8))
DREAMZERO_TIMES = tuple((i + 1) * 24 / 30 for i in range(6))


def draw_filmstrip_h2h(
    fastwam: dict,
    dreamzero: dict,
    out: Path,
    cameras: "tuple[str, ...]" = ("central", "left_arm_left_gripper"),
    fastwam_at: "tuple[float, ...]" = (4 / 15, 8 / 15, 0.8, 16 / 15),
    dreamzero_at: "tuple[float, ...]" = (0.8, 1.6, 2.4, 3.2, 4.0, 4.8),
) -> Path:
    """FastWAM and DreamZero predicting from the SAME last seen frame.

    Columns are one time axis: the seen frame, then seconds ahead. For each
    camera the rows are what happened, then each model's prediction and its
    difference from what happened (truth minus prediction, mid-grey is no
    error). The seen column shows each model's reconstruction of the frame it
    was given, so a blurred prediction can be told from a blurred autoencoder.
    A model's cell is blank where it predicts nothing. Each model is shown at
    its own resolution; the truth row takes FastWAM's frames up to its horizon
    and DreamZero's after.
    """
    from tool.eval_world_model import difference_image

    times = sorted({0.0, *fastwam_at, *dreamzero_at})

    def frame(parts: dict, grid: "tuple[float, ...]", t: float, kind: str):
        if t == 0.0:
            return parts["held"] if kind == "actual" else parts.get("held_recon")
        hits = [i for i, g in enumerate(grid) if abs(g - t) < 1e-3]
        return parts[kind][hits[0]] if hits else None

    rows = []
    for camera in cameras:
        fw, dz = fastwam[camera], dreamzero[camera]
        truth = [
            frame(fw, FASTWAM_TIMES, t, "actual")
            if t <= FASTWAM_TIMES[-1] + 1e-3
            else frame(dz, DREAMZERO_TIMES, t, "actual")
            for t in times
        ]
        label = "overhead" if camera == "central" else SENSOR_SHORT.get(camera, camera)
        rows.append((f"{label}\nactual", truth))
        for name, parts, grid in (
            ("FastWAM", fw, FASTWAM_TIMES),
            ("DreamZero", dz, DREAMZERO_TIMES),
        ):
            guess = [frame(parts, grid, t, "predicted") for t in times]
            own_truth = [frame(parts, grid, t, "actual") for t in times]
            diff = [
                None if g is None or a is None else difference_image(a, g)
                for g, a in zip(guess, own_truth)
            ]
            rows.append((f"{name}\npredicted", guess))
            rows.append((f"{name}\ndifference", diff))
    figure, axes = plt.subplots(
        len(rows), len(times), figsize=(1.05 * len(times) + 1.0, 0.95 * len(rows))
    )
    for r, (label, images) in enumerate(rows):
        for c, image in enumerate(images):
            axis = axes[r][c]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if image is not None:
                axis.imshow(image)
            if r == 0:
                axis.set_title(
                    "seen" if times[c] == 0 else f"+{times[c]:.2f} s", fontsize=7
                )
            if c == 0:
                axis.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    figure.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  head-to-head filmstrip -> {out}")
    return out


def patch_maps(path: Path) -> "dict[str, np.ndarray]":
    """Each fingertip's patch map, averaged over the analysed frames.

    Every frame's map is first scaled to sum to one, so a frame where the plan
    moved a lot does not outweigh the rest. A composite (FastWAM's fingertips) is
    cut back into its four sensors, in the order it was tiled.
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    payload = json.loads(path.read_text())
    sums: "dict[str, list[np.ndarray]]" = {}
    for episode in payload["episodes"].values():
        for frame in episode["frames"]:
            for camera, values in (frame.get("patches") or {}).items():
                grid = np.asarray(values, dtype=float)
                parts = {camera: grid}
                if camera == "tactile_quad":
                    rows, cols = grid.shape[0] // 2, grid.shape[1] // 2
                    parts = {
                        name: grid[
                            (i // 2) * rows : (i // 2 + 1) * rows,
                            (i % 2) * cols : (i % 2 + 1) * cols,
                        ]
                        for i, name in enumerate(TACTILE_CAMERAS)
                    }
                for name, part in parts.items():
                    total = part.sum()
                    if total > 0:
                        sums.setdefault(name, []).append(part / total)
    return {name: np.mean(maps, axis=0) for name, maps in sums.items()}


def patch_edge_ratio(maps: "dict[str, np.ndarray]", band: float = 0.2) -> float:
    """Share of a patch map in the outer ``band``, over that band's share of area.

    The band is in normalised coordinates (a cell counts if its centre lies in
    it), so a 10 x 10 grid and a 5 x 5 one both give the outer fifth 64 % of the
    area. 1 is no preference; the four sensors are averaged.
    """
    from tool.rim_attention import rim_mask

    ratios = []
    for grid in maps.values():
        mask = rim_mask(grid.shape, band)
        total = grid.sum()
        if total > 0 and mask.any():
            ratios.append(float(grid[mask].sum() / total / mask.mean()))
    return float(np.mean(ratios)) if ratios else float("nan")


def draw_patch_maps(entries: "list[tuple[str, Path]]", out: Path) -> "Path | None":
    """Where each model's plan comes from in the fingertip images.

    One column per model (and crop), one row per sensor: the mean patch map,
    darker where painting that cell over moved the plan more. Each map is scaled
    to its own peak, so compare where the dark cells are, not how dark.
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    found = [(label, patch_maps(path)) for label, path in entries if path.is_file()]
    found = [(label, maps) for label, maps in found if maps]
    if not found:
        print("  patch maps: none found")
        return None
    figure, axes = plt.subplots(
        4, len(found), figsize=(1.35 * len(found) + 0.8, 4.4), squeeze=False
    )
    for c, (label, maps) in enumerate(found):
        for r, camera in enumerate(TACTILE_CAMERAS):
            axis = axes[r][c]
            axis.set_xticks([])
            axis.set_yticks([])
            grid = maps.get(camera)
            if grid is not None:
                axis.imshow(grid / grid.max(), cmap="Greys", vmin=0, vmax=1)
            else:
                # A sensor this model does not read (pi0.5 reads two).
                axis.axis("off")
            if r == 0:
                axis.set_title(label, fontsize=7)
            if c == 0:
                axis.set_ylabel(SENSOR_SHORT[camera], fontsize=7)
    figure.tight_layout(pad=0.2)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  patch maps: {[label for label, _ in found]} -> {out}")
    return out


def draw_patch_overlays(
    entries: "list[tuple[str, Path]]",
    dataset: Path,
    episode: int,
    fraction: float,
    out: Path,
) -> "Path | None":
    """The patch maps of :func:`draw_patch_maps`, laid over the fingertip images.

    The background is each sensor's frame at ``fraction`` of the recording (the
    moment Figure ``model_inputs`` shows), in grey, so the colour is the map
    alone. A cell a cropped model never sees is left uncoloured. Each map is
    scaled to its own peak: compare where the bright cells are, not how bright.
    """
    from actoris_harena.policies.common.tactile import TACTILE_CAMERAS

    found = [(label, patch_maps(path)) for label, path in entries if path.is_file()]
    found = [(label, maps) for label, maps in found if maps]
    if not found:
        print("  patch overlays: none found")
        return None
    backgrounds = {
        camera: episode_frames(
            dataset, f"observation.images.{camera}", episode, [fraction]
        )[0][1]
        for camera in TACTILE_CAMERAS
    }
    figure, axes = plt.subplots(
        4, len(found), figsize=(1.75 * len(found) + 0.6, 5.6), squeeze=False
    )
    image = None
    for c, (label, maps) in enumerate(found):
        for r, camera in enumerate(TACTILE_CAMERAS):
            axis = axes[r][c]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            grid = maps.get(camera)
            if grid is None:
                # A sensor this model does not read (pi0.5 reads two).
                axis.axis("off")
            else:
                frame = backgrounds[camera]
                height, width = frame.shape[:2]
                axis.imshow(frame.mean(axis=2), cmap="gray", vmin=0, vmax=255)
                shown = np.where(grid > 0, grid / np.nanmax(grid), np.nan)
                # Opacity follows the value, so unimportant cells let the gel
                # show through and the important ones stand out on it.
                colours = plt.get_cmap("inferno")(np.nan_to_num(shown))
                colours[..., 3] = np.where(
                    np.isnan(shown), 0.0, 0.15 + 0.7 * np.nan_to_num(shown)
                )
                axis.imshow(
                    colours, interpolation="nearest", extent=(0, width, height, 0)
                )
                image = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
            if r == 0:
                axis.set_title(label, fontsize=7)
            if c == 0:
                axis.axis("on")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_ylabel(SENSOR_SHORT[camera], fontsize=7)
    if image is not None:
        bar = figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
        bar.set_label("plan moved (scaled to peak)", fontsize=7)
        bar.ax.tick_params(labelsize=6)
    figure.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"  patch overlays: {[label for label, _ in found]} -> {out}")
    return out


def draw_patch_rim(
    groups: "list[tuple[str, list[tuple[str, Path]]]]", out: Path
) -> "Path | None":
    """The edge measure from patch maps, per model and crop.

    ``groups`` is ``[(model, [(crop label, attribution.json), ...]), ...]``; a
    missing file is a gap, never a zero.
    """
    crops = []
    for _model, arms in groups:
        for crop, _path in arms:
            if crop not in crops:
                crops.append(crop)
    values = {
        (model, crop): patch_edge_ratio(patch_maps(path))
        for model, arms in groups
        for crop, path in arms
        if path.is_file()
    }
    if not values:
        print("  patch rim: none found")
        return None
    figure, axis = plt.subplots(figsize=(1.3 * len(groups) + 2.5, 2.8))
    width = 0.8 / len(crops)
    shades = ["#4a3aa7", "#e8743b", "#19a979", "#3fd0ff"]
    for k, crop in enumerate(crops):
        xs = [i + (k - (len(crops) - 1) / 2) * width for i in range(len(groups))]
        ys = [values.get((model, crop), np.nan) for model, _ in groups]
        axis.bar(xs, ys, width * 0.9, label=crop, color=shades[k % len(shades)])
    axis.axhline(1.0, color=MUTED, ls="--", lw=1)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([model for model, _ in groups], fontsize=8)
    axis.set_ylabel("edge weight / edge area", fontsize=8)
    axis.legend(
        fontsize=7,
        frameon=False,
        ncol=len(crops),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  patch rim: {len(values)} arms -> {out}")
    for key, value in values.items():
        print(f"    {key[0]:<12} {key[1]:<12} x{value:.2f}")
    return out


#: The patch-map runs behind the all-model patch figure, on held-out
#: demonstration 58 (10x10 grid per fingertip; FastWAM 5x5 per sensor of its
#: composite). Label, attribution.json.
PATCH_DECKS = (
    ("ACT", "outputs/analysis/2026-09-27/patches-act/attribution.json"),
    (
        "ACT\nfour-edge",
        "outputs/analysis/2026-09-27/patches-act_crop_edges/attribution.json",
    ),
    ("Diffusion", "outputs/analysis/2026-10-08/patches-diffusion/attribution.json"),
    ("pi0.5", "outputs/analysis/2026-10-08/patches-pi05/attribution.json"),
    (
        "pi0.5\nfour-edge",
        "outputs/analysis/2026-09-27/patches-pi05_crop_edges/attribution.json",
    ),
    (
        "Flow\nmatching",
        "outputs/analysis/2026-10-08/patches-flowmatch_resize/attribution.json",
    ),
    ("DreamZero", "outputs/analysis/2026-10-08/patches-dreamzero/attribution.json"),
    ("FastWAM", "outputs/analysis/2026-09-27/patches-fastwam/attribution.json"),
)


#: The input-contribution runs behind the all-model share figure, on held-out
#: demonstration 58: every 5th frame, 60 frames; images at their mean colour,
#: joints (and DreamZero's past commands) at their training mean.
STREAM_DECKS = (
    ("ACT", "outputs/analysis/2026-10-09/occ-act-mean/attribution.json"),
    ("Diffusion", "outputs/analysis/2026-10-09/occ-diffusion-mean/attribution.json"),
    (
        "Flow\nmatching",
        "outputs/analysis/2026-10-09/occ-flowmatch-mean/attribution.json",
    ),
    ("pi0.5", "outputs/analysis/2026-10-09/occ-pi05-mean/attribution.json"),
    ("DreamZero", "outputs/analysis/2026-10-09/occ-dreamzero-mean/attribution.json"),
    ("FastWAM", "outputs/analysis/2026-10-09/occ-fastwam-mean/attribution.json"),
)


def draw_stream_shares_all(
    entries: "list[tuple[str, Path]]", out: Path
) -> "Path | None":
    """What moves each model's plan: proprioception, past commands, overhead, touch.

    Stacked to 100 % because the shares are normalised within a model. Past
    commands appear only for DreamZero, the one model given them as input.
    """
    rows = []
    for label, path in entries:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        totals: "dict[str, list[float]]" = {}
        for episode in payload["episodes"].values():
            for frame in episode["frames"]:
                for name, effect in (
                    (frame.get("occlusion") or {}).get("streams", {}).items()
                ):
                    totals.setdefault(name, []).append(effect["share"])
        shares = {k: float(np.mean(v)) for k, v in totals.items()}
        rows.append(
            (
                label,
                [
                    shares.get("state", 0.0),
                    shares.get("past actions", 0.0),
                    shares.get("central", 0.0),
                    sum(
                        v
                        for k, v in shares.items()
                        if "gripper" in k or k == "tactile_quad"
                    ),
                ],
            )
        )
    if not rows:
        return None
    bands = (
        ("proprioception", "#4a3aa7"),
        ("past commands", "#9b8bd6"),
        ("overhead", "#8a8a8a"),
        ("fingertips", "#e8743b"),
    )
    figure, axis = plt.subplots(figsize=(1.1 * len(rows) + 2.2, 3.0))
    bottom = np.zeros(len(rows))
    for k, (name, colour) in enumerate(bands):
        values = np.array([r[1][k] for r in rows]) * 100
        if not values.any():
            continue
        axis.bar(
            [r[0] for r in rows], values, 0.6, bottom=bottom, color=colour, label=name
        )
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v >= 6:
                axis.annotate(
                    f"{v:.0f}%",
                    (i, b + v / 2),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
        bottom += values
    axis.set_ylim(0, 100)
    axis.set_ylabel("share of what moved the plan (%)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(
        fontsize=7,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
    )
    tidy(axis)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  stream shares, all models: {[r[0] for r in rows]} -> {out}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
