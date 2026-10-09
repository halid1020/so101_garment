#!/usr/bin/env python
"""Figures and tables for the report's third round, beside ``report_figures.py``.

Three analyses, each drawn from the files its job wrote on the cluster:

* ``--recordings model=matrix.json ...`` -- how far apart the 65 recordings look
  to each model's encoder (``tool/compare_recordings.py``): a heatmap per model
  and a table of nearest-training-recording distances.
* ``--curves model=curve.json ...`` -- training and held-out action error at
  every saved step (``tool/score_curve.py``).
* ``--grasp condition=dir ...`` -- FastWAM's fingertip prediction over the test
  grasps under each gripper condition (``tool/eval_world_model.py --frames``),
  with ``--grasps grasps.json`` from ``tool/find_grasps.py``.

The palette, the label names and the axis styling are ``report_figures``'s, so
a model keeps its colour across every figure in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool.report_figures import (  # noqa: E402
    ALL_MODELS,
    FAMILY,
    INK,
    MUTED,
    SENSOR_SHORT,
    tidy,
)

LABELS = {key: label for key, label, _ in ALL_MODELS}


def pairs(items: "list[str]") -> "list[tuple[str, Path]]":
    """``model=path`` arguments, in the report's model order, missing ones dropped."""
    given = dict(item.split("=", 1) for item in items)
    order = [key for key, _, _ in ALL_MODELS] + sorted(
        k for k in given if k not in LABELS
    )
    return [
        (k, Path(given[k])) for k in order if k in given and Path(given[k]).is_file()
    ]


# -- 1. recordings ----------------------------------------------------------


def draw_recordings(entries: "list[tuple[str, Path]]", out: Path) -> "Path | None":
    """One normalised distance matrix per model, the test recordings boxed."""
    if not entries:
        return None
    cols = min(3, len(entries))
    rows = int(np.ceil(len(entries) / cols))
    figure, axes = plt.subplots(
        rows, cols, figsize=(2.5 * cols + 0.6, 2.5 * rows), squeeze=False
    )
    image = None
    for axis, (key, path) in zip(axes.flat, entries):
        payload = json.loads(path.read_text())
        matrix = np.array(payload["normalised"])
        held = [payload["episodes"].index(e) for e in payload["held"]]
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        if held:
            edge = min(held) - 0.5
            for line in (axis.axhline, axis.axvline):
                line(edge, color="#c2407e", lw=1.0)
        axis.set_title(LABELS.get(key, key), fontsize=9, color=INK)
        axis.set_xticks([0, 29, 57, len(matrix) - 1])
        axis.set_yticks([0, 29, 57, len(matrix) - 1])
        axis.set_xticklabels([1, 30, 58, len(matrix)], fontsize=7)
        axis.set_yticklabels([1, 30, 58, len(matrix)], fontsize=7)
    for axis in list(axes.flat)[len(entries) :]:
        axis.axis("off")
    figure.tight_layout()
    bar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    bar.set_label("distance (1 = largest pair)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  recordings: {[k for k, _ in entries]} -> {out}")
    return out


def write_recordings_table(
    entries: "list[tuple[str, Path]]", out: Path
) -> "Path | None":
    """Nearest-training-recording distance, training vs test, per model."""
    if not entries:
        return None
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"model & training & test & test / training \\",
        r"\midrule",
    ]
    for key, path in entries:
        s = json.loads(path.read_text())["summary"]
        train, test = s["nearest_train"], s["nearest_held"]
        lines.append(
            f"{LABELS.get(key, key)} & {train:.2f} & {test:.2f} & {test / train:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  recordings table -> {out}")
    return out


# -- 2. learning curves -----------------------------------------------------


def draw_curves(entries: "list[tuple[str, Path]]", out: Path) -> "Path | None":
    """Train (dashed) and test (solid) error over the first ten steps, per model."""
    if not entries:
        return None
    cols = min(3, len(entries))
    rows = int(np.ceil(len(entries) / cols))
    # One y-axis for every panel, so the models' errors can be compared
    # directly; the scale is set by the largest error any panel shows.
    figure, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 2.1 * rows), squeeze=False, sharey=True
    )
    top = max(
        point[half]["rmse_first10"]
        for _, path in entries
        for point in json.loads(path.read_text())["steps"].values()
        for half in ("train", "validation")
    )
    for axis, (key, path) in zip(axes.flat, entries):
        payload = json.loads(path.read_text())
        steps = sorted(payload["steps"], key=int)
        x = [int(s) / 1000 for s in steps]
        colour = FAMILY.get(key, INK)
        for half, style, name in (
            ("train", "--", "training"),
            ("validation", "-", "test"),
        ):
            y = [payload["steps"][s][half]["rmse_first10"] for s in steps]
            axis.plot(x, y, style, color=colour, lw=2, marker="o", ms=3, label=name)
        best = min(
            steps, key=lambda s: payload["steps"][s]["validation"]["rmse_first10"]
        )
        axis.axvline(int(best) / 1000, color=MUTED, lw=0.8, ls=":")
        axis.set_title(LABELS.get(key, key), fontsize=9, color=INK)
        axis.set_xlabel("training step (thousands)", fontsize=7)
        axis.tick_params(labelsize=7, labelleft=True)
        axis.set_ylim(0, top * 1.05)
        tidy(axis)
    for axis in list(axes.flat)[len(entries) :]:
        axis.axis("off")
    axes.flat[0].set_ylabel("RMSE, first 10 steps (deg)", fontsize=7)
    axes.flat[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  learning curves: {[k for k, _ in entries]} -> {out}")
    return out


def curve_summary(entries: "list[tuple[str, Path]]") -> None:
    """Where each model's test error bottoms out, against its final step."""
    for key, path in entries:
        steps = json.loads(path.read_text())["steps"]
        order = sorted(steps, key=int)
        test = {s: steps[s]["validation"]["rmse_first10"] for s in order}
        best = min(test, key=lambda s: test[s])
        print(
            f"    {key:<10} best test {test[best]:.2f} at {int(best)}, "
            f"final {test[order[-1]]:.2f} at {int(order[-1])}"
        )


# -- 3. FastWAM around grasps -----------------------------------------------


def grasp_arm_sensors(arm: str) -> "list[str]":
    return [f"{arm}_arm_left_gripper", f"{arm}_arm_right_gripper"]


def grasp_frames(directory: Path, grasps: "list[dict]") -> "dict[int, dict]":
    """``{loaded_index: filmstrip}`` for every grasp window the run kept."""
    from tool.eval_world_model import load_filmstrip

    out = {}
    for grasp in grasps:
        path = directory / f"filmstrip_{grasp['loaded_index']}.npz"
        if path.is_file():
            out[grasp["loaded_index"]] = load_filmstrip(path)
    return out


def _change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)


def grasp_numbers(
    conditions: "dict[str, dict[int, dict]]", grasps: "list[dict]", reference: str
) -> "dict[str, dict[str, float]]":
    """Per condition, over the grasping arm's two fingertips and every step.

    ``error`` is the mean absolute pixel error against what happened, ``moved``
    how far the prediction differs from the ``reference`` condition's, and
    ``truth_moved`` how far the real gel changed from the seen frame -- the
    scale a condition's effect is to be read against.
    """
    out: "dict[str, dict[str, float]]" = {}
    for name, strips in conditions.items():
        errors, moved, truth_moved = [], [], []
        for grasp in grasps:
            index = grasp["loaded_index"]
            if index not in strips or index not in conditions[reference]:
                continue
            for sensor in grasp_arm_sensors(grasp["arm"]):
                parts = strips[index].get(sensor)
                ref = conditions[reference][index].get(sensor)
                if parts is None or ref is None:
                    continue
                for t, guess in enumerate(parts["predicted"]):
                    errors.append(_change(guess, parts["actual"][t]))
                    moved.append(_change(guess, ref["predicted"][t]))
                    truth_moved.append(_change(parts["actual"][t], parts["held"]))
        out[name] = {
            "error": float(np.mean(errors)) if errors else float("nan"),
            "moved": float(np.mean(moved)) if moved else float("nan"),
            "truth_moved": float(np.mean(truth_moved)) if truth_moved else float("nan"),
            "windows": len(
                {g["loaded_index"] for g in grasps if g["loaded_index"] in strips}
            ),
        }
    return out


def write_grasp_table(numbers: "dict[str, dict[str, float]]", out: Path) -> Path:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"condition & error & change from recorded & repeat-last error \\",
        r"\midrule",
    ]
    for name, n in numbers.items():
        lines.append(
            f"{name} & {100 * n['error']:.2f} & {100 * n['moved']:.2f} & "
            f"{100 * n['truth_moved']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  grasp table -> {out}")
    return out


def draw_grasp_strip(
    conditions: "dict[str, dict[int, dict]]",
    grasp: dict,
    reference: str,
    out: Path,
    frame_hz: float = 30.0,
    stride: int = 4,
) -> "Path | None":
    """One grasp, one fingertip: truth, each condition, and the difference.

    The difference row is the second condition minus the reference (mid-grey is
    none), so a condition that changes nothing draws a flat grey row.
    """
    from tool.eval_world_model import difference_image

    sensor = grasp_arm_sensors(grasp["arm"])[0]
    index = grasp["loaded_index"]
    names = list(conditions)
    if any(index not in conditions[n] for n in names):
        return None
    ref = conditions[reference][index][sensor]
    other = next(n for n in names if n != reference)
    rows = [("actual", [ref["held"], *ref["actual"]])]
    for name in names:
        parts = conditions[name][index][sensor]
        rows.append((name, [parts.get("held_recon"), *parts["predicted"]]))
    rows.append(
        (
            f"{other}\nminus {reference}",
            [None]
            + [
                difference_image(r, o)
                for r, o in zip(
                    ref["predicted"], conditions[other][index][sensor]["predicted"]
                )
            ],
        )
    )
    steps = len(rows[0][1])
    figure, axes = plt.subplots(
        len(rows), steps, figsize=(0.95 * steps + 1.0, 0.95 * len(rows)), squeeze=False
    )
    onset = (grasp["onset"] - grasp["start"]) / frame_hz
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
                t = c * stride / frame_hz
                axis.set_title(
                    "seen" if c == 0 else f"+{t:.2f} s", fontsize=7,
                    color="#c2407e" if c and abs(t - onset) < stride / frame_hz / 2 else INK,
                )  # fmt: skip
            if c == 0:
                axis.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    short = SENSOR_SHORT.get(sensor, sensor)
    figure.suptitle(
        f"episode {grasp['episode']}, {short}: the gripper is commanded closed at "
        f"+{onset:.2f} s",
        fontsize=8,
        y=1.0,
    )
    figure.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1, rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  grasp filmstrip -> {out}")
    return out


def draw_sensor_shift(
    dataset: Path,
    out: Path,
    camera: str = "left_arm_left_gripper",
    before: "tuple[int, int]" = (30, 42),
    after: "tuple[int, int]" = (42, 54),
    frames: int = 3,
) -> Path:
    """A fingertip's untouched view before and after a step change in the session.

    Each image is the mean of the first ``frames`` frames of each episode in the
    range (the arms at rest, the gel untouched), so what differs is the sensor,
    not the contact. Episode ranges are zero-based and half-open.
    """
    from actoris_harena.analysis.sources import DatasetSource

    source = DatasetSource(str(dataset), "", 1)
    key = f"observation.images.{camera}"

    def rest(lo: int, hi: int) -> np.ndarray:
        images = [
            np.asarray(source.dataset[row][key]).transpose(1, 2, 0)
            for episode in range(lo, hi)
            for row in source.rows(episode)[:frames]
        ]
        return np.mean(images, axis=0)

    first, second = rest(*before), rest(*after)
    difference = np.clip(0.5 + 3.0 * (second - first), 0.0, 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
    titles = (
        f"demonstrations {before[0] + 1}-{before[1]}",
        f"demonstrations {after[0] + 1}-{after[1]}",
        "difference (x3, grey = none)",
    )
    for axis, image, title in zip(axes, (first, second, difference), titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=8, color=INK)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  sensor shift: {camera} -> {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", nargs="*", default=[])
    parser.add_argument("--curves", nargs="*", default=[])
    parser.add_argument(
        "--grasp", nargs="*", default=[], help="condition=dir, reference first"
    )
    parser.add_argument(
        "--grasps", default=None, help="grasps.json from find_grasps.py"
    )
    parser.add_argument(
        "--grasp-example", type=int, default=0, help="index into grasps.json"
    )
    parser.add_argument("--tag", default="", help="suffix for the grasp outputs")
    parser.add_argument("--dataset", default=None, help="for the sensor-shift figure")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recordings = pairs(args.recordings)
    draw_recordings(recordings, out / "recordings.png")
    write_recordings_table(recordings, out / "recordings_table.tex")
    if args.dataset:
        draw_sensor_shift(Path(args.dataset).expanduser(), out / "sensor_shift.png")
    curves = pairs(args.curves)
    if draw_curves(curves, out / "curves.png"):
        curve_summary(curves)
    if args.grasp and args.grasps:
        grasps = json.loads(Path(args.grasps).read_text())
        conditions = {
            name: grasp_frames(Path(path), grasps)
            for name, path in (item.split("=", 1) for item in args.grasp)
        }
        reference = next(iter(conditions))
        numbers = grasp_numbers(conditions, grasps, reference)
        for name, n in numbers.items():
            print(f"    {name:<14} {n}")
        write_grasp_table(numbers, out / f"grasp_table{args.tag}.tex")
        draw_grasp_strip(
            conditions,
            grasps[args.grasp_example],
            reference,
            out / f"grasp_filmstrip{args.tag}.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
