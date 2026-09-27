#!/usr/bin/env python3
"""How far a policy's planned actions are from the ones actually recorded.

Training loss does not answer this. Each policy minimises a different objective
-- ACT a VAE-regularised L1, diffusion a denoising MSE, pi0.5 a flow-matching
velocity error -- so the losses are not comparable to each other, and a run
trained on cropped inputs is fitting different data from its baseline. What IS
comparable across all of them is the distance between the chunk a policy plans
and the chunk the operator actually performed.

    venv/bin/python tool/eval_action_mse.py \\
        --checkpoint <run>/checkpoints/last/pretrained_model \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile

WHAT IS REPORTED, and why not one number. Error grows along a chunk -- a policy
that is excellent one step ahead and useless thirty steps ahead is a different
animal from one that is mediocre throughout -- so the horizon axis is kept. The
gripper channels are reported apart from the arm joints because they span a few
tenths of open fraction while the joints span radians, so a mean over all twelve
dimensions is dominated by the joints and a grasp is won or lost in the part it
buries.

TRAIN AND VALIDATION. `--split` reproduces LeRobot's own rule exactly rather
than inventing one: `make_train_eval_datasets` holds out **the last
ceil(n * eval_split) episodes OF EACH TASK**. Reproducing it matters more than
it sounds -- a "validation" set that is not the one the trainer held out is
worse than no validation set at all, because it looks like evidence. The split
is read from the checkpoint's own `train_config.json` by default, and where the
run held nothing out (`eval_split: 0.0`, which is every checkpoint trained here
so far) the tool says so, in its output and in the JSON, and reports one number
labelled training data.

A STOCHASTIC POLICY NEEDS ITS FLOOR. Diffusion and pi0.5 plan differently twice
from one observation. Every plan here goes through `analysis.diffusion.plan`, so
the numbers repeat -- but the sampler's own spread is measured and reported
beside the result, because an MSE difference smaller than it is not a finding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from actoris_harena.action_layout import gripper_columns  # noqa: E402
from actoris_harena.analysis.diffusion import plan, sampler_spread  # noqa: E402
from actoris_harena.analysis.inference import Inference  # noqa: E402
from actoris_harena.analysis.paths import analysis_dir, content_name  # noqa: E402
from actoris_harena.analysis.sources import DatasetSource  # noqa: E402

import common  # noqa: E402,F401  -- declares this rig to actoris_harena

# One definition of how a camera set is named in a directory. It belongs beside
# content_name in actoris_harena.analysis.paths and should move there when the
# package is next open for editing; it is imported rather than copied so that
# an analysis directory and the run directory it describes cannot drift apart.
from tool.analyse_policy_inputs import camera_slug, task_prompt  # noqa: E402


def held_out_episodes(
    episode_tasks: "list", eval_split: float, episodes: "list[int] | None" = None
) -> "tuple[list[int], list[int]]":
    """LeRobot's own split, reproduced: ``(train, validation)``.

    `lerobot/datasets/factory.py:make_train_eval_datasets` takes the LAST
    ``ceil(n * eval_split)`` episodes **of each task**, grouped by the episode's
    first task string. Per task, not overall -- on a single-task dataset the two
    agree, and on a mixed one they do not, which is exactly when a hand-rolled
    split would quietly disagree with the run it claims to describe.
    """
    base = list(range(len(episode_tasks))) if episodes is None else list(episodes)
    if not eval_split:
        return base, []

    by_task: "dict[str, list[int]]" = {}
    for index in base:
        tasks = episode_tasks[index]
        key = tasks[0] if len(tasks) else ""
        by_task.setdefault(key, []).append(index)

    train: "list[int]" = []
    validation: "list[int]" = []
    for group in by_task.values():
        n_eval = math.ceil(len(group) * eval_split)
        train.extend(group[: len(group) - n_eval])
        validation.extend(group[len(group) - n_eval :])
    return sorted(train), sorted(validation)


def split_from_checkpoint(checkpoint: str) -> float:
    """The ``eval_split`` the run was trained with, or 0.0 if it held nothing out."""
    config = Path(checkpoint) / "train_config.json"
    try:
        blob = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0.0
    return float((blob.get("dataset") or {}).get("eval_split") or 0.0)


def chunk_errors(planned: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Squared error per horizon step per dimension, over the overlap.

    The two are trimmed to the shorter: a chunk that runs past the end of its
    episode has no ground truth to be wrong about, and padding it with the last
    action would credit the policy for holding still.
    """
    steps = min(len(planned), len(truth))
    if steps == 0:
        return np.zeros((0, planned.shape[-1]), dtype=np.float64)
    return (planned[:steps].astype(np.float64) - truth[:steps].astype(np.float64)) ** 2


def summarise(errors: "list[np.ndarray]", action_dim: int) -> "dict":
    """Fold per-frame squared errors into the numbers that go on a slide."""
    if not errors:
        return {"frames": 0}
    horizon = max(len(e) for e in errors)
    total = np.zeros((horizon, action_dim), dtype=np.float64)
    counts = np.zeros((horizon, 1), dtype=np.float64)
    for e in errors:
        total[: len(e)] += e
        counts[: len(e)] += 1
    per_step_dim = total / np.maximum(counts, 1.0)

    grippers = [c for c in gripper_columns() if c < action_dim]
    joints = [c for c in range(action_dim) if c not in grippers]
    return {
        "frames": len(errors),
        "horizon": int(horizon),
        "mse": float(per_step_dim.mean()),
        "rmse": float(np.sqrt(per_step_dim.mean())),
        "mse_per_step": [float(v) for v in per_step_dim.mean(axis=1)],
        "mse_per_dim": [float(v) for v in per_step_dim.mean(axis=0)],
        "mse_joints": float(per_step_dim[:, joints].mean()) if joints else None,
        "mse_grippers": float(per_step_dim[:, grippers].mean()) if grippers else None,
        "first_step_mse": float(per_step_dim[0].mean()),
        "last_step_mse": float(per_step_dim[-1].mean()),
    }


def evaluate(inference, source, episodes: "list[int]", seed: int) -> "dict":
    """Plan at every sampled frame of these episodes and score against the record."""
    errors: "list[np.ndarray]" = []
    for episode in episodes:
        truth = source.actions(episode)
        rows = source.rows(episode)
        lo = rows[0] if rows else 0
        for index, state, images, _action in source.observations(episode):
            planned = plan(inference, inference.batch(state, images), seed)
            start = index - lo
            errors.append(chunk_errors(planned, truth[start:]))
    return summarise(errors, inference.action_dim)


def needs_window(policy) -> bool:
    """A world model that conditions on a short VIDEO, not on one frame.

    DreamZero reads the frames named by ``observation_delta_indices`` -- a past
    chunk of video -- and refuses a single frame outright. Repeating the one
    frame into that window would hand it a frozen past, a state it never saw in
    training, so these are loaded with their real window instead.
    """
    indices = getattr(policy.config, "observation_delta_indices", None) or []
    return hasattr(policy, "predict_future_frames") and len(indices) > 1


def future_truth(item: dict, action_delta_indices: "list[int]") -> np.ndarray:
    """The recorded actions from time zero on, cut where the episode ended.

    The window also holds the PAST actions (negative offsets) the model is
    conditioned on; those are context, not something it predicts. Past the
    end of an episode LeRobot pads by repetition, and scoring against padding
    would credit the policy for holding still -- the same rule as
    :func:`chunk_errors` applies to the single-frame path.
    """
    start = list(action_delta_indices).index(0)
    actions = np.asarray(item["action"])[start:]
    pad = item.get("action_is_pad")
    if pad is not None:
        pad = np.asarray(pad)[start:]
        if pad.any():
            actions = actions[: int(np.argmax(pad))]
    return actions


def evaluate_windowed(
    inference, dataset: str, episodes: "list[int]", every: int, seed: int
) -> "dict":
    """As :func:`evaluate`, for a model that needs its video window."""
    from tool.eval_world_model import load_dataset, make_batch

    data = load_dataset(inference.policy, dataset, episodes)
    deltas = inference.policy.config.action_delta_indices
    errors: "list[np.ndarray]" = []
    for index in range(0, data.num_frames, every):
        item = data[index]
        batch = inference.to_device(inference.pre(make_batch(item, inference.device)))
        planned = plan(inference, batch, seed)
        errors.append(chunk_errors(planned, future_truth(item, deltas)))
    return summarise(errors, inference.action_dim)


def report(name: str, block: "dict") -> None:
    if not block.get("frames"):
        print(f"  {name:12s} no frames")
        return
    print(
        f"  {name:12s} RMSE {block['rmse']:.4f}   MSE {block['mse']:.5f}"
        f"   joints {block['mse_joints']:.5f}   grippers {block['mse_grippers']:.5f}"
        f"   step 1 {block['first_step_mse']:.5f} -> step {block['horizon']} "
        f"{block['last_step_mse']:.5f}   ({block['frames']} frames)"
    )


def compare(paths: "list[str]") -> "tuple[list[str], list[list[str]]]":
    """Several runs' numbers in one table, one column per run.

    Rows are the ones that survive being compared: RMSE overall, joints and
    grippers apart, and the first and last step of the chunk so a reader can see
    whether a policy degrades along its horizon. Training LOSS is deliberately
    absent -- the arms minimise different objectives on different inputs, so
    putting it here would invite exactly the comparison it cannot support.
    """
    blobs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]

    def cell(blob: dict, section: str, key: str) -> str:
        value = (blob.get(section) or {}).get(key)
        return "-" if value is None else f"{value:.4f}"

    names = [b.get("name") or Path(p).parent.name for b, p in zip(blobs, paths)]
    rows = [["policy", *[b["policy"] for b in blobs]]]
    for label, section, key in (
        ("RMSE (train)", "train", "rmse"),
        ("joints", "train", "mse_joints"),
        ("grippers", "train", "mse_grippers"),
        ("step 1", "train", "first_step_mse"),
        ("last step", "train", "last_step_mse"),
        ("RMSE (val)", "validation", "rmse"),
    ):
        rows.append([label, *[cell(b, section, key) for b in blobs]])
    rows.append(
        [
            "sampler floor",
            *[
                "-" if b.get("sampler_spread") is None else f"{b['sampler_spread']:.4f}"
                for b in blobs
            ],
        ]
    )
    rows.append(
        [
            "held out?",
            *["no" if not b["episodes"]["validation"] else "yes" for b in blobs],
        ]
    )
    return ["", *names], rows


def render(headers: "list[str]", rows: "list[list[str]]") -> str:
    widths = [
        max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))
    ]

    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Several action_mse.json files, tabulated side by side instead of "
        "scoring a checkpoint",
    )
    parser.add_argument("--checkpoint", help="A trained policy directory")
    parser.add_argument("--dataset", help="The LeRobotDataset root")
    parser.add_argument("--every", type=int, default=20, help="Sample 1 frame in N")
    parser.add_argument("--episodes", default="", help="e.g. 0-9; default all")
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="eval_split to reproduce (default: the checkpoint's own; 0.0 means "
        "the run held nothing out and there is no validation set to report)",
    )
    parser.add_argument(
        "--task", default="", help="Language task, if the policy reads one"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Pins a stochastic sampler")
    parser.add_argument("--name", default=None, help="Output directory name")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.compare:
        print(render(*compare(args.compare)))
        return
    if not (args.checkpoint and args.dataset):
        raise SystemExit("❌ pass --checkpoint and --dataset, or --compare <json>...")

    from tool.eval_sim_policy import build_batch, load_policy

    inference = Inference(
        args.checkpoint,
        args.device,
        args.task,
        load_policy=load_policy,
        build_batch=build_batch,
    )
    print(f"📦 {inference.describe()}")

    source = DatasetSource(args.dataset, args.episodes, args.every, inference.cameras)
    print(f"📁 {source.describe()}")

    split = split_from_checkpoint(args.checkpoint) if args.split is None else args.split
    tasks = source.dataset.meta.episodes["tasks"]
    inference.task = task_prompt(args.task, tasks)
    if inference.task != args.task:
        print(f"💬 prompting with the dataset's task: {inference.task!r}")
    train, validation = held_out_episodes(tasks, split, source.episodes)

    if not validation:
        print(
            "⚠️  this checkpoint held nothing out (eval_split = 0.0), so every "
            "number below is measured on data the policy TRAINED on. It says "
            "how well the run fitted what it saw, not how it generalises."
        )

    out: "dict" = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "policy": inference.type,
        "dataset": str(Path(args.dataset).name),
        "eval_split": split,
        "seed": args.seed,
        "episodes": {"train": train, "validation": validation},
    }

    print()
    windowed = needs_window(inference.policy)
    if windowed:
        print(
            "🎞️  scoring with each observation's video window, as this model reads it"
        )

    def score(episodes: "list[int]") -> "dict":
        if windowed:
            return evaluate_windowed(
                inference, args.dataset, episodes, args.every, args.seed
            )
        return evaluate(inference, source, episodes, args.seed)

    out["train"] = score(train)
    report("train", out["train"])
    if validation:
        out["validation"] = score(validation)
        report("validation", out["validation"])

    # The floor. A stochastic policy's own wobble bounds what a difference in
    # the numbers above can mean, so it is measured rather than assumed small.
    if inference.stochastic():
        first = next(iter(source.observations(source.episodes[0])))
        spread = sampler_spread(inference, inference.batch(first[1], first[2]))
        out["sampler_spread"] = float(spread)
        print(
            f"\n  sampler spread {spread:.5f} — this policy plans differently "
            "twice from one observation. A difference smaller than this is the "
            "sampler, not the policy."
        )

    name = args.name or content_name(inference.type, camera_slug(inference, source))
    out_dir = Path(args.out).expanduser() if args.out else analysis_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_mse.json").write_text(json.dumps(out, indent=2))
    print(f"\n📝 {out_dir / 'action_mse.json'}")


if __name__ == "__main__":
    main()
