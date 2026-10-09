#!/usr/bin/env python
"""How far apart the recordings look, to one policy's vision encoder.

Every frame of every episode goes through the checkpoint's own preprocessor and
vision encoder (``actoris_harena.analysis.features``); each pair of episodes is
then matched frame for frame in order and scored by the mean cosine distance
over the matched pairs (``actoris_harena.analysis.recordings``). The result is a
K x K matrix, normalised to 0..1, and a summary of how far the held-out episodes
sit from the training ones compared with how far the training ones sit from
each other.

Features are cached per episode under the output directory, so an interrupted
job resumes where it stopped and a second matrix (another --every, another
split) costs no forward passes.

    python tool/compare_recordings.py --checkpoint <run>/checkpoints/last/pretrained_model \\
        --dataset <root> --held 58-64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actoris_harena.analysis import recordings as rec  # noqa: E402
from actoris_harena.analysis.features import encoder_kind, frame_features  # noqa: E402
from actoris_harena.analysis.inference import Inference  # noqa: E402
from actoris_harena.analysis.paths import analysis_dir, content_name  # noqa: E402
from actoris_harena.analysis.sources import DatasetSource  # noqa: E402

import common  # noqa: E402,F401  (configures the shared pipeline for this rig)


def episode_features(inference, source, episode: int) -> np.ndarray:
    """``[frames, D]`` for one episode, one forward pass per kept frame."""
    rows = []
    for _, state, images, _ in source.observations(episode):
        batch = inference.batch(state, images)
        rows.append(frame_features(inference.policy, batch)[0].numpy())
    return np.stack(rows).astype(np.float32)


def cached_features(inference, source, cache: Path) -> "list[np.ndarray]":
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for episode in source.episodes:
        path = cache / f"episode_{episode:03d}.npy"
        if not path.exists():
            started = time.time()
            np.save(path, episode_features(inference, source, episode))
            print(f"  episode {episode}: {time.time() - started:.0f} s", flush=True)
        out.append(np.load(path))
    return out


def image_keys_of(checkpoint: str) -> "list[str]":
    """The checkpoint's camera keys, in its own order, from config.json alone."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return [
        key
        for key, feature in (config.get("input_features") or {}).items()
        if (feature or {}).get("type") == "VISUAL"
    ]


def camera_group(
    features: "list[np.ndarray]", kind: str, image_keys: "list[str]", group: str
) -> "list[np.ndarray]":
    """Each episode's features restricted to one group of cameras."""
    from actoris_harena.analysis.features import OVERHEAD_NAMES, feature_groups

    groups = feature_groups(kind, image_keys, int(features[0].shape[1]))
    wanted = [
        name for name in groups if (name in OVERHEAD_NAMES) == (group == "overhead")
    ]
    if not wanted:
        raise SystemExit(f"❌ this model reads no {group} camera ({', '.join(groups)})")
    columns = sorted(i for name in wanted for i in groups[name])
    print(f"   {group}: {', '.join(wanted)} ({len(columns)} features)")
    return [f[:, columns] for f in features]


def from_cache(args) -> int:
    """A camera group's matrix from features already cached by a full run."""
    out = Path(args.out).expanduser() / "recordings"
    full = json.loads((out / f"matrix_every{args.every}.json").read_text())
    cache = out / f"features_every{args.every}"
    episodes = full["episodes"]
    features = [np.load(cache / f"episode_{e:03d}.npy") for e in episodes]
    features = camera_group(
        features, full["encoder"], image_keys_of(full["checkpoint"]), args.cameras
    )
    raw = rec.distance_matrix(features)
    held = [i for i, e in enumerate(episodes) if e in set(full["held"])]
    payload = {
        **{
            k: full[k] for k in ("checkpoint", "policy", "encoder", "every", "episodes")
        },
        "cameras": args.cameras,
        "feature_dim": int(features[0].shape[1]),
        "held": full["held"],
        "raw": raw.round(6).tolist(),
        "summary": rec.split_summary(raw, held),
    }
    path = out / f"matrix_every{args.every}_{args.cameras}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=1))
    print(f"💾 {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--episodes", default="", help="e.g. 0-64; default all")
    parser.add_argument("--held", default="58-64", help="the held-out episodes")
    parser.add_argument("--every", type=int, default=1, help="keep 1 frame in N")
    parser.add_argument("--task", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None, help="analysis directory name")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--cameras",
        default="all",
        choices=("all", "overhead", "fingertips"),
        help="overhead or fingertips: one group's matrix from the features a full "
        "run cached under --out (no model is loaded)",
    )
    args = parser.parse_args()
    if args.cameras != "all":
        if not args.out:
            raise SystemExit("❌ --cameras needs --out, the full run's directory")
        return from_cache(args)

    if not (args.checkpoint and args.dataset):
        raise SystemExit("❌ a full run needs --checkpoint and --dataset")
    from tool.analyse_policy_inputs import camera_slug, task_prompt
    from tool.eval_sim_policy import build_batch, load_policy
    from tool.eval_world_model import parse_range

    inference = Inference(
        args.checkpoint,
        args.device,
        args.task,
        load_policy=load_policy,
        build_batch=build_batch,
    )
    source = DatasetSource(args.dataset, args.episodes, args.every, inference.cameras)
    inference.task = task_prompt(args.task, source.dataset.meta.episodes["tasks"])
    kind = encoder_kind(inference.policy)
    print(f"📦 {inference.describe()} -- encoder: {kind}")
    print(f"📁 {source.describe()}, 1 frame in {args.every}")

    out = (
        Path(args.out).expanduser()
        if args.out
        else analysis_dir(
            args.name or content_name(inference.type, camera_slug(inference, source))
        )
    )
    out = out / "recordings"
    features = cached_features(inference, source, out / f"features_every{args.every}")

    started = time.time()
    raw = rec.distance_matrix(features)
    print(f"🧮 {len(features)}x{len(features)} matrix in {time.time() - started:.0f} s")
    held_ids = set(parse_range(args.held))
    held = [i for i, e in enumerate(source.episodes) if e in held_ids]
    normalised = rec.normalise(raw)
    payload = {
        "checkpoint": inference.checkpoint,
        "policy": inference.type,
        "encoder": kind,
        "feature_dim": int(features[0].shape[1]),
        "every": args.every,
        "episodes": list(source.episodes),
        "frames": [int(len(f)) for f in features],
        "held": [source.episodes[i] for i in held],
        "raw": raw.round(6).tolist(),
        "normalised": normalised.round(6).tolist(),
        # On the raw cosine distances: already one scale (0 to 2) for every
        # model, so models can be compared directly.
        "summary": rec.split_summary(raw, held),
        "summary_normalised": rec.split_summary(normalised, held),
    }
    path = out / f"matrix_every{args.every}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=1))
    print(f"💾 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
