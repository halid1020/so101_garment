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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", default="", help="e.g. 0-64; default all")
    parser.add_argument("--held", default="58-64", help="the held-out episodes")
    parser.add_argument("--every", type=int, default=1, help="keep 1 frame in N")
    parser.add_argument("--task", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None, help="analysis directory name")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

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
        "summary": rec.split_summary(normalised, held),
    }
    path = out / f"matrix_every{args.every}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=1))
    print(f"💾 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
