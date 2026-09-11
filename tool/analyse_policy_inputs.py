#!/usr/bin/env python3
"""What each input stream contributes to the actions a policy plans.

Separate from the inference pipeline: this loads a checkpoint of its own,
answers questions about it, and writes numbers and figures. It commands no arm
and opens no bus, so it can be run on the machine that holds the GPU while the
rig is switched off.

    # over recorded episodes -- ground truth actions, many frames
    venv/bin/python tool/analyse_policy_inputs.py \\
        --checkpoint outputs/policies/<run> \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile \\
        --episodes 0-5

    # over a real rollout, recorded with tool/run_policy.py --log-frames
    venv/bin/python tool/analyse_policy_inputs.py \\
        --checkpoint outputs/policies/<run> --run outputs/policy_runs/<stamp>

Results land in `outputs/analysis/<YYYY-MM-DD>/<policy>-<camera slug>/` unless
`--out` says otherwise -- see `common/analysis/paths.py` for why the day is part
of the path.

WHAT IT ASKS, and why more than one method. `occlusion` replaces a stream and
re-infers: it is behaviour, and it is the ground truth the others are scored
against. `ig` (integrated gradients) decomposes the plan with attributions that
sum to the change, so per-stream shares are parts of one whole. `gradcam` says
WHERE in a frame. `attention` reads ACT's decoder directly -- and is reported as
a deviation from what token count alone would give, because measured on this rig
the raw mass is within one per cent of uniform and means almost nothing.

Written to `--out`: `attribution.json` (every number, so a figure can be redrawn
without a GPU) and PNGs beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.analysis import attention as attn  # noqa: E402
from common.analysis import gradients as grads  # noqa: E402
from common.analysis import phases, report  # noqa: E402
from common.analysis.inference import Inference  # noqa: E402
from common.analysis.paths import analysis_dir, content_name  # noqa: E402
from common.analysis.perturb import (  # noqa: E402
    BASELINES,
    baseline_frame,
    occlusion,
    ranking,
)
from common.analysis.sources import DatasetSource, RunSource  # noqa: E402

METHODS = ("occlusion", "ig", "gradcam", "attention")
JOINT_NAMES = [
    "L_pan",
    "L_lift",
    "L_elbow",
    "L_wflex",
    "L_wroll",
    "L_grip",
    "R_pan",
    "R_lift",
    "R_elbow",
    "R_wflex",
    "R_wroll",
    "R_grip",
]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def analyse_frame(inference, state, images, args, alternative=None) -> dict:
    """Every requested method on one observation."""
    out: "dict[str, object]" = {}
    if "occlusion" in args.method:
        result = occlusion(
            inference,
            state,
            images,
            baseline=args.baseline,
            alternative=alternative,
            direction=args.direction,
        )
        out["occlusion"] = {
            "streams": {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in result["streams"].items()
            },
            "all": result["all"],
            "baseline": result["baseline"],
            "direction": result["direction"],
        }
    needs_batch = {"ig", "gradcam", "attention"} & set(args.method)
    if not needs_batch:
        return out

    batch = inference.batch(state, images)
    if "ig" in args.method:
        base_images = {c: baseline_frame(v, args.baseline) for c, v in images.items()}
        result = grads.integrated_gradients(
            inference,
            batch,
            inference.batch(state, base_images),
            steps=args.ig_steps,
            target=args.target,
        )
        totals = grads.per_stream(result["attributions"], inference)
        out["ig"] = {
            "totals": totals,
            "shares": grads.shares(totals),
            "completeness_error": result["completeness_error"],
            "steps": result["steps"],
        }
    # family(), not type: a ported `so101_act` checkpoint is ACT in every way
    # that matters here, and testing the config string silently dropped
    # attention from its deck while layout() still treated it as ACT.
    if "attention" in args.method and inference.family() == "act":
        spans, total, _ = inference.layout()
        weights = attn.cross_attention(inference, batch)
        if weights is not None:
            per_action = attn.by_stream(weights, spans)
            means = attn.summarise(per_action)
            out["attention"] = {
                "mean": means,
                "deviation": attn.deviation(means, spans, total),
                "per_action": {k: v.tolist() for k, v in per_action.items()},
                "uniform": attn.uniform_share(spans, total),
            }
    if "gradcam" in args.method:
        # A token model has no convolutional feature map to weight. Say so in the
        # payload rather than raising: a deck that is missing a method should
        # record why it is missing, so a slide cannot imply the method agreed.
        try:
            out["gradcam"] = {
                k: v.tolist() for k, v in grads.grad_cam(inference, batch).items()
            }
        except RuntimeError as problem:
            out["gradcam_unavailable"] = str(problem)
    return out


def camera_slug(inference, source) -> "str | None":
    """How the camera set is named in a directory, as a run directory names it.

    ``all`` when the policy reads every camera the source has -- otherwise the
    joined list, which is what a camera-ablation view is called. Naming the full
    five-camera set by listing it produces a 78-character directory that says no
    more than the word does, and stops an analysis directory matching the run
    directory it analysed.
    """
    cameras = list(inference.cameras)
    if not cameras:
        return None
    available = set(getattr(source, "cameras", None) or ())
    if available and set(cameras) == available:
        return "all"
    return "+".join(cameras)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="A trained policy directory"
    )
    parser.add_argument("--dataset", help="A LeRobotDataset root to analyse")
    parser.add_argument("--run", help="A run log from tool/run_policy.py --log-frames")
    parser.add_argument("--episodes", default="", help="e.g. 0-4,7 (default: all)")
    parser.add_argument(
        "--every",
        type=int,
        default=20,
        help="Analyse one frame in N of an episode (default 20)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Stop after this many frames per episode",
    )
    parser.add_argument(
        "--method",
        default="occlusion,ig,attention",
        help=f"Comma list of {', '.join(METHODS)}",
    )
    parser.add_argument(
        "--baseline",
        default="mean",
        choices=list(BASELINES),
        help="What a removed stream is replaced by (default mean)",
    )
    parser.add_argument(
        "--direction", default="leave_one_out", choices=("leave_one_out", "only_one_in")
    )
    parser.add_argument(
        "--target",
        default="norm",
        choices=list(grads.TARGETS),
        help="The scalar the gradients attribute (default: the whole plan)",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Integration steps (see common/analysis/gradients.py)",
    )
    parser.add_argument("--device", default=None, help="cpu/cuda (default auto)")
    parser.add_argument(
        "--task", default="", help="Language task, if the policy reads one"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: outputs/analysis/<today>/<name>)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="The <content> half of the default output directory, normally "
        "<policy>-<camera slug> (default: derived from the policy and cameras)",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Also run the model-randomisation check (Adebayo et al.): "
        "a method whose answer barely moves when the weights are "
        "randomised is measuring the input, not the policy",
    )
    parser.add_argument("--no-figures", action="store_true", help="Numbers only")
    args = parser.parse_args()

    args.method = [m.strip() for m in args.method.split(",") if m.strip()]
    unknown = [m for m in args.method if m not in METHODS]
    if unknown:
        raise SystemExit(f"❌ unknown method(s): {', '.join(unknown)}")
    if bool(args.dataset) == bool(args.run):
        raise SystemExit("❌ pass exactly one of --dataset or --run")

    inference = Inference(args.checkpoint, args.device, args.task)
    print(f"📦 {inference.describe()}")
    spans, total, problems = inference.layout()
    for problem in problems:
        print(f"⚠️  layout: {problem}")
    print(f"   conditioning: {total} entries over {len(spans)} stream(s)")

    source = (
        DatasetSource(args.dataset, args.episodes, args.every, inference.cameras)
        if args.dataset
        else RunSource(args.run, inference.cameras)
    )
    print(f"📁 {source.describe()}")

    # The convention: outputs/analysis/<YYYY-MM-DD>/<policy>-<camera slug>/.
    # The old default was `outputs/analysis/<basename of --checkpoint>`, which on
    # a LeRobot checkpoint is the literal word "pretrained_model" -- so every
    # analysis anyone forgot to name landed in one directory and overwrote the
    # last one.
    if args.out:
        out_dir = Path(args.out).expanduser()
    else:
        out_dir = analysis_dir(
            args.name or content_name(inference.type, camera_slug(inference, source))
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 {out_dir}")

    started = time.time()
    episodes: "dict[str, dict]" = {}
    for episode in source.episodes:
        frames: "list[dict]" = []
        for index, state, images, truth in source.observations(episode):
            if len(frames) >= args.max_frames:
                break
            record: "dict[str, Any]" = {"index": int(index)}
            record.update(analyse_frame(inference, state, images, args))
            if truth is not None:
                planned = inference.chunk(state, images)
                record["first_action_error"] = float(np.linalg.norm(planned[0] - truth))
            frames.append(record)
            print(
                f"  episode {episode} frame {index} "
                f"({len(frames)}) {time.time() - started:5.0f}s",
                end="\r",
            )
        if not frames:
            continue
        episodes[str(episode)] = {
            "frames": frames,
            "phases": phases.segment(source.actions(episode)),
        }
        print(f"  episode {episode}: {len(frames)} frame(s) analysed" + " " * 24)

    payload = {
        "checkpoint": inference.checkpoint,
        "policy": inference.type,
        "cameras": inference.cameras,
        "streams": [s.stream.name for s in spans],
        "methods": args.method,
        "baseline": args.baseline,
        "direction": args.direction,
        "target": args.target,
        "source": source.describe(),
        "episodes": episodes,
        "seconds": round(time.time() - started, 1),
    }
    if args.sanity:
        payload["sanity"] = sanity_check(args, inference, source)

    (out_dir / "attribution.json").write_text(json.dumps(_jsonable(payload), indent=1))
    print(f"\n📝 {out_dir / 'attribution.json'}")
    summarise(payload)
    if not args.no_figures:
        draw(payload, source, inference, args, out_dir)
    print(f"\n✓ {payload['seconds']:.0f}s")


def sanity_check(args, inference, source) -> dict:
    """Does the answer change when the weights do? See Adebayo et al. (2018)."""
    print("\n🧪 sanity check: randomising the weights ...")
    episode = source.episodes[0]
    first = next(iter(source.observations(episode)), None)
    if first is None:
        return {}
    _, state, images, _ = first
    before = occlusion(inference, state, images, baseline=args.baseline)
    before_shares = {k: v["share"] for k, v in before["streams"].items()}
    grads.randomise_weights(inference)
    after = occlusion(inference, state, images, baseline=args.baseline)
    after_shares = {k: v["share"] for k, v in after["streams"].items()}
    moved = max(abs(before_shares[k] - after_shares.get(k, 0.0)) for k in before_shares)
    agreement = grads.rank_agreement(before_shares, after_shares)
    total = sum(after_shares.values())
    print(f"   largest share moved by {moved:.3f}")
    if total <= 1e-9:
        # The strongest pass available, and it has no ranking to correlate: a
        # randomised policy plans the same chunk whatever it is shown, so no
        # stream affects it at all and every share is exactly zero. A rank
        # correlation of nan here would read as a broken check rather than as
        # the clean result it is.
        verdict = (
            "the randomised policy ignores every input entirely (all shares 0), "
            "so this attribution is a property of the TRAINED weights"
        )
    elif agreement != agreement or abs(agreement) < 0.5:
        verdict = "the ranking did not survive randomisation, as it should not"
    else:
        verdict = (
            "⚠️  the ranking SURVIVED randomisation: this attribution may be "
            "measuring the input rather than the policy (Adebayo et al., 2018)"
        )
    print(f"   {verdict}")
    print("   (the checkpoint in memory is now randomised — reload before reusing it)")
    return {
        "trained": before_shares,
        "randomised": after_shares,
        "max_share_shift": moved,
        "rank_agreement": None if agreement != agreement else agreement,
        "verdict": verdict,
    }


def summarise(payload: dict) -> None:
    """The table that answers the question, printed where it will be read."""
    for episode, data in payload["episodes"].items():
        frames = data["frames"]
        if not frames or "occlusion" not in frames[0]:
            continue
        streams = list(frames[0]["occlusion"]["streams"])
        print(f"\n📊 episode {episode}: mean share of the plan's movement, by stream")
        pooled = {
            name: float(
                np.mean([f["occlusion"]["streams"][name]["share"] for f in frames])
            )
            for name in streams
        }
        for name, share in sorted(pooled.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(round(share * 40))
            print(f"   {name:26} {share:6.1%} {bar}")
        if "ig" in frames[0]:
            errors = [f["ig"]["completeness_error"] for f in frames]
            print(
                f"   integrated gradients: completeness error "
                f"{np.mean(errors):.3f} mean, {np.max(errors):.3f} worst"
            )
            ig_shares = {
                n: float(np.mean([f["ig"]["shares"].get(n, 0.0) for f in frames]))
                for n in payload["cameras"]
            }
            occ_cams = {n: pooled[n] for n in payload["cameras"] if n in pooled}
            print(
                f"   IG agrees with occlusion at rank correlation "
                f"{grads.rank_agreement(ig_shares, occ_cams):+.3f}"
            )
        if "attention" in frames[0]:
            dev = {
                n: float(
                    np.mean([f["attention"]["deviation"].get(n, 0.0) for f in frames])
                )
                for n in payload["cameras"]
            }
            occ_cams = {n: pooled[n] for n in payload["cameras"] if n in pooled}
            print(
                f"   attention (deviation from uniform) agrees at "
                f"{grads.rank_agreement(dev, occ_cams):+.3f}"
            )


def draw(payload, source, inference, args, out_dir: Path) -> None:
    """Every figure the collected numbers support."""
    written: "list[str]" = []
    for episode, data in payload["episodes"].items():
        frames = data["frames"]
        if not frames:
            continue
        indices = [f["index"] for f in frames]
        if "occlusion" in frames[0]:
            streams = list(frames[0]["occlusion"]["streams"])
            series = {
                name: [f["occlusion"]["streams"][name]["share"] for f in frames]
                for name in streams
            }
            labels = (
                [data["phases"][min(i, len(data["phases"]) - 1)] for i in indices]
                if data["phases"]
                else None
            )
            written.append(
                report.contribution_over_time(
                    out_dir / f"episode{episode}_over_time.png",
                    indices,
                    series,
                    labels,
                    phases.spans_of,
                    title=f"What the plan depended on, episode {episode} "
                    f"({args.direction}, baseline={args.baseline})",
                )
            )
            last = frames[len(frames) // 2]
            written.append(
                report.stream_bars(
                    out_dir / f"episode{episode}_streams.png",
                    ranking({"streams": last["occlusion"]["streams"]}),
                    title=f"Effect of removing each stream (frame {last['index']})",
                    xlabel="mean per-step distance the plan moved",
                )
            )
            per_joint = {
                name: list(
                    np.mean(
                        [f["occlusion"]["streams"][name]["per_joint"] for f in frames],
                        axis=0,
                    )
                )
                for name in streams
            }
            written.append(
                report.stream_joint_heatmap(
                    out_dir / f"episode{episode}_joints.png",
                    per_joint,
                    JOINT_NAMES[: len(next(iter(per_joint.values())))],
                    title=f"Which stream drives which joint, episode {episode}",
                )
            )
        if "attention" in frames[0]:
            middle = frames[len(frames) // 2]
            uniform = middle["attention"]["uniform"]
            written.append(
                report.within_chunk(
                    out_dir / f"episode{episode}_within_chunk.png",
                    {
                        name: np.asarray(values) - uniform.get(name, 0.0)
                        for name, values in middle["attention"]["per_action"].items()
                        if name in payload["cameras"]
                    },
                    title=f"Where in the plan each camera was consulted (frame {middle['index']})",
                )
            )
        methods = {}
        if "occlusion" in frames[0]:
            methods["occlusion"] = {
                n: float(
                    np.mean([f["occlusion"]["streams"][n]["share"] for f in frames])
                )
                for n in payload["cameras"]
            }
        if "ig" in frames[0]:
            methods["integrated gradients"] = {
                n: float(np.mean([f["ig"]["shares"].get(n, 0.0) for f in frames]))
                for n in payload["cameras"]
            }
        if "attention" in frames[0]:
            methods["attention"] = {
                n: float(np.mean([f["attention"]["mean"].get(n, 0.0) for f in frames]))
                for n in payload["cameras"]
            }
        if len(methods) > 1:
            written.append(
                report.agreement(
                    out_dir / f"episode{episode}_methods.png",
                    methods,
                    title=f"Do the methods agree? Episode {episode}",
                )
            )
        if "gradcam" in frames[0]:
            middle = frames[len(frames) // 2]
            picked = next(
                (
                    obs
                    for obs in source.observations(int(episode))
                    if obs[0] == middle["index"]
                ),
                None,
            )
            if picked is not None:
                maps = {k: np.asarray(v) for k, v in middle["gradcam"].items()}
                written.append(
                    report.overlays(
                        out_dir / f"episode{episode}_gradcam.png",
                        picked[2],
                        maps,
                        title=f"Grad-CAM, episode {episode} frame {middle['index']}",
                    )
                )
    for path in written:
        if path:
            print(f"🖼️  {path}")


if __name__ == "__main__":
    main()
