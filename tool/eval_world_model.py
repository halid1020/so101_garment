#!/usr/bin/env python
"""How accurately does the world model predict what happens next?

The question DreamZero exists to answer, and the one the paper never measures
directly -- it reports task progress, from which prediction quality can only be
inferred. Here the ground-truth future is on disk, so predicted frames are
compared against what actually happened.

Two things this insists on, because both change the conclusion:

* **Per camera.** On this rig four of five views are tactile and behave nothing
  like the overhead one -- a gel image is nearly static until contact, then
  changes fast. A single averaged number hides exactly the moment worth
  predicting.
* **Against the held-last-frame baseline.** "Nothing changes" is a strong
  predictor of a static scene. A model reported without that bar looks good for
  the wrong reason, and the honest headline is whether it BEATS it.

    venv/bin/python tool/eval_world_model.py \\
        --checkpoint outputs/policies/<run>/pretrained_model \\
        --dataset <collection>/<dataset> --episodes 0-9

Writes ``prediction.json`` -- every number, so a figure can be redrawn without
re-running the model -- and, unless ``--no-figures``, the plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from so101_policies.dreamzero import metrics  # noqa: E402


def parse_range(text: str) -> "list[int]":
    """``0-9``, ``0,3,7`` or ``0-4,9`` to a list of episode indices."""
    out: "list[int]" = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(piece))
    return out


def load_dataset(policy, root: str, episodes: "list[int]"):
    """The dataset, with the delta timestamps this policy's chunking needs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = policy.config
    fps_deltas = {}
    root_path = Path(root).expanduser()
    info = json.loads((root_path / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    for key in config.image_features:
        fps_deltas[key] = [index / fps for index in config.observation_delta_indices]
    fps_deltas["observation.state"] = [
        index / fps for index in config.observation_delta_indices
    ]
    fps_deltas["action"] = [index / fps for index in config.action_delta_indices]
    return LeRobotDataset(
        f"local/{root_path.name}",
        root=str(root_path),
        episodes=episodes or None,
        delta_timestamps=fps_deltas,
        video_backend="pyav",
    )


@torch.no_grad()
def evaluate_frame(policy, batch: dict) -> "dict[str, dict[str, list[float]]]":
    """Predicted vs actual for one observation, per camera, per horizon step."""
    config = policy.config
    context_frames = config.n_context_chunks * config.latent_frames_per_chunk

    predicted_tiled = policy.predict_future_frames(batch)
    actual_tiled = policy.tile_cameras(batch)
    context_tiled = actual_tiled[:, :context_frames]
    future_tiled = actual_tiled[:, context_frames:]

    predicted = policy.untile_cameras(predicted_tiled)
    actual = policy.untile_cameras(future_tiled)
    context = policy.untile_cameras(context_tiled)

    out = {}
    for key in predicted:
        result = metrics.compare(predicted[key], actual[key], context[key])
        out[key.split(".")[-1]] = {
            name: [round(float(v), 4) for v in value.tolist()]
            for name, value in result.items()
        }
    return out


def summarise(frames: "list[dict]") -> "dict[str, dict[str, list[float]]]":
    """Mean over frames, keeping the horizon axis: the fall with horizon IS the result."""
    if not frames:
        return {}
    cameras = list(frames[0])
    out = {}
    for camera in cameras:
        names = list(frames[0][camera])
        out[camera] = {
            name: np.mean([f[camera][name] for f in frames], axis=0).round(4).tolist()
            for name in names
        }
    return out


def report(summary: "dict[str, dict[str, list[float]]]") -> str:
    """A table a person can read, and the verdict that matters."""
    lines = [
        "",
        f"{'camera':<28} {'PSNR':>8} {'held':>8} {'SSIM':>7} {'held':>7}  beats held?",
        "-" * 74,
    ]
    for camera, values in summary.items():
        model_psnr = float(np.mean(values["psnr"]))
        held_psnr = float(np.mean(values["psnr_baseline"]))
        model_ssim = float(np.mean(values["ssim"]))
        held_ssim = float(np.mean(values["ssim_baseline"]))
        steps = sum(1 for a, b in zip(values["psnr"], values["psnr_baseline"]) if a > b)
        total = len(values["psnr"])
        verdict = f"{steps}/{total} steps"
        lines.append(
            f"{camera:<28} {model_psnr:>8.2f} {held_psnr:>8.2f} "
            f"{model_ssim:>7.3f} {held_ssim:>7.3f}  {verdict}"
        )
    lines.append("")
    lines.append(
        "'held' is the last observed frame repeated. A model that does not beat it "
        "has not\nlearned that camera's dynamics, whatever its PSNR says."
    )
    return "\n".join(lines)


def draw(summary: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, unit in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
        fig, axis = plt.subplots(figsize=(7, 4.2))
        for camera, values in summary.items():
            steps = range(1, len(values[name]) + 1)
            line = axis.plot(steps, values[name], marker="o", label=camera)[0]
            # The baseline in the SAME colour, dashed: the pairing is the point.
            axis.plot(
                steps,
                values[f"{name}_baseline"],
                linestyle="--",
                alpha=0.6,
                color=line.get_color(),
            )
        axis.set_xlabel("horizon step")
        axis.set_ylabel(unit)
        axis.set_title(
            f"Future prediction: {unit} per camera\n(dashed: last frame held)"
        )
        axis.grid(alpha=0.3)
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"prediction_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="dataset directory")
    parser.add_argument("--episodes", default="0-4")
    parser.add_argument("--every", type=int, default=20, help="sample one frame in N")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    from tool.eval_sim_policy import load_policy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _pre, _post, policy_type = load_policy(args.checkpoint, device)
    if not hasattr(policy, "predict_future_frames"):
        raise SystemExit(
            f"❌ {policy_type} is not a world model: it predicts actions but not "
            "observations, so there is no future to score."
        )

    episodes = parse_range(args.episodes)
    dataset = load_dataset(policy, args.dataset, episodes)
    print(f"checkpoint : {args.checkpoint}\npolicy     : {policy_type}")
    print(f"dataset    : {args.dataset} ({dataset.num_frames} frames)")

    collected: "list[dict]" = []
    for index in range(0, dataset.num_frames, args.every):
        if len(collected) >= args.max_frames:
            break
        item = dataset[index]
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in item.items()
            if isinstance(value, torch.Tensor)
        }
        try:
            collected.append(evaluate_frame(policy, batch))
        except ValueError as exc:  # a frame too near an episode edge to pad
            print(f"  skipped frame {index}: {exc}")
        print(f"\r  {len(collected)} frames", end="", flush=True)
    print()

    summary = summarise(collected)
    out_dir = Path(args.out or Path(args.checkpoint).parent / "prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "policy": policy_type,
        "dataset": str(args.dataset),
        "episodes": episodes,
        "frames": len(collected),
        "per_camera": summary,
    }
    (out_dir / "prediction.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(report(summary))
    if not args.no_figures and summary:
        draw(summary, out_dir)
    print(f"\n✅ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
