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
from actoris_harena.policies.dreamzero import metrics  # noqa: E402


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


def preprocess(pre, batch: dict, device) -> dict:
    """The batch as the model saw it in training: through its own preprocessor.

    Leaving this out does not fail. It hands the model raw joint angles in
    DEGREES, up to about 110, as its state and past actions, where training
    gave it values normalised to within a few units of zero -- and the model
    duly predicts noise. MEASURED 2026-09-25 on the 80 000-step DreamZero, same
    frames, same seed: 6.2 / 6.2 / 7.0 dB raw against 16.6 / 18.3 / 14.3 dB
    preprocessed, with a VAE round trip of about 27 dB as the ceiling. Every
    DreamZero number before that date was scored raw.

    The preprocessor ends by moving the batch to the device it was TRAINED on,
    so tensors are put back where this run's weights are.
    """
    out = pre(dict(batch))
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in out.items()
    }


def truth_batch(pre, batch: dict) -> dict:
    """The raw batch with the checkpoint's tactile CROP applied, and nothing else.

    The truth comes from the raw batch so that normalisation can never move it
    along with the prediction (see :func:`preprocess`). But a model trained on
    cropped fingertips predicts cropped fingertips, and scoring that against the
    uncropped frame would charge it for the rim it was never shown. So the crop
    steps -- and only those -- are taken from the checkpoint's own pipeline and
    applied here too. A checkpoint with no crop gets the batch back unchanged.
    """
    from actoris_harena.policies.common.tactile import HarenaTactileCropProcessorStep

    out = dict(batch)
    for step in getattr(pre, "steps", []):
        if isinstance(step, HarenaTactileCropProcessorStep):
            out = step.observation(out)
    return out


def split_composites(
    views: "dict[str, torch.Tensor]", composites: "dict[str, tuple[str, ...]]"
) -> "dict[str, torch.Tensor]":
    """A composite camera split back into one entry per sensor.

    FastWAM reads the four fingertips as one tiled image, so without this its
    numbers are per COMPOSITE while DreamZero's are per sensor, and the two
    cannot share a table. The tiles are cut on the grid the composite was
    written with (``recording.dataset_view.composite_grid``), in the order its
    parts are listed, which is the order they were tiled in.
    """
    from actoris_harena.recording.dataset_view import composite_grid

    out: "dict[str, torch.Tensor]" = {}
    for key, value in views.items():
        parts = composites.get(key.split(".")[-1])
        if not parts:
            out[key] = value
            continue
        rows, cols = composite_grid(len(parts))
        tile_h, tile_w = value.shape[-2] // rows, value.shape[-1] // cols
        for index, part in enumerate(parts):
            row, col = divmod(index, cols)
            out[part] = value[
                ...,
                row * tile_h : (row + 1) * tile_h,
                col * tile_w : (col + 1) * tile_w,
            ]
    return out


def make_batch(item: dict, device) -> dict:
    """One dataset row as a batch of one, keeping what is not a tensor.

    The task description is a STRING, so a tensors-only batch drops it.
    DreamZero never misses it -- it conditions on a single learned task
    embedding -- but a language-conditioned world model refuses to run without
    a prompt, and refuses it on every frame. This scorer was written against
    the model that did not need one.
    """
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    if "task" in item and not isinstance(item["task"], torch.Tensor):
        batch["task"] = item["task"]
    return batch


def as_predictor(policy):
    """A FastWAM of any variant, given the methods the scorer asks for.

    ``harena_fastwam_predict`` adds four methods and two config fields to
    FastWAM and changes nothing else, and a checkpoint is normally retargeted
    to it. A SIBLING variant cannot be: the action-conditioned run is a
    ``harena_fastwam_crop`` (with the crop switched off), and the retarget
    refuses to cross from one variant to another, rightly, since their configs
    differ. So the predict class's methods are bound onto the loaded policy
    instead, and its config gains the predict defaults. Any other policy, and
    a FastWAM that already predicts, comes back unchanged.
    """
    if hasattr(policy, "predict_future_frames"):
        return policy
    import types

    from actoris_harena.policies.fastwam.modeling_fastwam import HarenaFastwamPolicy
    from actoris_harena.policies.fastwam_predict.configuration_fastwam_predict import (
        HarenaFastwamPredictConfig,
    )
    from actoris_harena.policies.fastwam_predict.modeling_fastwam_predict import (
        HarenaFastwamPredictPolicy,
    )

    if not isinstance(policy, HarenaFastwamPolicy):
        return policy
    for name in (
        "predict_future_frames",
        "tile_cameras",
        "untile_cameras",
        "reconstruct_frames",
    ):
        method = getattr(HarenaFastwamPredictPolicy, name)
        object.__setattr__(policy, name, types.MethodType(method, policy))
    config = policy.config
    defaults = HarenaFastwamPredictConfig.__dataclass_fields__
    for name in ("predict_inference_steps", "predict_seed"):
        if not hasattr(config, name):
            setattr(config, name, defaults[name].default)
    # The predict config's context arithmetic: one observed frame, see there.
    config.n_context_chunks = 1
    config.latent_frames_per_chunk = 1
    return policy


def parse_overrides(text: str, stats: "dict | None" = None) -> "dict[int, float]":
    """``"5=0.02,11=max"`` -> ``{5: 0.02, 11: <that column's max>}``.

    ``min`` and ``max`` read the dataset's own ``observation.state`` statistics,
    so "gripper closed" and "gripper open" are values the recordings contain
    rather than numbers typed in from memory of the arm.
    """
    out: "dict[int, float]" = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        column, _, value = part.partition("=")
        index = int(column)
        if value in ("min", "max"):
            if not stats:
                raise ValueError(f"{part}: min/max needs the dataset's stats")
            out[index] = float(stats["observation.state"][value][index])
        else:
            out[index] = float(value)
    return out


def override_state(batch: dict, values: "dict[int, float]") -> dict:
    """The raw batch with chosen state columns replaced at every time step."""
    if not values:
        return batch
    out = dict(batch)
    state = batch["observation.state"].clone()
    for column, value in values.items():
        state[..., column] = value
    out["observation.state"] = state
    return out


def conditioning_actions(
    model_batch: dict, horizon: int, mode: str, gripper_columns: "tuple[int, ...]"
) -> "torch.Tensor | None":
    """The action chunk an action-conditioned video model is shown, or None.

    ``recorded`` is what the operator actually commanded next. ``hold-gripper``
    is the same chunk with each gripper column frozen at its first value, so a
    grasp that the recording closes on never gets its closing command. Taken
    from the PREPROCESSED batch, in the normalised units the model trained on;
    the normaliser is affine per column, so holding a normalised value holds the
    raw one.
    """
    if mode == "none":
        return None
    actions = model_batch["action"][:, :horizon].clone()
    if mode == "hold-gripper":
        for column in gripper_columns:
            actions[:, :, column] = actions[:, :1, column]
    elif mode != "recorded":
        raise ValueError(f"unknown action conditioning {mode!r}")
    return actions


def check_coverage(scored: int, skipped: "list[str]") -> str:
    """Refuse a result that rests on nothing; warn about one resting on little.

    A run that scored NOTHING used to print an empty table, write a result file
    and exit zero with a tick. MEASURED 2026-09-18: a shape mismatch in the
    conditioning state made the scorer refuse all twenty-four sampled frames,
    and the only evidence was in a log nobody had to read.

    The per-frame exception this backs onto exists for the occasional frame too
    near an episode edge to pad. That is an edge case; every frame failing is a
    fault in the run, and the two must not end the same way.
    """
    if scored == 0:
        reasons = sorted(set(skipped))
        detail = "\n  ".join(reasons[:3]) or "no frames were sampled at all"
        raise SystemExit(
            f"❌ scored no frames of {len(skipped)} attempted, so nothing was "
            f"written. Every frame was refused, which is a fault in the run and "
            f"not an edge case:\n  {detail}"
        )
    if len(skipped) > scored:
        return (
            f"⚠️  {len(skipped)} frames skipped against {scored} scored; "
            "the result rests on a minority of the recording"
        )
    return ""


@torch.no_grad()
def as_uint8(frame) -> np.ndarray:
    """One ``[C, H, W]`` frame in ``[0, 1]`` as an ``H x W x C`` image."""
    array = frame.detach().float().clamp(0, 1).cpu().numpy()
    return (array.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def save_filmstrip(kept: dict, out: Path) -> Path:
    """The arrays behind a filmstrip, so a report can redraw any subset of them.

    Keys are ``<camera>/held``, ``<camera>/actual`` and ``<camera>/predicted``,
    and ``<camera>/held_recon`` -- the held frame through the model's own
    autoencoder -- when the model can reconstruct.
    """
    arrays = {}
    for camera, parts in kept.items():
        arrays[f"{camera}/held"] = parts["held"]
        if parts.get("held_recon") is not None:
            arrays[f"{camera}/held_recon"] = parts["held_recon"]
        arrays[f"{camera}/actual"] = np.stack(parts["actual"])
        arrays[f"{camera}/predicted"] = np.stack(parts["predicted"])
    np.savez_compressed(out, **arrays)
    return out


def load_filmstrip(path: Path) -> dict:
    """The inverse of :func:`save_filmstrip`."""
    kept: dict = {}
    with np.load(path) as data:
        for key in data.files:
            camera, kind = key.rsplit("/", 1)
            value = data[key]
            single = kind in ("held", "held_recon")
            kept.setdefault(camera, {})[kind] = value if single else list(value)
    return kept


def difference_image(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Truth minus prediction, both in ``[0, 1]``, mapped from ``[-1, 1]`` to ``[0, 1]``.

    Mid-grey is no error; brighter is where the model drew too dark, darker
    where it drew too bright. Two uint8 images of the same shape in, one out.
    """
    diff = truth.astype(np.float64) / 255.0 - predicted.astype(np.float64) / 255.0
    return ((diff + 1.0) / 2.0 * 255.0).round().astype(np.uint8)


def draw_filmstrip(
    kept: dict, out: Path, title: str = "", cameras: "list[str] | None" = None
) -> Path:
    """Actual, predicted and their difference, per camera, across the horizon.

    Three rows per camera. Column 0 is the last frame the model saw: the actual
    row shows it, the predicted row shows the model's reconstruction of it (its
    own autoencoder, when it has one), and every later column is one step of the
    horizon, so the rows line up in time. The difference row is truth minus
    prediction (see :func:`difference_image`): flat grey is a perfect frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cameras = [c for c in (cameras or list(kept)) if c in kept]
    steps = len(kept[cameras[0]]["actual"])
    rows = [
        (c, kind) for c in cameras for kind in ("actual", "predicted", "difference")
    ]
    columns = steps + 1
    figure, axes = plt.subplots(
        len(rows),
        columns,
        figsize=(1.5 * columns + 1.2, 1.5 * len(rows)),
        squeeze=False,
    )
    for r, (camera, kind) in enumerate(rows):
        parts = kept[camera]
        truth = [parts["held"], *parts["actual"]]
        guess = [parts.get("held_recon"), *parts["predicted"]]
        for column in range(columns):
            axis = axes[r][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if kind == "actual":
                image = truth[column]
            elif guess[column] is None:
                image = None
            elif kind == "predicted":
                image = guess[column]
            else:
                image = difference_image(truth[column], guess[column])
            if image is None:
                axis.axis("off")
            else:
                axis.imshow(image)
            if r == 0:
                axis.set_title("seen" if column == 0 else f"step {column}", fontsize=9)
            if column == 0:
                label = f"{camera}\n{kind}" if kind == "actual" else kind
                axis.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    # The title needs its own band: left to tight_layout it lands on the
    # middle column's "step" header, which is how the first render came out.
    figure.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    if title:
        figure.suptitle(title, fontsize=11, y=0.995)
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def evaluate_frame(
    policy,
    batch: dict,
    seed: "int | None" = None,
    model_batch: "dict | None" = None,
    keep: "list | None" = None,
    composites: "dict[str, tuple[str, ...]] | None" = None,
    predict_kwargs: "dict | None" = None,
) -> "dict[str, dict[str, list[float]]]":
    """Predicted vs actual for one observation, per camera, per horizon step.

    ``model_batch`` is what the MODEL sees -- the batch after the checkpoint's
    own preprocessor -- and ``batch`` is where the truth comes from. They are
    kept apart so that a processor which rescaled the images could never move
    the ground truth along with the prediction. See :func:`preprocess`.

    ``keep``, when given, receives the IMAGES behind the numbers -- per camera,
    the last observed frame (and its reconstruction), the actual future and the
    predicted one -- so a
    filmstrip shows the frames that were scored and not a separate rollout.

    ``predict_kwargs`` go to ``predict_future_frames`` as they are -- the action
    chunk an action-conditioned model is shown, for one.

    ``composites`` names any tiled camera's parts, so every number is reported
    per SENSOR (see :func:`split_composites`).

    When the model can round-trip frames through its own autoencoder, the
    observed frames are also scored that way -- ``recon_psnr`` / ``recon_ssim``,
    one value averaged over the context frames. That is the ceiling on the
    prediction: what the model loses by compressing an image it was shown.

    ``seed`` pins the sampler. It is not optional in spirit, only in signature:
    DreamZero integrates its flow from ``torch.randn`` and FastWAM's
    ``infer_joint`` samples too, so two passes over ONE observation disagree and
    a run scored twice reports two different answers.

    MEASURED 2026-09-16, by accident -- two copies of the same scoring script
    ran over the same step-2000 checkpoint and wrote in turn. Per-camera PSNR
    moved by up to 0.05 dB (5.77 against 5.82 on one fingertip) while every
    held-last-frame baseline was identical to the digit, which is what locates
    the difference in the sampler rather than in frame selection: holding a
    frame involves no sampling. 0.05 dB is small beside the three to eight dB by
    which the 80 000-step DreamZero beats holding (2026-09-25, scored through
    its preprocessor -- the "13 dB" gap once quoted here came from a scorer
    that skipped it), and it is NOT small beside the difference two trained
    arms would be compared on.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # FastWAM seeds its OWN generator from its config and never reads the
        # global one, so the global seed alone would leave it on whatever the
        # config says -- a --seed that silently changes nothing.
        from actoris_harena.analysis.diffusion import SEED_FIELDS

        for name in SEED_FIELDS:
            if hasattr(policy.config, name):
                setattr(policy.config, name, seed)

    config = policy.config
    context_frames = config.n_context_chunks * config.latent_frames_per_chunk

    predicted_tiled = policy.predict_future_frames(
        batch if model_batch is None else model_batch, **(predict_kwargs or {})
    )
    actual_tiled = policy.tile_cameras(batch)
    context_tiled = actual_tiled[:, :context_frames]
    future_tiled = actual_tiled[:, context_frames:]

    parts = composites or {}
    predicted = split_composites(policy.untile_cameras(predicted_tiled), parts)
    actual = split_composites(policy.untile_cameras(future_tiled), parts)
    context = split_composites(policy.untile_cameras(context_tiled), parts)
    reconstructed = None
    if hasattr(policy, "reconstruct_frames"):
        reconstructed = split_composites(
            policy.untile_cameras(
                policy.reconstruct_frames(context_tiled).to(context_tiled.device)
            ),
            parts,
        )

    if keep is not None:
        keep.append(
            {
                key.split(".")[-1]: {
                    "held": as_uint8(context[key][0, -1]),
                    "held_recon": (
                        None
                        if reconstructed is None
                        else as_uint8(reconstructed[key][0, -1])
                    ),
                    "actual": [as_uint8(f) for f in actual[key][0]],
                    "predicted": [as_uint8(f) for f in predicted[key][0]],
                }
                for key in predicted
            }
        )

    out = {}
    for key in predicted:
        result = metrics.compare(predicted[key], actual[key], context[key])
        row = {
            name: [round(float(v), 4) for v in value.tolist()]
            for name, value in result.items()
        }
        if reconstructed is not None:
            row["recon_psnr"] = [
                round(float(metrics.psnr(reconstructed[key], context[key]).mean()), 4)
            ]
            row["recon_ssim"] = [
                round(float(metrics.ssim(reconstructed[key], context[key]).mean()), 4)
            ]
        out[key.split(".")[-1]] = row
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
        f"{'camera':<28} {'recon':>7} {'PSNR':>8} {'held':>8} {'SSIM':>7} {'held':>7}  beats held?",
        "-" * 82,
    ]
    for camera, values in summary.items():
        model_psnr = float(np.mean(values["psnr"]))
        held_psnr = float(np.mean(values["psnr_baseline"]))
        model_ssim = float(np.mean(values["ssim"]))
        held_ssim = float(np.mean(values["ssim_baseline"]))
        steps = sum(1 for a, b in zip(values["psnr"], values["psnr_baseline"]) if a > b)
        total = len(values["psnr"])
        verdict = f"{steps}/{total} steps"
        recon = values.get("recon_psnr")
        recon_text = f"{recon[0]:>7.2f}" if recon else f"{'-':>7}"
        lines.append(
            f"{camera:<28} {recon_text} {model_psnr:>8.2f} {held_psnr:>8.2f} "
            f"{model_ssim:>7.3f} {held_ssim:>7.3f}  {verdict}"
        )
    lines.append("")
    lines.append(
        "'held' is the last observed frame repeated. A model that does not beat it "
        "has not\nlearned that camera's dynamics, whatever its PSNR says. 'recon' is "
        "the frames it was\nshown, through its own autoencoder and back: the "
        "sharpest a prediction can be."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="pins the sampler; --seed -1 leaves it free, which makes the run "
        "irreproducible and is only useful for measuring the sampler's own spread",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--filmstrip",
        type=int,
        nargs="*",
        default=[],
        help="frame indices (as the loop counts them, within the loaded "
        "episodes) to draw as actual / predicted / held filmstrips",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=[],
        help="score exactly these frame indices (within the loaded episodes) "
        "instead of every --every-th; each also gets a filmstrip",
    )
    parser.add_argument(
        "--state-override",
        default="",
        help='replace state columns before the model sees them, e.g. "5=min,11=min" '
        "(min/max: the dataset's own range for that column)",
    )
    parser.add_argument(
        "--condition-actions",
        choices=["auto", "none", "recorded", "hold-gripper"],
        default="auto",
        help="the action chunk an action-conditioned video model is shown; auto "
        "is recorded for such a model and none otherwise",
    )
    args = parser.parse_args()

    from tool.eval_sim_policy import load_policy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, _post, policy_type = load_policy(args.checkpoint, device)
    policy = as_predictor(policy)
    if not hasattr(policy, "predict_future_frames"):
        raise SystemExit(
            f"❌ {policy_type} is not a world model: it predicts actions but not "
            "observations, so there is no future to score."
        )

    from actoris_harena.recording import camera_profile

    import common  # noqa: F401  -- declares this rig's cameras to actoris_harena
    from common.robot_schema import GRIPPER_COLUMNS

    composites = dict(camera_profile.profile().composites)
    stats_path = Path(args.dataset).expanduser() / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    overrides = parse_overrides(args.state_override, stats)
    video_config = getattr(policy.config, "video_dit_config", None) or {}
    conditioned = bool(video_config.get("action_conditioned"))
    mode = args.condition_actions
    if mode == "auto":
        mode = "recorded" if conditioned else "none"
    if mode != "none" and not conditioned:
        print(
            "⚠️  this model's video never reads an action, so the chunk passed "
            "with --condition-actions cannot change a predicted frame"
        )
    if overrides:
        print(f"state override: {overrides}")
    print(f"actions    : {mode}")
    pinned = None if args.seed < 0 else args.seed
    episodes = parse_range(args.episodes)
    dataset = load_dataset(policy, args.dataset, episodes)
    print(f"checkpoint : {args.checkpoint}\npolicy     : {policy_type}")
    print(f"dataset    : {args.dataset} ({dataset.num_frames} frames)")

    wanted = set(args.filmstrip) | set(args.frames)
    strips: "dict[int, list]" = {}
    collected: "list[dict]" = []
    skipped: "list[str]" = []
    indices = args.frames or range(0, dataset.num_frames, args.every)
    for index in indices:
        if len(collected) >= args.max_frames:
            break
        batch = override_state(make_batch(dataset[index], device), overrides)
        model_batch = preprocess(pre, batch, device)
        actions = conditioning_actions(
            model_batch,
            int(getattr(policy.config, "action_horizon", 0) or 0),
            mode,
            GRIPPER_COLUMNS,
        )
        try:
            # The SAME seed for every frame, not seed+index: each frame is an
            # independent prediction, and what has to be reproducible is the
            # noise this observation is integrated from.
            collected.append(
                evaluate_frame(
                    policy,
                    truth_batch(pre, batch),
                    seed=pinned,
                    model_batch=model_batch,
                    keep=strips.setdefault(index, []) if index in wanted else None,
                    composites=composites,
                    predict_kwargs=None if actions is None else {"action": actions},
                )
            )
        except ValueError as exc:  # a frame too near an episode edge to pad
            skipped.append(str(exc))
            print(f"  skipped frame {index}: {exc}")
        print(f"\r  {len(collected)} frames", end="", flush=True)
    print()

    warning = check_coverage(len(collected), skipped)
    if warning:
        print(warning)

    summary = summarise(collected)
    out_dir = Path(args.out or Path(args.checkpoint).parent / "prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "policy": policy_type,
        "dataset": str(args.dataset),
        "episodes": episodes,
        "frames": len(collected),
        "skipped": len(skipped),
        "seed": pinned,
        "state_override": {str(k): v for k, v in overrides.items()},
        "condition_actions": mode,
        "per_camera": summary,
    }
    (out_dir / "prediction.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(report(summary))
    if not args.no_figures and summary:
        draw(summary, out_dir)
    for index, kept in sorted(strips.items()):
        if kept:
            drawn = draw_filmstrip(
                kept[0], out_dir / f"filmstrip_{index}.png", f"frame {index}"
            )
            save_filmstrip(kept[0], out_dir / f"filmstrip_{index}.npz")
            print(f"🎞️  {drawn}")
    missed = sorted(wanted - {i for i, k in strips.items() if k})
    if missed:
        print(f"⚠️  no filmstrip for {missed}: not on the sampling grid, or skipped")
    print(f"\n✅ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
