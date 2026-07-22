#!/usr/bin/env python3
"""Evaluate a trained LeRobot policy on the twin pick-and-place tasks.

Rolls a policy checkpoint out in the SAME ``PickPlaceTwinEnv`` the oracle
collected in, so observations are built identically to training frames (12-D
``observation.state`` in URDF degrees + gripper open fractions, and the three
twin cameras). Success is latched with the collection criterion — the cube
resting on the target within 2 cm, settled, with both grippers released.

The scenarios come from the disjoint EVAL seed pool (``full``, all 30 seeds,
one trial each) or the seed-0 scenario repeated (``simple``, the
overfit-one-scenario sanity mode). The camera resolution MUST match the value
the dataset was collected at, or the policy sees a distribution it never
trained on — the system scripts pass identical ``--camera-*`` to collect and
eval.

Examples
--------
    # 30-seed evaluation of an ACT run on the single-arm task
    python tool/eval_sim_policy.py --task single \\
        --checkpoint outputs/.../checkpoints/last/pretrained_model \\
        --out results.json --video-dir videos

    # sanity check: seed-0 scenario, five trials
    python tool/eval_sim_policy.py --task handover --seeds simple --episodes 5 \\
        --checkpoint <run>/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.configs import GRIPPER_OPEN_MAX_FRAC  # noqa: E402
from sim_benchmark.constants import SIDES  # noqa: E402
from sim_datagen.env import TASKS, PickPlaceTwinEnv  # noqa: E402
from sim_datagen.oracle import (  # noqa: E402
    HandoverContactScript,
    SinglePickPlaceScript,
    generate_relay_scenarios,
    generate_single_scenarios,
)
from sim_datagen.seeds import EVAL_SEEDS  # noqa: E402

# The rgb_scene view is the one written to the per-episode eval video (the
# analysis notebook composes head-to-head GIFs from these).
SCENE_CAMERA = "scene"
# A gripper counts as released once its open fraction exceeds half the cap, so
# a cube merely held over the target is never mistaken for a placement.
RELEASE_FRAC = 0.5 * GRIPPER_OPEN_MAX_FRAC


# ---------------------------------------------------------------------------
def _scenario_for_seed(task: str, seed: int) -> Any:
    if task == "single":
        return generate_single_scenarios(1, seed=seed)[0]
    return generate_relay_scenarios(1, seed=seed)[0]


def _make_script(task: str, scenario: Any, env: PickPlaceTwinEnv) -> Any:
    if task == "single":
        return SinglePickPlaceScript(scenario, env.neutral_ik_poses, env.table_z)
    return HandoverContactScript(scenario, env.neutral_ik_poses, env.table_z)


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
def load_policy(checkpoint: str, device: str) -> tuple[Any, Any, Any, str]:
    """Load a trained policy and its pre/post processors from a checkpoint dir.

    Returns ``(policy, preprocessor, postprocessor, policy_type)``. The
    checkpoint's own config carries the input/output feature shapes and the
    normalisation statistics, so no dataset metadata is needed here.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=checkpoint
    )
    return policy, preprocessor, postprocessor, cfg.type


def build_batch(
    state: np.ndarray, images: dict[str, np.ndarray], task_str: str, device: str
) -> dict[str, Any]:
    """Assemble the policy input batch from one env observation.

    Reuses LeRobot's ``preprocess_observation`` so the image tensors are shaped
    exactly as the training dataloader delivered them (channel-first float32 in
    ``[0, 1]``, keyed ``observation.images.<camera>``); the policy processor
    pipeline then applies normalisation.
    """
    import torch
    from lerobot.envs.utils import preprocess_observation

    raw = {"pixels": dict(images), "agent_pos": state.astype(np.float32)}
    batch = preprocess_observation(raw)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["task"] = [task_str]
    return batch


def decode_action(action12: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Split the 12-D policy action into 10-DOF joint radians + gripper fractions.

    Layout matches the recorder: ``[left5 deg, left_grip, right5 deg,
    right_grip]``. The gripper channels are the capped open fraction (in
    ``[0, GRIPPER_OPEN_MAX_FRAC]``, as recorded); dividing by the cap recovers
    the trigger-like fraction ``env.tick`` expects.
    """
    a = np.asarray(action12, dtype=float)
    left_deg, left_grip = a[0:5], a[5]
    right_deg, right_grip = a[6:11], a[11]
    q_rad = np.radians(np.concatenate([left_deg, right_deg]))
    grip = {}
    for side, g in (("left", left_grip), ("right", right_grip)):
        grip[side] = float(
            np.clip(g, 0.0, GRIPPER_OPEN_MAX_FRAC) / GRIPPER_OPEN_MAX_FRAC
        )
    return q_rad, grip


def _released(env: PickPlaceTwinEnv) -> bool:
    return all(env.sim.gripper_open_frac(s) > RELEASE_FRAC for s in SIDES)


# ---------------------------------------------------------------------------
def run_episode(
    env: PickPlaceTwinEnv,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    scenario: Any,
    task_str: str,
    fps: int,
    camera_wh: tuple[int, int],
    device: str,
    max_ticks: int,
    scene_frames: list[np.ndarray] | None,
) -> dict[str, Any]:
    """Roll one scenario out under the policy; return the per-episode record."""
    import torch

    env.reset(scenario)
    policy.reset()
    success = False
    t0 = time.time()
    for _ in range(max_ticks):
        state, images = env.observe(camera_wh)
        if scene_frames is not None:
            scene_frames.append(images[SCENE_CAMERA])
        batch = build_batch(state, images, task_str, device)
        with torch.inference_mode():
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
        action12 = np.asarray(action.squeeze(0).to("cpu")).astype(float)
        q_rad, grip = decode_action(action12)
        env.tick(q_rad, grip)

        if env.success() and _released(env):
            success = True
            break
    return {
        "success": bool(success),
        "place_err_mm": float(env.place_error() * 1e3),
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to a trained checkpoint's pretrained_model directory",
    )
    parser.add_argument(
        "--seeds",
        choices=["simple", "full"],
        default="full",
        help="full = every EVAL seed once (30 trials); simple = seed 0 repeated",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="simple mode: number of seed-0 trials (default 30); "
        "full mode: cap on EVAL seeds evaluated (default all 30)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="per-episode time budget (default 1.5x the oracle script duration)",
    )
    parser.add_argument("--out", type=str, default=None, metavar="PATH")
    parser.add_argument("--video-dir", type=str, default=None, metavar="DIR")
    args = parser.parse_args()

    device = _pick_device(args.device)
    camera_wh = (args.camera_width, args.camera_height)
    task_str = TASKS[args.task]

    env = PickPlaceTwinEnv(args.task)
    policy, preprocessor, postprocessor, policy_type = load_policy(
        args.checkpoint, device
    )
    print(
        f"🤖 {policy_type} on {device}; task '{args.task}', {args.seeds} seeds, "
        f"cameras {camera_wh[0]}x{camera_wh[1]}"
    )

    if args.seeds == "simple":
        n = args.episodes if args.episodes is not None else 30
        seeds = [20000] * n  # a fixed EVAL seed, repeated (scenario is constant)
    else:
        seeds = list(EVAL_SEEDS)
        if args.episodes is not None:
            seeds = seeds[: args.episodes]

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        scenario = _scenario_for_seed(args.task, seed)
        script = _make_script(args.task, scenario, env)
        if args.max_seconds is not None:
            max_ticks = int(np.ceil(args.max_seconds * args.fps))
        else:
            max_ticks = int(np.ceil(1.5 * script.duration * args.fps))
        scene_frames: list[np.ndarray] | None = [] if video_dir is not None else None
        rec = run_episode(
            env,
            policy,
            preprocessor,
            postprocessor,
            scenario,
            task_str,
            args.fps,
            camera_wh,
            device,
            max_ticks,
            scene_frames,
        )
        rec = {"scenario_seed": int(seed), **rec}
        # simple mode reuses one scenario; disambiguate the records by trial.
        if args.seeds == "simple":
            rec["trial"] = i
        episodes.append(rec)
        if video_dir is not None and scene_frames:
            import imageio.v2 as imageio

            name = (
                f"ep_seed{seed}_{i}.mp4"
                if args.seeds == "simple"
                else f"ep_seed{seed}.mp4"
            )
            imageio.mimsave(video_dir / name, scene_frames, fps=args.fps)
        tag = "✅" if rec["success"] else "❌"
        n_ok = sum(e["success"] for e in episodes)
        print(
            f"  seed {seed:>5} {tag}  place_err {rec['place_err_mm']:5.1f} mm  "
            f"({n_ok}/{len(episodes)} ok)",
            flush=True,
        )

    n_ok = sum(e["success"] for e in episodes)
    place_errs = [e["place_err_mm"] for e in episodes]
    result: dict[str, Any] = {
        "policy_type": policy_type,
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "seeds_mode": args.seeds,
        "n_episodes": len(episodes),
        "success_rate": n_ok / len(episodes) if episodes else 0.0,
        "place_err_mm_mean": float(np.mean(place_errs)) if place_errs else 0.0,
        "place_err_mm_p95": float(np.percentile(place_errs, 95)) if place_errs else 0.0,
        "camera_wh": list(camera_wh),
        "device": device,
        "episodes": episodes,
    }
    print("\n=== eval summary ===")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"📝 results → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
