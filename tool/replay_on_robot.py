"""Replay a recorded episode on the PHYSICAL followers and measure tracking.

Where ``tool/replay_recording.py`` re-renders a recorded episode on screen,
this tool drives the real follower arms: for every recorded frame it sends the
episode's ``action`` (the leader target that was commanded during collection) to
the followers and reads their measured joints back, then reports how faithfully
the followers reproduced the motion — a per-joint matplotlib plot (recorded vs
replayed) plus a printed mean/max error summary.

The action-to-hardware conversion mirrors the live command path exactly
(``common.threads.dual_joint_state``): the recorded action is URDF-space body
targets plus a 0–1 gripper fraction; each is turned back into the hardware goal
``signs * (urdf - offset)`` with the gripper scaled to 0–100.

SAFETY: the followers MOVE. The arms first ramp slowly to the episode's first
target (after a confirmation, unless ``--yes``), then step through the episode
at the dataset rate scaled by ``--speed``. Torque is disabled again on exit,
including on Ctrl+C or error. Keep the workspace clear and a hand near the power.

Usage:

    venv/bin/python tool/replay_on_robot.py \\
        --dir /media/hdd/so101 --name towel_fold --episode 0 --speed 0.5

    --dataset-root DIR --repo-id ID   # instead of --dir/--name
    --speed F        # playback rate factor (default 0.5 = half speed)
    --plot PATH      # tracking-error PNG (default under $SO101_OUTPUT_DIR)
    --yes            # skip the "arms will move" confirmation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.configs import (  # noqa: E402
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)

_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
_JOINT_LABELS = [*_BODY_JOINTS, "gripper"]
_OFFSETS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
}
_SIGNS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
}
# 12-channel action/state layout (actoris_harena.recording.features.STATE_NAMES):
# left 5 body + left gripper, then right 5 body + right gripper.
_SIDE_ACTION_SLICE = {"left": (slice(0, 5), 5), "right": (slice(6, 11), 11)}


def saved_episode_count(root: Path) -> int:
    """Episodes saved in a local dataset (``total_episodes`` from meta/info.json).

    Zero for a stillborn dataset (created, then quit before the first save).
    Kept local so this tool needs nothing from the Hub. Pure — unit-tested.
    """
    info = Path(root) / "meta" / "info.json"
    if not info.is_file():
        return 0
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except (ValueError, json.JSONDecodeError):
        return 0


def split_action(vec12) -> "dict[str, tuple[np.ndarray, float]]":
    """12-channel action/state → ``{side: (body5_urdf, gripper_frac)}``. Pure."""
    v = np.asarray(vec12, dtype=np.float64).reshape(-1)
    if v.shape[0] != 12:
        raise ValueError(f"expected a 12-channel vector, got {v.shape[0]}")
    out = {}
    for side, (body_sl, grip_i) in _SIDE_ACTION_SLICE.items():
        out[side] = (v[body_sl].copy(), float(v[grip_i]))
    return out


def action_to_goal(side: str, body_urdf, gripper_frac: float) -> dict:
    """URDF body targets + 0–1 gripper → a hardware ``Goal_Position`` dict.

    Inverse of the measured→URDF read: ``signs * (urdf - offset)`` for the body
    joints (the exact conversion in dual_joint_state), gripper scaled to 0–100.
    Pure — unit-tested.
    """
    hw = _SIGNS[side] * (np.asarray(body_urdf, dtype=np.float64) - _OFFSETS[side])
    goal = dict(zip(_BODY_JOINTS, hw))
    goal["gripper"] = float(gripper_frac) * 100.0
    return goal


def present_to_urdf(side: str, positions: dict) -> np.ndarray:
    """A follower ``Present_Position`` read → 6-vector [5 body URDF, gripper 0–1].

    Mirrors dual_joint_state's measured conversion so the replayed state is
    comparable to the recorded ``observation.state``. Pure — unit-tested.
    """
    body = (
        _SIGNS[side] * np.array([positions[j] for j in _BODY_JOINTS], dtype=np.float64)
        + _OFFSETS[side]
    )
    return np.array([*body, positions["gripper"] / 100.0], dtype=np.float64)


def tracking_errors(recorded, replayed) -> "tuple[np.ndarray, np.ndarray]":
    """Per-channel (mean, max) absolute error between two ``(T, 12)`` arrays. Pure."""
    diff = np.abs(np.asarray(recorded, dtype=np.float64) - np.asarray(replayed))
    return diff.mean(axis=0), diff.max(axis=0)


def _default_plot_path(name: str, episode: int) -> Path:
    base = Path(os.environ.get("SO101_OUTPUT_DIR", _root / "outputs"))
    return base / "replay_on_robot" / f"{name}_ep{episode:06d}.png"


def _print_error_summary(mean_err: np.ndarray, max_err: np.ndarray) -> None:
    print("\n📊 recorded vs replayed joint error (abs; body deg, gripper frac):")
    for side in ("left", "right"):
        base = 0 if side == "left" else 6
        for k, label in enumerate(_JOINT_LABELS):
            i = base + k
            print(
                f"    {side:5s} {label:13s} mean {mean_err[i]:7.3f}  "
                f"max {max_err[i]:7.3f}"
            )
    body_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    print(
        f"  body joints (deg): mean {mean_err[body_idx].mean():.3f}  "
        f"max {max_err[body_idx].max():.3f}"
    )


def _write_plot(recorded, replayed, out: Path, name: str, episode: int, fps: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recorded = np.asarray(recorded)
    replayed = np.asarray(replayed)
    t = np.arange(recorded.shape[0]) / max(fps, 1.0)
    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    for col, side in enumerate(("left", "right")):
        base = 0 if side == "left" else 6
        for k, label in enumerate(_JOINT_LABELS):
            ax = axes[k, col]
            i = base + k
            ax.plot(t, recorded[:, i], label="recorded", lw=1.4)
            ax.plot(t, replayed[:, i], label="replayed", lw=1.0, ls="--")
            unit = "frac" if label == "gripper" else "deg"
            ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
            if k == 0:
                ax.set_title(f"{side} arm", fontsize=11)
            if k == 5:
                ax.set_xlabel("time (s)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"replay-on-robot tracking — {name} episode {episode}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  💾 wrote {out}")


def _resolve_root_repo(args) -> "tuple[Path, str]":
    if args.dataset_root:
        root = Path(args.dataset_root).expanduser()
        repo_id = args.repo_id or root.name
    else:
        if not (args.dir and args.name):
            raise SystemExit("❌ give --dir and --name, or --dataset-root [--repo-id]")
        root = Path(args.dir).expanduser() / args.name
        repo_id = args.repo_id or args.name
    if not root.exists():
        raise SystemExit(f"❌ dataset not found at {root}")
    return root, repo_id


def _ramp_to(buses: dict, goals: dict, duration: float = 3.0, hz: float = 50.0) -> None:
    """Interpolate both followers from their present pose to ``goals`` gently."""
    starts = {s: b.sync_read("Present_Position") for s, b in buses.items()}
    steps = max(1, int(duration * hz))
    for k in range(1, steps + 1):
        a = k / steps
        for s, b in buses.items():
            pos = {j: (1 - a) * starts[s][j] + a * goals[s][j] for j in goals[s]}
            b.sync_write("Goal_Position", pos, normalize=True, num_retry=2)
        time.sleep(1.0 / hz)


def _connect_followers() -> dict:
    """Both followers, limp, bound to the sensor-map ports (physical L/R arms)."""
    from common.follower_bus import connect_follower_bus
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    arms = (
        load_sensor_map(SENSOR_MAP_PATH).get("arms", {})
        if SENSOR_MAP_PATH.exists()
        else {}
    )
    # connect_follower_bus("left"/"right", port) reconstructs the same bus the
    # teleop drove: side "left" → follower_1 calibration, "right" → follower_0,
    # each on its sensor-map port so the recorded left/right match the arms.
    return {
        "left": connect_follower_bus("left", arms.get("left")),
        "right": connect_follower_bus("right", arms.get("right")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", help="Collection directory (with --name)")
    parser.add_argument("--name", help="Dataset name (with --dir)")
    parser.add_argument(
        "--dataset-root", help="Dataset directory (instead of --dir/--name)"
    )
    parser.add_argument("--repo-id", help="Dataset repo id (defaults to the name)")
    parser.add_argument(
        "--episode", type=int, required=True, help="Episode id to replay"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Playback rate factor (default 0.5 = half the dataset rate)",
    )
    parser.add_argument("--plot", help="Tracking-error PNG path")
    parser.add_argument(
        "--yes", action="store_true", help="Skip the 'arms will move' confirmation"
    )
    args = parser.parse_args()

    if args.speed <= 0:
        raise SystemExit("❌ --speed must be > 0")

    root, repo_id = _resolve_root_repo(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    n_saved = saved_episode_count(root)
    if n_saved == 0:
        raise SystemExit(f"❌ {root} has no saved episodes yet — nothing to replay.")
    if not 0 <= args.episode < n_saved:
        raise SystemExit(
            f"❌ episode {args.episode} out of range; dataset has {n_saved} "
            f"episode(s) (valid: 0..{n_saved - 1})"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[args.episode])
    n = len(ds)
    if n == 0:
        raise SystemExit(f"❌ episode {args.episode} has no frames")
    fps = float(ds.meta.fps)
    dt = 1.0 / (fps * args.speed)
    print(
        f"▶ replay-on-robot episode {args.episode}: {n} frames, dataset {fps:.0f} fps, "
        f"speed {args.speed}× → {1.0 / dt:.1f} Hz command rate"
    )

    # Pre-extract the recorded action (command) and state (measured) per frame.
    actions = [split_action(ds[i]["action"]) for i in range(n)]
    recorded_state = np.stack(
        [np.asarray(ds[i]["observation.state"]) for i in range(n)]
    )

    if not args.yes:
        print(
            "\n⚠️  The follower arms will MOVE: ramp to the first recorded pose, "
            "then replay the episode.\n   Clear the workspace. Press Enter to "
            "proceed (Ctrl+C to abort)..."
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("aborted")

    buses = _connect_followers()
    replayed_state = np.zeros((n, 12), dtype=np.float64)
    try:
        for b in buses.values():
            b.enable_torque()

        print("🏁 ramping to the first recorded target...")
        first = {s: action_to_goal(s, *actions[0][s]) for s in ("left", "right")}
        _ramp_to(buses, first, duration=3.0)
        time.sleep(0.5)

        print("🔴 replaying...")
        for i in range(n):
            for s in ("left", "right"):
                buses[s].sync_write(
                    "Goal_Position",
                    action_to_goal(s, *actions[i][s]),
                    normalize=True,
                    num_retry=2,
                )
            time.sleep(dt)
            for s in ("left", "right"):
                pos = buses[s].sync_read("Present_Position", num_retry=2)
                base = 0 if s == "left" else 6
                replayed_state[i, base : base + 6] = present_to_urdf(s, pos)
        print(f"✓ replayed {n} frames")
    finally:
        for b in buses.values():
            try:
                b.disable_torque(num_retry=3)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  could not disable torque on a follower: {e}")

    mean_err, max_err = tracking_errors(recorded_state, replayed_state)
    _print_error_summary(mean_err, max_err)
    out = Path(args.plot) if args.plot else _default_plot_path(repo_id, args.episode)
    _write_plot(recorded_state, replayed_state, out, repo_id, args.episode, fps)


if __name__ == "__main__":
    main()
