"""Run a trained policy on the PHYSICAL followers (on-robot inference).

The real-hardware analogue of ``tool/eval_sim_policy.py``: instead of rolling a
checkpoint out in the twin, this builds observations from the live cameras and
the followers' measured joints, runs the SAME inference (``load_policy`` ->
preprocess -> ``select_action`` -> postprocess), and sends the predicted 12-D
action to the followers through the exact conversion the recorder and
``tool/replay_on_robot.py`` use. So the action definition the dataset froze is
what drives the arms — no separate real-inference code path.

The policy input must match training: the enabled cameras (``recording.yaml``)
supply ``observation.images.<name>`` and both followers supply the 12-D
``observation.state`` (URDF degrees + gripper open fractions). Capture at the
resolution the dataset was collected at.

SAFETY: the followers MOVE (unless ``--dry-run``). After a confirmation (skip
with ``--yes``) the arms ramp slowly to the policy's first action, then run at
``--hz`` until ``--seconds`` elapse or Ctrl+C. Torque is disabled again on exit,
including on error. ``--dry-run`` reads sensors and prints the chosen actions
but never enables torque or writes a goal. Keep the workspace clear.

Usage:

    venv/bin/python tool/run_policy_real.py \\
        --checkpoint <run>/checkpoints/last/pretrained_model \\
        --task "fold the towel" --dry-run          # infer only, no motion

    venv/bin/python tool/run_policy_real.py \\
        --checkpoint <ckpt> --task "fold the towel" --hz 15 --seconds 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from tool.replay_on_robot import (  # noqa: E402
    _connect_followers,
    _ramp_to,
    action_to_goal,
    present_to_urdf,
    split_action,
)

_SIDES = ("left", "right")


def policy_action_to_goals(action12) -> "dict[str, dict]":
    """A 12-D policy action (URDF deg + gripper frac) → per-side hardware goals.

    Composes ``split_action`` (the 12-channel layout) with ``action_to_goal``
    (the URDF->hardware conversion), so the policy's action reaches the motors
    through the exact path a recorded action does in ``replay_on_robot``. Pure —
    unit-tested.
    """
    per_side = split_action(action12)
    return {s: action_to_goal(s, *per_side[s]) for s in _SIDES}


def read_state(buses: dict) -> np.ndarray:
    """Assemble the 12-D ``observation.state`` from both followers' present pose."""
    state = np.zeros(12, dtype=np.float64)
    for s in _SIDES:
        pos = buses[s].sync_read("Present_Position", num_retry=2)
        base = 0 if s == "left" else 6
        state[base : base + 6] = present_to_urdf(s, pos)
    return state


def _start_cameras(data_manager):
    """Open + start the recording.yaml cameras, publishing into ``data_manager``.

    Reuses the teleop tool's stream resolution so the on-robot policy sees the
    same camera set (and stable by-path devices) the dataset was collected with.
    Returns the started capture objects (RGB cameras plus any central RealSense).
    """
    from types import SimpleNamespace

    from common.config_parser import load_recording_config
    from common.recording.cameras import CameraCapture
    from tool.meta_quest_teleopration import (
        build_realsense_capture,
        overlay_sensor_map_devices,
        resolve_camera_streams,
    )
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    rec_cfg = load_recording_config()
    shim = SimpleNamespace(
        enable_camera=[], disable_camera=[], tactile=False, central_depth=False
    )
    streams = resolve_camera_streams(rec_cfg, shim)
    sensor_map = load_sensor_map(SENSOR_MAP_PATH) if SENSOR_MAP_PATH.exists() else {}
    if sensor_map:
        streams = overlay_sensor_map_devices(streams, sensor_map)

    caps: list = []
    for name, cfg in streams.items():
        cam = CameraCapture(
            name=name,
            device=cfg["device"],
            width=cfg["width"],
            height=cfg["height"],
            fps=cfg["fps"],
            rotate180=cfg["rotate180"],
            fourcc=cfg["fourcc"],
        )
        if not cam.open():
            for c in caps:
                c.stop()
            raise SystemExit(
                f"❌ camera '{name}' failed to open (device {cfg['device']})"
            )
        caps.append(cam)

    rs = build_realsense_capture(rec_cfg, shim, sensor_map, for_view=False)
    if rs is not None and rs.open():
        caps.append(rs)

    for c in caps:
        c.start(data_manager)
    return caps


def _gather_images(data_manager, names: "list[str]", timeout_s: float = 5.0) -> dict:
    """Latest RGB frame for each camera; waits briefly for all to arrive."""
    deadline = time.monotonic() + timeout_s
    while True:
        images = {n: data_manager.get_rgb_image(n) for n in names}
        if all(v is not None for v in images.values()):
            return images
        if time.monotonic() > deadline:
            missing = [n for n, v in images.items() if v is None]
            raise SystemExit(f"❌ no camera frames for {missing} after {timeout_s}s")
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .../pretrained_model dir"
    )
    parser.add_argument(
        "--task", required=True, help="Language task string fed to the policy"
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Control rate (Hz)")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="Run duration before stopping"
    )
    parser.add_argument("--device", default=None, help="cpu/cuda (default auto)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Infer + print actions but never enable torque or command the arms",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the 'arms will move' confirmation"
    )
    args = parser.parse_args()

    if args.hz <= 0 or args.seconds <= 0:
        raise SystemExit("❌ --hz and --seconds must be > 0")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"❌ checkpoint not found: {ckpt}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch

    from tool.eval_sim_policy import build_batch, load_policy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 loading policy from {ckpt} on {device} ...")
    policy, preprocessor, postprocessor, ptype = load_policy(str(ckpt), device)
    policy.reset()
    print(f"  ✓ loaded a '{ptype}' policy")

    if not args.dry_run and not args.yes:
        print(
            "\n⚠️  The follower arms will MOVE under policy control: ramp to the "
            "first action, then run.\n   Clear the workspace. Press Enter to "
            "proceed (Ctrl+C to abort)..."
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("aborted")

    from common.data_manager_dual import DualDataManager

    data_manager = DualDataManager()
    captures = _start_cameras(data_manager)
    image_names = [c.name for c in captures]
    buses = _connect_followers()

    dt = 1.0 / args.hz
    n_ticks = int(args.seconds * args.hz)
    torque_on = False
    try:
        # Warm up the observation, then run one inference for the ramp target.
        images = _gather_images(data_manager, image_names)
        state = read_state(buses)
        action12 = _infer(
            policy,
            preprocessor,
            postprocessor,
            build_batch,
            state,
            images,
            args.task,
            device,
            torch,
        )
        goals = policy_action_to_goals(action12)

        if args.dry_run:
            print(
                "🧪 dry-run: inferring at "
                f"{args.hz:.0f} Hz for {args.seconds:.0f}s, NO motor writes"
            )
        else:
            for b in buses.values():
                b.enable_torque()
            torque_on = True
            print("🏁 ramping to the policy's first action ...")
            _ramp_to(buses, goals, duration=3.0)
            time.sleep(0.5)
            print("🔴 running policy ...")

        for tick in range(n_ticks):
            t0 = time.perf_counter()
            images = _gather_images(data_manager, image_names)
            state = read_state(buses)
            action12 = _infer(
                policy,
                preprocessor,
                postprocessor,
                build_batch,
                state,
                images,
                args.task,
                device,
                torch,
            )
            goals = policy_action_to_goals(action12)
            if args.dry_run:
                if tick % max(1, int(args.hz)) == 0:
                    print(
                        f"  t={tick / args.hz:5.1f}s  action[:6]={action12[:6].round(2)}"
                    )
            else:
                for s in _SIDES:
                    buses[s].sync_write(
                        "Goal_Position", goals[s], normalize=True, num_retry=2
                    )
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        print(f"✓ finished {n_ticks} ticks")
    except KeyboardInterrupt:
        print("\n⏹️  interrupted")
    finally:
        if torque_on:
            for b in buses.values():
                try:
                    b.disable_torque(num_retry=3)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️  could not disable torque on a follower: {e}")
        data_manager.request_shutdown()
        for c in captures:
            try:
                c.stop()
            except Exception:  # noqa: BLE001
                pass


def _infer(
    policy, preprocessor, postprocessor, build_batch, state, images, task, device, torch
) -> np.ndarray:
    """One policy step: observation -> 12-D action (numpy). Mirrors eval_sim_policy."""
    batch = build_batch(state, images, task, device)
    with torch.no_grad():
        batch = preprocessor(batch)
        action = policy.select_action(batch)
        action = postprocessor(action)
    return np.asarray(action.squeeze(0).to("cpu")).astype(float)


if __name__ == "__main__":
    main()
