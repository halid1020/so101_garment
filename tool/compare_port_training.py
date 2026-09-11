#!/usr/bin/env python3
"""Does this repo's port of a policy TRAIN like the LeRobot one it came from?

The package makes a strong claim about its ports and, until this, measured only
half of it. ``test/unit/test_policy_ports.py`` proves the nine files are the
same bytes as upstream after four mechanical substitutions, and
``test/integration/test_policy_ports_checkpoints.py`` proves the same weights
produce the same actions, the same loss and the same gradients. What none of
that touches is everything ``lerobot-train`` assembles AROUND the model: the
pre/post-processor pipeline the factory builds from the config class name, the
optimiser and scheduler presets the config carries, the dataloader, and the
plugin discovery that has to resolve a type this repo defines.

So this runs the real trainer twice, on the same data, from the same seed, and
compares the loss the two implementations logged step for step. A port that
diverges here is a port in name only, however identical its source is.

**It compares what the log holds, which is three decimal places.**
``MetricsTracker.__str__`` formats the loss ``:.3f``, so an agreement here is an
agreement to a thousandth -- broad coverage at coarse precision. The integration
test is the other way round: one batch, but the loss and every gradient compared
bit for bit. Neither replaces the other, and a divergence too small to show here
would still show there.

    venv/bin/python tool/compare_port_training.py \\
        --dataset-root /media/hdd/so101/cube-pnp-new --policy act --steps 200

Exits non-zero on any disagreement, and prints where the first one was.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from actoris_harena.training.matrix import PORTED_FROM  # noqa: E402
from actoris_harena.training.progress import parse_log  # noqa: E402

#: Everything that has to be pinned for two runs to be comparable at all. The
#: seed is lerobot's own default, restated because a default that moved would
#: turn this into a test of nothing; cudnn_deterministic and a single-process
#: loader remove the two remaining sources of run-to-run variation.
DETERMINISM = (
    "--seed=1000",
    "--cudnn_deterministic=true",
    "--num_workers=0",
)


def train_once(
    policy: str,
    dataset_root: Path,
    out_dir: Path,
    steps: int,
    batch: int,
    device: str,
    log_freq: int,
) -> str:
    """Run lerobot-train once and return its log. Raises on a non-zero exit."""
    args = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_root.name}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        f"--output_dir={out_dir}",
        f"--policy.type={policy}",
        f"--policy.device={device}",
        f"--steps={steps}",
        f"--batch_size={batch}",
        f"--log_freq={log_freq}",
        # Nothing is being kept, and a save is minutes of disk for a run that
        # exists only to be read out of its own log.
        f"--save_freq={steps}",
        "--save_checkpoint=false",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        *DETERMINISM,
    ]
    if policy in PORTED_FROM or policy.startswith("so101_"):
        args.append("--policy.discover_packages_path=so101_policies")
    if policy.endswith("diffusion"):
        # The torchvision ImageNet weights come off a CDN whose hash check is
        # flaky, and an initialisation that differs between the two runs would
        # be read as a difference between the implementations.
        args.append("--policy.pretrained_backbone_weights=null")

    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'src'}"}
    proc = subprocess.run(
        args, capture_output=True, text=True, env=env, errors="replace"
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(log.replace("\r", "\n").splitlines()[-25:])
        raise SystemExit(
            f"❌ training {policy} failed (exit {proc.returncode}):\n{tail}"
        )
    return log


def losses(log: str) -> "list[tuple[int, float]]":
    """``[(step, loss)]`` from a driver log, using the console's own parser."""
    return [
        (point["step"], point["loss"])
        for point in parse_log(log)["points"]
        if "loss" in point
    ]


def compare(left: "list[tuple[int, float]]", right: "list[tuple[int, float]]") -> int:
    """Print the pairing and return the number of steps that disagree."""
    if not left or not right:
        raise SystemExit("❌ one of the runs logged no loss at all — nothing to compare")
    if [s for s, _ in left] != [s for s, _ in right]:
        raise SystemExit(
            f"❌ the two runs logged different steps: {[s for s, _ in left][:5]}… "
            f"against {[s for s, _ in right][:5]}…"
        )
    print(f"\n  {'step':>8}  {'lerobot':>12}  {'this repo':>12}  {'difference':>12}")
    disagreements = 0
    for (step, a), (_, b) in zip(left, right):
        mark = "" if a == b else "   ← differs"
        if a != b:
            disagreements += 1
        print(f"  {step:>8}  {a:>12.6f}  {b:>12.6f}  {abs(a - b):>12.3e}{mark}")
    return disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--policy",
        default="act",
        help="the LeRobot policy to compare against its port (default: act)",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--keep", action="store_true", help="keep both run directories and their logs"
    )
    args = parser.parse_args()

    port = f"so101_{args.policy}"
    if PORTED_FROM.get(port) != args.policy:
        raise SystemExit(
            f"❌ {args.policy!r} has no port in this repo. "
            f"Ported policies: {', '.join(sorted(PORTED_FROM))}"
        )
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"❌ no dataset at {args.dataset_root}")

    work = Path(tempfile.mkdtemp(prefix="port-parity-"))
    print(
        f"Comparing {args.policy} against {port}\n"
        f"  dataset  : {args.dataset_root}\n"
        f"  schedule : {args.steps} steps, batch {args.batch}, on the {args.device}\n"
        f"  pinned   : {' '.join(DETERMINISM)}\n"
        f"  scratch  : {work}"
    )
    logs = {}
    for name in (args.policy, port):
        started = time.time()
        print(f"\n▶ training {name} …", flush=True)
        logs[name] = train_once(
            name,
            args.dataset_root,
            work / name,
            args.steps,
            args.batch,
            args.device,
            args.log_freq,
        )
        (work / f"{name}.log").write_text(logs[name], errors="replace")
        print(f"  done in {time.time() - started:.0f}s")

    disagreements = compare(losses(logs[args.policy]), losses(logs[port]))
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"\n  logs kept in {work}")

    if disagreements:
        raise SystemExit(
            f"\n❌ {disagreements} logged step(s) differ. The port is not training "
            "like the policy it was ported from — compare the two logs' "
            "configuration dumps before trusting any result measured with it."
        )
    print(
        f"\n✅ {args.policy} and {port} logged an identical loss at every one of "
        f"{len(losses(logs[port]))} points."
    )


if __name__ == "__main__":
    main()
