#!/usr/bin/env python
"""Point a checkpoint at the other implementation of the same policy.

A checkpoint names the policy type that trained it, and that name decides which
class loads it. The ports in ``src/so101_policies/`` keep the upstream module
tree, so the WEIGHTS fit either class -- only the name in ``config.json``
disagrees. This rewrites just that name.

Two uses, both real:

* **Finetuning from a pretrained base.** ``lerobot/pi05_base`` says ``pi05``, so
  ``--policy.path`` to it loads LeRobot's class however the run was asked for.
  Retargeting it once gives ``so101_pi05`` a base of its own.
* **Reading a finished run through the port.** The 80 000-step ACT checkpoint
  says ``act``; this lets our copy replay it without retraining anything.

The weights are SYMLINKED, never copied -- ``lerobot/pi05_base`` is about
14.5 GB and a second copy of it buys nothing.

    venv/bin/python tool/retarget_checkpoint.py \\
        --checkpoint lerobot/pi05_base --to so101_pi05 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actoris_harena.training.matrix import PORTED_FROM  # noqa: E402


def resolve(checkpoint: str) -> Path:
    """A local directory, or a Hub id already in the cache. Never downloads.

    A download here would be a multi-gigabyte surprise in the middle of a run
    that has already reserved a GPU, so an uncached id is refused instead.
    """
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(checkpoint, local_files_only=True)).resolve()
    except Exception as exc:  # noqa: BLE001 - the cause is reported, whatever it is
        raise SystemExit(
            f"❌ {checkpoint} is neither a local directory nor a cached Hub "
            f"snapshot ({exc}). Download it first; this tool never will."
        ) from exc


def compatible(source_type: str, target_type: str) -> bool:
    """Are these two names the same policy under two implementations?"""
    pairs = {(ported, upstream) for ported, upstream in PORTED_FROM.items()}
    return (target_type, source_type) in pairs or (source_type, target_type) in pairs


def retarget(source: Path, target_type: str, out: Path, force: bool = False) -> Path:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"❌ no config.json in {source}")
    config = json.loads(config_path.read_text())
    source_type = config.get("type", "")

    if source_type == target_type:
        return source
    if not compatible(source_type, target_type):
        raise SystemExit(
            f"❌ {source_type} and {target_type} are not two implementations of "
            f"one policy, so the weights would not fit. Ported pairs: "
            + ", ".join(f"{a}<->{b}" for a, b in sorted(PORTED_FROM.items()))
        )

    if out.exists() and not force:
        existing = out / "config.json"
        if (
            existing.is_file()
            and json.loads(existing.read_text()).get("type") == target_type
        ):
            return out
        raise SystemExit(
            f"❌ {out} exists and is not a {target_type} checkpoint; pass --force"
        )

    out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name == "config.json":
            continue
        link = out / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        # Relative would break the moment either directory moved; these are
        # scratch paths on a cluster and they do move.
        os.symlink(entry.resolve(), link)

    config["type"] = target_type
    (out / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="directory or cached Hub id"
    )
    parser.add_argument("--to", required=True, help="policy type to retarget to")
    parser.add_argument(
        "--out", required=True, help="where to write the retargeted view"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print only the resulting path, for a shell to capture",
    )
    args = parser.parse_args()

    result = retarget(
        resolve(args.checkpoint), args.to, Path(args.out).expanduser(), args.force
    )
    if args.print_path:
        print(result)
    else:
        print(f"✅ {args.checkpoint} readable as {args.to} at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
