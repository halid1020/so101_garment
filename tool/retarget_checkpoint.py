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

from actoris_harena.training.matrix import TWIN_OF  # noqa: E402


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


def _ancestry(policy: str) -> "list[str]":
    """A policy and everything it is a variant of, nearest first.

    `so101_pi05_crop -> so101_pi05 -> pi05`. The `seen` guard is not paranoia:
    a mis-typed entry pointing a name at itself would otherwise hang here rather
    than fail.
    """
    chain, seen = [policy], {policy}
    while True:
        nxt = TWIN_OF.get(chain[-1])
        if nxt is None or nxt in seen:
            return chain
        chain.append(nxt)
        seen.add(nxt)


def compatible(source_type: str, target_type: str) -> bool:
    """Are these two names the same model, whatever the config is called?

    Walks TWIN_OF rather than testing a flat pair list, because a variant
    may be several hops from the policy whose weights it loads. That is not a
    hypothetical: `so101_pi05_crop` carries `ported_from: None` -- it is a
    subclass, not a port -- so a membership test refused it, and a cropped pi0.5
    run died on the cluster before it trained a step.

    One direction is a real restriction and is kept. A config may be rebuilt as
    something FURTHER along its own chain or nearer to its root, because a
    variant only ever ADDS fields to its twin (the crop adds `tactile_crop` and
    `tactile_cameras`), and `loading.config_as` refuses a field the target does
    not declare. Two names on different chains share no weights and are refused.
    """
    return target_type in _ancestry(source_type) or source_type in _ancestry(
        target_type
    )


def retarget(source: Path, target_type: str, out: Path, force: bool = False) -> Path:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"❌ no config.json in {source}")
    config = json.loads(config_path.read_text())
    source_type = config.get("type", "")

    if source_type == target_type:
        return source
    if not compatible(source_type, target_type):
        # Name what IS accepted, which is every chain and not just the ported
        # pairs -- quoting the narrower list would send a reader looking for a
        # missing port when what they have is a variant of something else.
        raise SystemExit(
            f"❌ {source_type} and {target_type} are not the same model, so the "
            f"weights would not fit. {source_type} belongs to "
            + " <- ".join(reversed(_ancestry(source_type)))
            + f", and {target_type} to "
            + " <- ".join(reversed(_ancestry(target_type)))
            + ". A checkpoint may be retargeted along one chain, never across two."
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
