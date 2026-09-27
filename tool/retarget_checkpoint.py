#!/usr/bin/env python
"""Point a checkpoint at the other implementation of the same policy.

A checkpoint names the policy type that trained it, and that name decides which
class loads it. The ports in ``actoris_harena/policies/`` keep the upstream module
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


#: The shim's contract, in one line. The policies were renamed `so101_*` ->
#: `harena_*`, and the old names survive as registry aliases because they are
#: written into Slurm scripts already submitted and into `train_config.json`
#: files a resuming job reads. A checkpoint on disk right now says
#: `so101_pi05_crop`; the run matrix says `harena_pi05_crop`. Both must reach
#: the same chain or a retarget refuses a pair it should accept.
_LEGACY_PREFIX = "so101_"
_CANONICAL_PREFIX = "harena_"


def _canonical(policy: str) -> str:
    """The name the run matrix knows, given either spelling."""
    from actoris_harena.training.matrix import POLICIES

    if policy in POLICIES or not policy.startswith(_LEGACY_PREFIX):
        return policy
    renamed = _CANONICAL_PREFIX + policy[len(_LEGACY_PREFIX) :]
    return renamed if renamed in POLICIES else policy


def _ancestry(policy: str) -> "list[str]":
    """A policy and everything it is a variant of, nearest first.

    `so101_pi05_crop -> so101_pi05 -> pi05`. The `seen` guard is not paranoia:
    a mis-typed entry pointing a name at itself would otherwise hang here rather
    than fail.
    """
    policy = _canonical(policy)
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


def retarget(
    source: Path,
    target_type: str,
    out: Path,
    force: bool = False,
    crop: "tuple[float, float] | None" = None,
) -> Path:
    """Rewrite the type, and give the base the pipeline its target declares.

    ``crop`` is the run's ``--policy.tactile_crop``, when it overrides the
    target's default. It has to arrive here, not only at the trainer: a
    finetune loads the base's SAVED pipeline, so a crop step written with the
    default fraction is the crop the run trains with, whatever the command line
    said. MEASURED 2026-09-27: a four-edge pi0.5 smoke run saved
    ``fraction [0.8, 1.0]`` -- the rows-only default -- under a
    ``--policy.tactile_crop=[0.8,0.8]`` override.
    """
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
            # Reused, but the pipeline is rewritten from the source every time,
            # so a base made for one crop cannot serve a run asking for another.
            _carry_extra_processor_steps(source, out, target_type, crop)
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
    _carry_extra_processor_steps(source, out, target_type, crop)
    return out


def _carry_extra_processor_steps(
    source: Path,
    out: Path,
    target_type: str,
    crop: "tuple[float, float] | None" = None,
) -> None:
    """Put the target's own preprocessor steps into the retargeted base.

    THE BUG THIS EXISTS FOR, and it is silent. `make_pre_post_processors` begins
    `if pretrained_path:` and loads the SAVED pipeline off disk -- it never calls
    the policy's factory. Every pi0.5 finetune uses `--policy.path`, because the
    base carries the pretrained image slots, so
    `make_harena_pi05_crop_pre_post_processors` is never invoked and its crop
    step never exists. MEASURED: a 3-hour `so101_pi05_crop` run trained on
    UNCROPPED tactile images and scored bit-identically to plain pi0.5 --
    RMSE 16.044489 both, to every digit. It was a second baseline wearing the
    crop's name, and nothing in the run said so.

    So the retarget, which already exists to make a repo-local base, also writes
    the pipeline that base should carry. Only steps the target declares and the
    source lacks are added, at the FRONT -- the crop has to run before the
    rename, or pi0.5's slot names hide the cameras from it. The normalizer's
    safetensors stay symlinked and keep working because nothing after index 0
    moves.
    """
    config_file = out / "policy_preprocessor.json"
    if not config_file.exists() and not config_file.is_symlink():
        return
    try:
        pipeline = json.loads((source / "policy_preprocessor.json").read_text())
    except (OSError, ValueError):
        return

    extra = _target_only_steps(target_type, crop)
    if not extra:
        return
    present = {step.get("registry_name") for step in pipeline.get("steps", [])}
    additions = [s for s in extra if s["registry_name"] not in present]
    if not additions:
        return

    pipeline["steps"] = additions + list(pipeline.get("steps", []))
    if config_file.is_symlink():
        config_file.unlink()  # do not write through into the shared base
    config_file.write_text(json.dumps(pipeline, indent=4) + "\n")
    names = ", ".join(s["registry_name"] for s in additions)
    # STDERR. --print-path exists so a shell can capture the path in a
    # command substitution, and anything else on stdout is captured with it --
    # which is how a chatty line here became part of a Hub repo id and killed
    # the run: "Repo id must be in the form 'repo_name'...: '  preprocessor:
    # added so101_tactile_crop...'".
    print(
        f"  preprocessor: added {names} ahead of the base's own steps",
        file=sys.stderr,
    )


def _target_only_steps(
    target_type: str, crop_override: "tuple[float, float] | None" = None
) -> "list[dict]":
    """The serialised steps this policy adds over the one it is a variant of.

    Built from the target's CONFIG rather than by constructing the policy: the
    step's own `get_config` is the authority on its fields, and a retarget must
    not need 14 GB of weights loaded to decide what a pipeline looks like.
    """
    from actoris_harena.policies.loading import ensure_registered
    from lerobot.configs import PreTrainedConfig

    ensure_registered()
    try:
        config = PreTrainedConfig.get_choice_class(target_type)()
    except Exception:  # an unregistered or unconstructable target adds nothing
        return []

    crop = getattr(config, "tactile_crop", None)
    if crop is None:
        return []
    if crop_override is not None:
        crop = crop_override
    from actoris_harena.policies.common.tactile import (
        TACTILE_CAMERAS,
        HarenaTactileCropProcessorStep,
    )

    step = HarenaTactileCropProcessorStep(
        fraction=tuple(crop),
        cameras=tuple(getattr(config, "tactile_cameras", TACTILE_CAMERAS)),
    )
    return [
        {
            "registry_name": "so101_tactile_crop",
            "config": step.get_config(),
        }
    ]


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
        "--tactile-crop",
        default=None,
        help="HEIGHT,WIDTH fractions kept, when the run overrides the target's "
        "default crop (its --policy.tactile_crop)",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print only the resulting path, for a shell to capture",
    )
    args = parser.parse_args()

    crop = None
    if args.tactile_crop:
        height, width = (float(v) for v in args.tactile_crop.strip("[]").split(","))
        crop = (height, width)
    result = retarget(
        resolve(args.checkpoint),
        args.to,
        Path(args.out).expanduser(),
        args.force,
        crop=crop,
    )
    if args.print_path:
        print(result)
    else:
        print(f"✅ {args.checkpoint} readable as {args.to} at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
