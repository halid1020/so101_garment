#!/usr/bin/env python
"""How big each checkpoint is, how much of it trained, and on what budget.

Total parameters count every weight the loaded policy holds, frozen parts
included (an image autoencoder, a text encoder, a pretrained backbone). Trained
parameters are the ones the optimiser updated:

* a LoRA finetune trains only its adapters, whose names carry ``lora_``;
* any other model trains what it marks ``requires_grad`` at construction, which
  is what the frozen autoencoders and text encoders switch off.

The training budget -- batch size, update steps, learning rate, optimiser -- is
read from the checkpoint's own ``train_config.json``, so the table describes the
run that produced the weights and not the defaults of today's code.

    python tool/count_parameters.py --checkpoint act=<dir> pi05=<dir> ... --out params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def count(module) -> "dict[str, int]":
    """``{"total": n, "trained": m}`` for one loaded policy."""
    total = trained = 0
    lora = any("lora_" in name for name, _ in module.named_parameters())
    for name, parameter in module.named_parameters():
        size = parameter.numel()
        total += size
        if (lora and "lora_" in name) or (not lora and parameter.requires_grad):
            trained += size
    return {"total": total, "trained": trained, "lora": lora}


def budget(checkpoint: Path) -> dict:
    """Batch, steps and optimiser settings of the run, from its train_config."""
    path = checkpoint / "train_config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text())
    optimizer = config.get("optimizer") or {}
    scheduler = config.get("scheduler") or {}
    peft = config.get("peft") or {}
    return {
        "batch_size": config.get("batch_size"),
        "steps": config.get("steps"),
        "seed": config.get("seed"),
        "eval_split": (config.get("dataset") or {}).get("eval_split"),
        "optimizer": optimizer.get("type"),
        "lr": optimizer.get("lr"),
        "weight_decay": optimizer.get("weight_decay"),
        "scheduler": scheduler.get("type"),
        "warmup_steps": scheduler.get("num_warmup_steps"),
        "lora_r": peft.get("r"),
        "lora_targets": peft.get("target_modules"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", required=True, help="name=dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import common  # noqa: F401  (configures the shared pipeline for this rig)
    from tool.eval_sim_policy import load_policy

    out = {}
    for item in args.checkpoint:
        name, _, directory = item.partition("=")
        checkpoint = Path(directory).expanduser().resolve()
        policy, _pre, _post, policy_type = load_policy(str(checkpoint), args.device)
        out[name] = {"type": policy_type, **count(policy), **budget(checkpoint)}
        print(f"{name:10s} {out[name]}", flush=True)
        del policy
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"💾 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
