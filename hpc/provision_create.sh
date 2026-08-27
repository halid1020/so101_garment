#!/usr/bin/env bash
# =====================================================================
# One-time provisioning of the sim-VLA training/eval environment on the
# KCL CREATE cluster (or any Slurm HPC login node with internet access).
#
# This is the SIM-ONLY subset of ../install.sh: it builds the venv,
# installs LeRobot from source at the pinned commit with the training
# extras, adds this repo's requirements, and warms the torchvision
# backbone cache so that offline GPU compute nodes never try to download
# weights mid-train.
#
# It deliberately SKIPS install.sh steps 4-6 (meta_quest_teleop, adb,
# openscad, pre-commit): none are needed for simulation training/eval,
# and their `sudo apt install` steps abort on a cluster with no sudo.
#
# Run ONCE, on a CREATE *login* node (compute nodes have no internet):
#     bash hpc/provision_create.sh
# It needs a Python >=3.12: it uses one on PATH if present (e.g. a
# `module load`ed Python), else bootstraps a standalone CPython via `uv`
# (CREATE has no >=3.12 module -- see hpc/README.md).
#
# Idempotent: safe to re-run. The pins below MUST match install.sh --
# keep them in sync if install.sh changes.
# =====================================================================
set -euo pipefail

# ---- pins (keep identical to ../install.sh) --------------------------
LEROBOT_COMMIT="3dd19d043e2f3fe5673b13ea0ebe4f31884c0797"
# `fastwam` is listed but INERT until the pinned commit above ships
# src/lerobot/policies/fastwam -- it is not in 3dd19d04 (2026-06-27), only
# upstream. MEASURED: pip ignores an extra the checkout does not define
# (`pip install -e "../lerobot[fastwam]"` exits 0 against this pin), so
# naming it now costs nothing and makes the bump one line, not two files.
LEROBOT_EXTRAS="feetech,dataset,pi,libero,pusht,training,diffusion,peft,fastwam"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# LeRobot requires Python >=3.12, and a venv built on an older python3
# silently fails its editable install (see CLAUDE.md: the
# Anaconda-python3.9-ahead-of-PATH pitfall). We resolve a >=3.12
# interpreter into $PYTHON: first anything already on PATH (e.g. a
# `module load`ed Python), otherwise a self-contained CPython bootstrapped
# with `uv`. CREATE has NO >=3.12 module (`module avail python` shows only
# 3.10/3.11), so the uv path is the normal one there: the login node has
# internet, and uv stores the interpreter under ~/.local (shared home), so
# the compute node sees it through the venv that references it.
py_ok() {  # return 0 iff "$1" is a runnable Python >=3.12
    command -v "$1" >/dev/null 2>&1 || return 1
    local v
    v="$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || return 1
    case "$v" in 3.1[2-9] | 3.[2-9][0-9]) return 0 ;; *) return 1 ;; esac
}

PYTHON=""
for cand in python3.13 python3.12 python3; do
    if py_ok "$cand"; then PYTHON="$(command -v "$cand")"; break; fi
done

if [ -z "$PYTHON" ]; then
    echo "=> No Python >=3.12 on PATH; bootstrapping a standalone one with uv..."
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    # uv installs to ~/.local/bin (or ~/.cargo/bin on older installers).
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || {
        echo "❌ uv not found after install -- is curl/internet available on this node?" >&2
        exit 1
    }
    uv python install 3.12
    PYTHON="$(uv python find 3.12)"
fi
echo "=> Using interpreter: ${PYTHON} ($("$PYTHON" --version))"

# 1. Virtual environment ------------------------------------------------
if [ ! -d venv ]; then
    echo "=> Creating Python venv..."
    "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "=> Upgrading pip..."
pip install --upgrade pip

# 2. LeRobot from source (parallel ../lerobot), pinned commit + extras --
echo "=> Installing LeRobot ${LEROBOT_COMMIT:0:10} [${LEROBOT_EXTRAS}]..."
cd "$REPO_ROOT/.."
if [ ! -d lerobot ]; then
    echo "=> Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git
fi
cd lerobot
git fetch --quiet origin || true
git checkout --quiet "$LEROBOT_COMMIT"
# egl_probe / hf-egl-probe (LIBERO deps) need cmake<4 and a no-isolation
# pre-build -- identical to install.sh:71-75.
pip install "cmake<4" setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    pip install --no-build-isolation egl_probe hf-egl-probe ||
    echo "⚠️  egl_probe pre-build skipped (already satisfied)."
pip install -e ".[${LEROBOT_EXTRAS}]"

# 3. This repo's extra requirements ------------------------------------
cd "$REPO_ROOT"
echo "=> Installing project requirements..."
pip install -r requirements.txt

# 4. Warm the vision-backbone cache (CRITICAL for offline compute nodes)-
# Both ACT and Diffusion Policy build a torchvision ResNet18 whose default
# (ImageNet) weights download from a CDN on first use. Compute nodes have
# no internet, so fetch them now into ~/.cache/torch/hub/checkpoints,
# which is on shared home and therefore visible to the compute node.
echo "=> Warming torchvision ResNet18 backbone cache..."
python - <<'PY'
import torchvision.models as m

# ResNet18_Weights.DEFAULT == IMAGENET1K_V1, the file LeRobot's default
# ACT/Diffusion backbone config requests (resnet18-f37072fd.pth).
m.resnet18(weights=m.ResNet18_Weights.DEFAULT)
print("✓ ResNet18 weights cached under ~/.cache/torch/hub/checkpoints")
PY

# 5. Pre-stage the pi0.5 base + its tokenizer (only if pi05 rows are trained)-
# pi0.5 is a FINETUNE: lerobot-train needs the base weights on disk before it
# starts, and a compute node cannot fetch them. ~14.5 GB, so this is opt-in --
# `SO101_STAGE_PI05=1 bash hpc/provision_create.sh` -- and it goes to the shared
# HF cache the job reads with HF_HUB_OFFLINE=1.
#
# TWO repos are needed, and only one of them is easy:
#   * lerobot/pi05_base is NOT licence-gated (checked 2026-08-24), so it is a
#     plain download -- but a big one, and a login-node watchdog that kills it
#     mid-flight leaves a `.incomplete` blob behind. huggingface_hub picks a new
#     temp name each run, so a later attempt does NOT resume, and nothing
#     complains: the half-download simply sits there until a compute node tries
#     to use it. We therefore ASSERT the snapshot afterwards.
#   * google/paligemma-3b-pt-224 IS licence-gated (`gated: manual`) and supplies
#     pi0.5's tokenizer/processor, which LeRobot builds BY NAME. Without it the
#     job dies at startup with "Failed to instantiate processor step
#     'tokenizer_processor' ... couldn't connect to https://huggingface.co".
#     It cost three array tasks in August 2026, hours in, on offline nodes.
PI05_REPO="${SO101_PI05_REPO:-lerobot/pi05_base}"
PI05_TOKENIZER="${SO101_PI05_TOKENIZER:-google/paligemma-3b-pt-224}"
if [ "${SO101_STAGE_PI05:-0}" = "1" ]; then
    echo "=> Pre-staging ${PI05_REPO} (~14.5 GB) into the HF cache..."
    python - "$PI05_REPO" <<'PY' || exit 1
import sys
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

repo = sys.argv[1]
path = Path(snapshot_download(repo_id=repo))

# --- assert the snapshot is WHOLE, every time -------------------------------
# snapshot_download returning is not proof: a previous run killed mid-file
# leaves the small JSONs plus a stale `.incomplete` blob, and this call is
# happy to hand that back. Fail here, on a login node, rather than 17 hours
# into a GPU job.
repo_dir = path.parent.parent          # .../models--<org>--<name>
stale = sorted(repo_dir.glob("blobs/*.incomplete"))
if stale:
    print(
        f"❌ {repo} is only PART-downloaded: {len(stale)} unfinished blob(s)\n"
        f"   under {repo_dir / 'blobs'}. Delete them and re-run this script;\n"
        "   huggingface_hub renames its temp file each attempt, so it will NOT\n"
        "   resume on its own.",
        file=sys.stderr,
    )
    sys.exit(1)

weights = path / "model.safetensors"
if not weights.exists():
    print(f"❌ {repo} cached at {path} but model.safetensors is missing.", file=sys.stderr)
    sys.exit(1)
have = weights.stat().st_size
try:
    want = get_hf_file_metadata(hf_hub_url(repo, "model.safetensors")).size
except Exception:  # no network here is not this check's problem
    want = None
if want is not None and have != want:
    print(
        f"❌ {repo}: model.safetensors is {have} bytes, the Hub says {want}.\n"
        "   Delete the blob and re-run; a truncated file only fails at train time.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"✓ pi0.5 base cached whole at {path} ({have / 1e9:.1f} GB)")
print("  Train against it by name; HF_HUB_OFFLINE=1 resolves it from this cache.")
PY

    echo "=> Staging the pi0.5 tokenizer (${PI05_TOKENIZER}, ~22 MB)..."
    python - "$PI05_TOKENIZER" <<'PY' || exit 1
import os
import sys

repo = sys.argv[1]

# Weights are NOT wanted: pi0.5 carries its own. Only the tokenizer/processor
# JSONs and the sentencepiece model, which is why this is megabytes not 11 GB.
WANTED = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

GATE_HELP = f"""❌ Could not stage {repo}, which supplies pi0.5's tokenizer.

   This repo is licence-gated (`gated: manual`): the Hub answers 401 for its
   files unless the request carries a token whose account has been GRANTED
   access. A CREATE login node has internet but no token, so it cannot fetch
   this on its own. Either:

     (a) accept the terms at https://huggingface.co/{repo} with your HF
         account, then `hf auth login` on this node and re-run; or

     (b) copy the tokenizer from a machine that already has it (~22 MB) --
         no token travels, only the files:

           # on the machine that HAS it
           H=~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224
           REV=$(cat $H/refs/main)
           cd $H && tar -czf /tmp/pg_tok.tgz refs/main \\
             $(for f in snapshots/$REV/*; do
                 case "$f" in *.safetensors|*.gguf) continue;; esac
                 echo "$f blobs/$(basename $(readlink -f $f))"
               done)
           # then, on this node
           mkdir -p $H && tar -xzf pg_tok.tgz -C $H

   Do NOT skip this: pi05 rows will queue, run for hours on an offline GPU
   node, and only then fail building the policy."""


def usable() -> bool:
    """Can the processor actually be built with no network? The only test."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(repo)
        return True
    except Exception:
        return False
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)


if usable():
    print(f"✓ {repo} tokenizer already staged and loads offline.")
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, allow_patterns=WANTED)
except Exception as exc:
    print(GATE_HELP, file=sys.stderr)
    print(f"\n   (underlying error: {type(exc).__name__}: {exc})", file=sys.stderr)
    sys.exit(1)

if not usable():
    print(GATE_HELP, file=sys.stderr)
    print("\n   (files downloaded, but the processor still will not build)", file=sys.stderr)
    sys.exit(1)
print(f"✓ {repo} tokenizer staged; builds offline.")
PY
else
    echo "=> Skipping the pi0.5 base (${PI05_REPO}, ~14.5 GB) and its tokenizer."
    echo "   Training a pi05 row needs both: re-run with SO101_STAGE_PI05=1"
fi

echo "=================================================="
echo "✓ CREATE provisioning complete (sim-only subset)."
echo "  Next: stage the dataset to scratch, then submit the job."
echo "  See hpc/README.md."
echo "=================================================="
