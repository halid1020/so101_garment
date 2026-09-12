#!/usr/bin/env bash
# =====================================================================
# Real-data VLA PIPELINE smoke test  (train -> checkpoint -> load).
#
# Purpose: prove that a policy trains on a REAL collected LeRobot dataset
# (the dual-arm 12-D action + the recorded cameras) through the SAME LeRobot
# pipeline the sim side uses, before the long multi-policy run on a big GPU
# (hpc/create_real_vla.sbatch + test/system/long_vla_real.sh). It validates
# the WIRING, not task skill: a few optimiser steps, no success assertion.
#
# Unlike the sim smoke there is NO collect phase (the real dataset already
# exists) and NO on-robot eval (that needs the rig — see tool/run_policy.py).
# The eval step here just loads the trained checkpoint back, which is the same
# load the on-robot runner performs, so a green run means the checkpoint is
# usable for inference.
#
#   bash test/system/smoke_vla_real.sh --dataset-root /media/hdd/so101/short_fold
#   bash test/system/smoke_vla_real.sh --dir /media/hdd/so101 --name short_fold
#   bash test/system/smoke_vla_real.sh --dataset-root <ds> --steps 40 --device cpu
#
# The real experiment (high success, big GPU) is the HPC long run.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (tiny on purpose) --------------------------------------
POLICY="diffusion"         # the plumbing policy (matches the sim smoke)
POLICY_SET=0               # 1 once the user passes --policy (pins their choice)
STEPS=20                   # a handful of optimiser steps (CPU-friendly)
BATCH=2                    # small: fits CPU RAM
DEVICE=""                  # empty => auto-pick (<8 GB VRAM -> CPU)
DATASET_ROOT=""            # the dataset dir (…/so101/<name>)
DIR=""; NAME=""            # or --dir + --name
REPO_ID=""                 # defaults to the dataset dir name
RUN_NAME="vla_real_smoke_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2;;
        --dir) DIR="$2"; shift 2;;
        --name) NAME="$2"; shift 2;;
        --repo-id) REPO_ID="$2"; shift 2;;
        --policy) POLICY="$2"; POLICY_SET=1; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        -h|--help) sed -n '2,25p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

# ---- resolve the dataset ---------------------------------------------
if [ -z "$DATASET_ROOT" ]; then
    if [ -n "$DIR" ] && [ -n "$NAME" ]; then
        DATASET_ROOT="$DIR/$NAME"
    else
        echo "❌ give --dataset-root DIR, or --dir DIR --name NAME" >&2; exit 2
    fi
fi
[ -z "$REPO_ID" ] && REPO_ID="$(basename "$DATASET_ROOT")"
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "❌ no dataset at $DATASET_ROOT (is the data drive mounted?)" >&2; exit 2
fi

# ---- environment -----------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh"
fi
export PYTHONPATH="${PYTHONPATH:-.:src}"
# Purely local dataset; never reach for the Hub on an incomplete one.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
PY="$REPO_ROOT/venv/bin/python"

# Episodes deleted in the review tool are only MARKED until that dataset is
# compacted: they are still on disk, so training would still learn from takes the
# operator threw away. Ask the marker itself rather than parsing it here.
"$PY" - "$DATASET_ROOT" <<'PY' || exit 2
import sys
from actoris_harena.recording.dataset_edit import read_soft_deleted

marked = read_soft_deleted(sys.argv[1])
if marked:
    print(
        f"❌ {sys.argv[1]} has {len(marked)} episode(s) marked for deletion that are\n"
        f"   still on disk: {marked}\n"
        "   Open tool/rig_web.py and press 'Remove for good'\n"
        "   (or Restore them) before training.",
        file=sys.stderr,
    )
    sys.exit(2)
PY

OUT_ROOT="${SO101_OUTPUT_DIR:-$REPO_ROOT/outputs}"
RUN_DIR="$OUT_ROOT/vla_real_smoke/$RUN_NAME"
[ -e "$RUN_DIR" ] && { echo "❌ Run dir exists: $RUN_DIR (fresh --run-name)"; exit 2; }
mkdir -p "$RUN_DIR/logs"

# Device: same rule as the sim smoke — comfortable VRAM to use CUDA, else CPU.
if [ -z "$DEVICE" ]; then
    DEVICE="$("$PY" - <<'PY'
import torch
MIN_GB = 8.0
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/1e9 >= MIN_GB:
    print("cuda")
else:
    print("cpu")
PY
)"
    echo "ℹ️  Auto-selected device: $DEVICE (override with --device)."
fi

# The default diffusion policy is ~293M params: it OOMs a typical CPU box (and
# any <8 GB GPU). On CPU, fall back to the much lighter ~52M 'act' policy so the
# no-arg smoke still passes on a laptop; the user's explicit --policy always wins.
if [ "$DEVICE" = "cpu" ] && [ "$POLICY_SET" = "0" ] && [ "$POLICY" = "diffusion" ]; then
    POLICY="act"
    echo "ℹ️  CPU device: using the lighter 'act' policy (the 293M diffusion policy"
    echo "    OOMs most CPU machines). Override with --policy diffusion."
fi

cat <<BANNER

======================================================================
 REAL-VLA SMOKE TEST  (train -> checkpoint -> load; plumbing only)
----------------------------------------------------------------------
 dataset : $REPO_ID  ($DATASET_ROOT)
 train   : $POLICY, $STEPS steps, batch $BATCH
 load    : reload the checkpoint (the on-robot runner's load path)
 device  : $DEVICE
 output  : $RUN_DIR
----------------------------------------------------------------------
 On-robot evaluation is NOT part of this smoke (needs the rig); use
 tool/run_policy.py at the rig for that.
======================================================================

BANNER

START=$(date +%s)
phase() { echo; echo "### [$(date +%H:%M:%S)] $1"; echo; }
fail() { echo; echo "❌ Real-VLA smoke test FAILED during: $1"; exit 1; }

# ---- phase 0: preflight ----------------------------------------------
phase "Phase 0 — preflight"
"$PY" - "$POLICY" <<'PY' || fail "preflight"
import importlib.util as u, sys
policy = sys.argv[1]
checks = ["lerobot", "torch", f"lerobot.policies.{policy}"]
missing = [m for m in checks if not u.find_spec(m)]
if missing:
    print("  ✗ missing:", missing); sys.exit(1)
import torch
print("  compute:", "cuda" if torch.cuda.is_available() else "cpu")
PY
echo "  ✓ deps present; dataset at $DATASET_ROOT"

# ---- phase 1: train ---------------------------------------------------
OUT="$RUN_DIR/train/${POLICY}"
phase "Phase 1 — train $POLICY on $REPO_ID ($STEPS steps)"
extra=()
# Both diffusion and act carry a torchvision ResNet image backbone that otherwise
# pulls ImageNet weights from a flaky CDN; skip it — irrelevant to a plumbing check.
case "$POLICY" in
    diffusion | act) extra+=(--policy.pretrained_backbone_weights=null) ;;
esac
lerobot-train \
    --policy.type="$POLICY" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$OUT" \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers=2 \
    --save_freq="$STEPS" \
    --log_freq=10 \
    --env_eval_freq=0 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --policy.device="$DEVICE" \
    "${extra[@]}" 2>&1 | tee "$RUN_DIR/logs/train_${POLICY}.log" \
    || fail "train ($POLICY)"
CKPT="$OUT/checkpoints/last/pretrained_model"
[ -d "$CKPT" ] || fail "train ($POLICY): no checkpoint at $CKPT"

# ---- phase 2: load the checkpoint (the on-robot runner's load path) ---
phase "Phase 2 — reload the checkpoint"
"$PY" - "$CKPT" <<'PY' || fail "checkpoint load"
import sys
sys.path.insert(0, "tool")
from eval_sim_policy import load_policy
policy, pre, post, ptype = load_policy(sys.argv[1], "cpu")
print(f"  ✓ loaded a '{ptype}' policy from the checkpoint")
PY

# ---- summary ----------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
phase "Done in $((ELAPSED/60))m $((ELAPSED%60))s"
echo "✅ REAL-VLA SMOKE TEST PASSED — real dataset trains + the checkpoint loads."
echo "   Success rate is NOT meaningful here; run the HPC long run for a real policy."
