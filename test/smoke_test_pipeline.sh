#!/usr/bin/env bash
# =====================================================================
# Training + evaluation PIPELINE smoke test.
#
# Purpose: prove the LeRobot train -> checkpoint -> eval plumbing end to
# end on THIS machine before spending GPU hours on the real pi0.5 + LIBERO
# run (which needs a big GPU — see README "Training & evaluation").
#
# It deliberately uses a SMALL diffusion policy on a FEW episodes of the
# lightweight PushT dataset, so it runs in minutes and will not freeze a
# laptop. Everything lands in ONE run directory (checkpoints + eval).
#
#   bash test/smoke_test_pipeline.sh                 # defaults (below)
#   bash test/smoke_test_pipeline.sh --steps 100 --eval-episodes 3
#   bash test/smoke_test_pipeline.sh --device cpu    # force CPU
#
# For the real LIBERO pipeline (heavy) see the commands in the README.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (tiny on purpose) --------------------------------------
POLICY="diffusion"                 # small non-VLA policy; won't OOM a laptop
DATASET="lerobot/pusht"            # ~1 GB, downloads once
ENV="pusht"                        # matching lightweight sim for eval
EPISODES="[0,1,2,3]"               # only load a few episodes -> fast
STEPS=20                           # a handful of optimizer steps (CPU-friendly)
BATCH=4                            # small: fits CPU RAM, faster per step
EVAL_EPISODES=1                    # a PushT rollout is 300 steps (~5 min on CPU)
DEVICE=""                          # empty => LeRobot auto-selects (cuda/mps/cpu)
RUN_NAME="smoke_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
    case "$1" in
        --policy) POLICY="$2"; shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        --env) ENV="$2"; shift 2;;
        --episodes) EPISODES="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --eval-episodes) EVAL_EPISODES="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        -h|--help) sed -n '2,20p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

# ---- environment -----------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"   # PushT/LIBERO offscreen render

OUT_ROOT="${SO101_OUTPUT_DIR:-$REPO_ROOT/outputs}"
RUN_DIR="$OUT_ROOT/pipeline_smoke/$RUN_NAME"
# NOTE: do NOT create $RUN_DIR here — lerobot-train refuses to write into
# an existing output_dir (guards against clobbering a real run). It
# creates $RUN_DIR itself; we move logs in afterwards.
if [ -e "$RUN_DIR" ]; then
    echo "❌ Run dir already exists: $RUN_DIR"
    echo "   Pass a fresh --run-name (default is timestamped)."
    exit 2
fi
mkdir -p "$(dirname "$RUN_DIR")"
TRAIN_LOG="$(mktemp)"

# Device: if the user didn't force one, auto-pick. A small GPU (e.g. a
# 4 GB laptop) OOMs even on the tiny diffusion policy once Adam states are
# allocated, so require a comfortable VRAM budget before using CUDA;
# otherwise fall back to CPU (slower but safe — this is only a smoke test).
if [ -z "$DEVICE" ]; then
    DEVICE="$(python - <<'PY'
import torch
MIN_GB = 8.0
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/1e9 >= MIN_GB:
    print("cuda")
elif torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
    echo "ℹ️  Auto-selected device: $DEVICE (override with --device)."
    [ "$DEVICE" = "cpu" ] && echo "   (GPU absent or <8 GB VRAM — CPU is slower but won't OOM/freeze.)"
fi
DEV_ARG=(--policy.device="$DEVICE")

# ---- banner + time estimate ------------------------------------------
cat <<BANNER

======================================================================
 PIPELINE SMOKE TEST  (train -> checkpoint -> eval)
----------------------------------------------------------------------
 policy   : $POLICY            (small; validates the plumbing, not skill)
 dataset  : $DATASET  episodes $EPISODES
 train    : $STEPS steps, batch $BATCH
 eval     : $ENV, $EVAL_EPISODES episodes
 device   : ${DEVICE:-auto}
 output   : $RUN_DIR
----------------------------------------------------------------------
 Expected wall time (device: ${DEVICE}):
   GPU  : ~3-6 min   |   CPU: ~8-20 min  (a 263M policy on CPU is slow)
   + a one-time ~1 GB $DATASET download on the very first run.
 Progress bars are shown by lerobot-train / lerobot-eval below.
======================================================================

BANNER

START=$(date +%s)
phase() { echo; echo "### [$(date +%H:%M:%S)] $1"; echo; }
fail() {
    echo; echo "❌ Smoke test FAILED during: $1"
    [ -s "$TRAIN_LOG" ] && [ ! -d "$RUN_DIR/logs" ] && cp "$TRAIN_LOG" "/tmp/smoke_train.log" 2>/dev/null && echo "   train log: /tmp/smoke_train.log"
    [ -d "$RUN_DIR/logs" ] && echo "   logs in $RUN_DIR/logs"
    exit 1
}

# ---- preflight (quick, with a small progress bar) --------------------
phase "Preflight checks"
python - "$POLICY" "$ENV" <<'PY' || fail "preflight"
import importlib.util as u, sys
from tqdm import tqdm
policy, env = sys.argv[1], sys.argv[2]
checks = [
    ("lerobot", lambda: u.find_spec("lerobot")),
    ("torch",   lambda: u.find_spec("torch")),
    (f"policy:{policy}", lambda: u.find_spec(f"lerobot.policies.{policy}")),
    (f"env:gym_{env}", lambda: u.find_spec(f"gym_{env}")),
]
ok = True
for name, fn in tqdm(checks, desc="preflight", ncols=70):
    present = bool(fn())
    if not present:
        tqdm.write(f"  ✗ missing: {name}")
        ok = False
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tqdm.write(f"  compute: {dev}")
if not ok:
    tqdm.write("  Run install.sh — a dependency is missing.")
    sys.exit(1)
PY

# ---- phase 1: train --------------------------------------------------
phase "Phase 1/2 — training ($STEPS steps) -> $RUN_DIR/checkpoints"
# --policy.pretrained_backbone_weights=null: skip the torchvision ImageNet
# download (flaky CDN hash checks) — a smoke test only needs the wiring, so
# a randomly-initialised vision backbone is fine. Applies to diffusion; the
# flag is harmless for policies without a torchvision backbone.
BACKBONE_ARG=(); [ "$POLICY" = "diffusion" ] && BACKBONE_ARG=(--policy.pretrained_backbone_weights=null)
lerobot-train \
    --policy.type="$POLICY" \
    --dataset.repo_id="$DATASET" \
    --dataset.episodes="$EPISODES" \
    --output_dir="$RUN_DIR" \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers=2 \
    --save_freq="$STEPS" \
    --log_freq=10 \
    --env_eval_freq=0 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    "${BACKBONE_ARG[@]}" \
    "${DEV_ARG[@]}" 2>&1 | tee "$TRAIN_LOG" || fail "training"

CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
[ -d "$CKPT" ] || fail "training (no checkpoint written at $CKPT)"
mkdir -p "$RUN_DIR/logs"; mv "$TRAIN_LOG" "$RUN_DIR/logs/train.log"
echo "✓ checkpoint: $CKPT"

# ---- phase 2: eval ---------------------------------------------------
phase "Phase 2/2 — evaluation ($EVAL_EPISODES episodes) -> $RUN_DIR/eval"
lerobot-eval \
    --policy.path="$CKPT" \
    --env.type="$ENV" \
    --eval.n_episodes="$EVAL_EPISODES" \
    --eval.batch_size=1 \
    --output_dir="$RUN_DIR/eval" \
    "${DEV_ARG[@]}" 2>&1 | tee "$RUN_DIR/logs/eval.log" || fail "evaluation"

# ---- summary ---------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
phase "Done in $((ELAPSED/60))m $((ELAPSED%60))s"
python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
info = run / "eval" / "eval_info.json"
if info.exists():
    d = json.loads(info.read_text())
    overall = d.get("overall", {})
    sr = overall.get("pc_success")
    print(f"  eval success rate (smoke, NOT meaningful): {sr}%  "
          f"over {overall.get('n_episodes','?')} episode(s)")
print("  Run directory layout:")
for p in sorted(run.rglob("*")):
    if p.is_file() and p.stat().st_size > 0:
        rel = p.relative_to(run)
        if len(rel.parts) <= 3:
            print(f"    {rel}")
PY

echo
echo "✅ PIPELINE SMOKE TEST PASSED — train + eval wiring works."
echo "   Next: the real pi0.5 + LIBERO run (big GPU) — see README."
