#!/usr/bin/env bash
# =====================================================================
# Sim-VLA PIPELINE smoke test  (collect -> train -> eval, all three policies).
#
# Purpose: prove the twin-oracle collection -> LeRobot train -> checkpoint
# -> twin eval plumbing end to end on THIS machine, before the multi-day
# long run on a big GPU (test/system/long_vla_sim.sh). It validates the
# WIRING, not task skill: tiny datasets, ~20 optimiser steps, 1 eval
# rollout — success rate is NOT asserted.
#
# Rotation covers both datasets and all three policies in three runs:
#   ACT       on the handover dataset
#   Diffusion on the single   dataset
#   pi0.5     on the handover dataset (gemma_300m variant, CPU-sized)
#
#   bash test/system/smoke_vla_sim.sh                 # defaults (below)
#   bash test/system/smoke_vla_sim.sh --steps 40 --device cpu
#   bash test/system/smoke_vla_sim.sh --skip-collect  # reuse last datasets
#
# The real experiment (high success, big GPU) is long_vla_sim.sh.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (tiny on purpose) --------------------------------------
EPISODES=2                 # demos per task (direct oracle -> deterministic)
STEPS=20                   # a handful of optimiser steps (CPU-friendly)
BATCH=2                    # small: fits CPU RAM
CAM_W=320; CAM_H=240       # small frames -> fast collect/render/train
ORACLE="direct"            # deterministic + faster than real-time for a smoke
DEVICE=""                  # empty => auto-pick (<8 GB VRAM -> CPU)
RUN_NAME="vla_smoke_$(date +%Y%m%d_%H%M%S)"
SKIP_COLLECT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --episodes) EPISODES="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --camera-width) CAM_W="$2"; shift 2;;
        --camera-height) CAM_H="$2"; shift 2;;
        --oracle) ORACLE="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --skip-collect) SKIP_COLLECT=1; shift;;
        -h|--help) sed -n '2,22p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

# ---- environment -----------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-.:src}"
PY="$REPO_ROOT/venv/bin/python"

OUT_ROOT="${SO101_OUTPUT_DIR:-$REPO_ROOT/outputs}"
RUN_DIR="$OUT_ROOT/vla_sim_smoke/$RUN_NAME"
DATA_ROOT="$RUN_DIR/datasets"
if [ "$SKIP_COLLECT" = "0" ] && [ -e "$RUN_DIR" ]; then
    echo "❌ Run dir already exists: $RUN_DIR (pass a fresh --run-name)"; exit 2
fi
mkdir -p "$RUN_DIR/logs"

# Device: same rule as the pipeline smoke test — need comfortable VRAM to use
# CUDA (the diffusion/pi0.5 policies OOM a 4 GB laptop), else CPU.
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

cat <<BANNER

======================================================================
 SIM-VLA SMOKE TEST  (collect -> train -> eval; plumbing only)
----------------------------------------------------------------------
 collect : $EPISODES demos/task via the $ORACLE oracle, ${CAM_W}x${CAM_H}
 train   : $STEPS steps, batch $BATCH  (act, diffusion, pi05/gemma_300m)
 eval    : 1 rollout/policy through tool/eval_sim_policy.py
 device  : $DEVICE
 output  : $RUN_DIR
----------------------------------------------------------------------
 Expected wall time: ~15-45 min on CPU (+ one-time PaliGemma tokenizer
 download for pi0.5 on the first run).
======================================================================

BANNER

START=$(date +%s)
phase() { echo; echo "### [$(date +%H:%M:%S)] $1"; echo; }
fail() { echo; echo "❌ Sim-VLA smoke test FAILED during: $1"; exit 1; }

# ---- phase 0: preflight ----------------------------------------------
phase "Phase 0 — preflight"
"$PY" - <<'PY' || fail "preflight"
import importlib.util as u, sys
checks = ["lerobot", "torch", "lerobot.policies.act",
          "lerobot.policies.diffusion", "lerobot.policies.pi05"]
ok = all(u.find_spec(m) for m in checks)
missing = [m for m in checks if not u.find_spec(m)]
if missing:
    print("  ✗ missing:", missing); sys.exit(1)
import torch
print("  compute:", "cuda" if torch.cuda.is_available() else "cpu")
PY
# Twin assets must exist (payload scene builds from build/twin).
"$PY" -m sim_twin.verify >/dev/null 2>&1 || fail "twin assets (run: python -m sim_twin.verify)"
echo "  ✓ deps + twin assets present"
# pi0.5's tokenizer lives in a GATED HF repo (google/paligemma-3b-pt-224):
# it needs an authenticated account that accepted the licence. Without it,
# skip the pi0.5 cell rather than fail — ACT + Diffusion already prove the
# collect->train->eval wiring; prove pi0.5 on a machine with HF access.
PI05_OK="$("$PY" - <<'PY'
try:
    from huggingface_hub import auth_check
    auth_check("google/paligemma-3b-pt-224")
    print(1)
except Exception:
    print(0)
PY
)"
if [ "$PI05_OK" = "1" ]; then
    echo "  ✓ gated PaliGemma tokenizer accessible — pi0.5 cell enabled"
else
    echo "  ⚠️  no access to the gated PaliGemma tokenizer — pi0.5 cell will be"
    echo "     SKIPPED. To enable: venv/bin/hf auth login, then accept the"
    echo "     licence at https://huggingface.co/google/paligemma-3b-pt-224"
fi

# ---- phase 1: collect -------------------------------------------------
collect_task() {
    local task="$1"
    local root="$DATA_ROOT/$task"
    if [ "$SKIP_COLLECT" = "1" ] && [ -d "$root" ]; then
        echo "  ↷ reusing $root"; return 0
    fi
    "$PY" tool/collect_sim_dataset.py \
        --task "$task" --oracle "$ORACLE" --seeds simple --episodes "$EPISODES" \
        --repo-id "local/smoke_sim_$task" --root "$root" \
        --camera-width "$CAM_W" --camera-height "$CAM_H" \
        2>&1 | tee "$RUN_DIR/logs/collect_$task.log" || fail "collect ($task)"
    [ -d "$root" ] || fail "collect ($task): no dataset at $root"
}
phase "Phase 1 — collect ($EPISODES demos/task)"
collect_task single
collect_task handover

# ---- phase 2: train ---------------------------------------------------
# One tiny run per (policy, task) cell; each writes its own output_dir.
train_cell() {
    local policy="$1" task="$2"
    local root="$DATA_ROOT/$task"
    local out="$RUN_DIR/train/${policy}_${task}"
    # Idempotent resume: a finished cell is reused; a partial one (interrupted
    # run — dir exists but no checkpoint) is cleared, since lerobot-train
    # refuses to write into an existing output_dir.
    if [ -d "$out/checkpoints/last/pretrained_model" ]; then
        echo "  ↷ reusing checkpoint $out"; return 0
    fi
    [ -d "$out" ] && rm -rf "$out"
    local extra=()
    [ "$policy" = "diffusion" ] && extra+=(--policy.pretrained_backbone_weights=null)
    [ "$policy" = "pi05" ] && extra+=(--policy.paligemma_variant=gemma_300m)
    phase "Phase 2 — train $policy on $task ($STEPS steps)"
    lerobot-train \
        --policy.type="$policy" \
        --dataset.repo_id="local/smoke_sim_$task" \
        --dataset.root="$root" \
        --dataset.video_backend=pyav \
        --output_dir="$out" \
        --steps="$STEPS" \
        --batch_size="$BATCH" \
        --num_workers=2 \
        --save_freq="$STEPS" \
        --log_freq=10 \
        --env_eval_freq=0 \
        --wandb.enable=false \
        --policy.push_to_hub=false \
        --policy.device="$DEVICE" \
        "${extra[@]}" 2>&1 | tee "$RUN_DIR/logs/train_${policy}_${task}.log" \
        || fail "train ($policy/$task)"
    [ -d "$out/checkpoints/last/pretrained_model" ] \
        || fail "train ($policy/$task): no checkpoint"
}
# Rotation covers both datasets even when pi0.5 is skipped:
train_cell act handover
train_cell diffusion single
if [ "$PI05_OK" = "1" ]; then
    train_cell pi05 handover
else
    phase "Phase 2 — train pi05 SKIPPED (gated tokenizer inaccessible)"
fi

# ---- phase 3: eval ----------------------------------------------------
eval_cell() {
    local policy="$1" task="$2"
    local ckpt="$RUN_DIR/train/${policy}_${task}/checkpoints/last/pretrained_model"
    local out="$RUN_DIR/eval/${policy}_${task}.json"
    phase "Phase 3 — eval $policy on $task (1 rollout)"
    "$PY" tool/eval_sim_policy.py \
        --task "$task" --checkpoint "$ckpt" \
        --seeds simple --episodes 1 \
        --camera-width "$CAM_W" --camera-height "$CAM_H" \
        --device "$DEVICE" --out "$out" \
        2>&1 | tee "$RUN_DIR/logs/eval_${policy}_${task}.log" || fail "eval ($policy/$task)"
    [ -f "$out" ] || fail "eval ($policy/$task): no results.json"
}
eval_cell act handover
eval_cell diffusion single
EXPECTED=2
if [ "$PI05_OK" = "1" ]; then
    eval_cell pi05 handover
    EXPECTED=3
fi

# ---- summary ----------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
phase "Done in $((ELAPSED/60))m $((ELAPSED%60))s"
n_json=$(find "$RUN_DIR/eval" -name '*.json' | wc -l | tr -d ' ')
echo "  eval result files: $n_json (expected $EXPECTED)"
[ "$n_json" = "$EXPECTED" ] || fail "summary (expected $EXPECTED eval result files)"
[ "$PI05_OK" = "1" ] || echo "  ⚠️  pi0.5 cell was skipped (gated tokenizer) — rerun on an HF-authenticated machine for full coverage."
echo
echo "✅ SIM-VLA SMOKE TEST PASSED — collect + train + eval wiring works."
echo "   Success rate is NOT meaningful here; run long_vla_sim.sh on a GPU."
