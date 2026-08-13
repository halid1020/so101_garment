#!/usr/bin/env bash
# =====================================================================
# Real-data VLA LONG training  (train -> checkpoint -> report), sized for a
# big GPU. The real-world counterpart of test/system/long_vla_sim.sh, minus
# the parts that only exist in simulation:
#   * NO oracle gate / collection — the dataset was teleoperated on the rig;
#   * NO in-loop evaluation — a real policy is evaluated ON THE ROBOT with
#     tool/run_policy_real.py at the rig, not on the cluster.
# For each policy (act, diffusion by default) it trains a long run on the
# staged real dataset, saves checkpoints, and writes a results.md pointing at
# them and their final training loss. pi0.5 is a deliberate follow-up (it needs
# the licence-gated base pre-staged + LoRA — see hpc/README.md).
#
#   bash test/system/long_vla_real.sh --dataset-root <ds>
#   bash test/system/long_vla_real.sh --dir /media/hdd/so101 --name towel_fold
#   bash test/system/long_vla_real.sh --dataset-root <ds> --only diffusion
#   bash test/system/long_vla_real.sh --dataset-root <ds> --act-steps 40000
#
# The script resumes: a finished checkpoint is reused, a partial train dir is
# cleared. Run it under Slurm (hpc/create_real_vla.sbatch) or tmux/nohup.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults --------------------------------------------------------
ONLY="act,diffusion"         # comma list of policies to train
DATASET_ROOT=""; DIR=""; NAME=""; REPO_ID=""
DEVICE=""                    # empty => auto-pick (>=8 GB VRAM -> cuda)
RUN_NAME="vla_real_long_$(date +%Y%m%d_%H%M%S)"
ACT_STEPS=80000;  ACT_BATCH=8;   ACT_SAVE=10000
DIFF_STEPS=100000; DIFF_BATCH=32; DIFF_SAVE=10000
DIFF_RESIZE_H=180; DIFF_RESIZE_W=240   # downsample cams for the diffusion encoder (3:4)
SKIP_TRAIN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2;;
        --dir) DIR="$2"; shift 2;;
        --name) NAME="$2"; shift 2;;
        --repo-id) REPO_ID="$2"; shift 2;;
        --only) ONLY="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --act-steps) ACT_STEPS="$2"; shift 2;;
        --diff-steps) DIFF_STEPS="$2"; shift 2;;
        --diffusion-resize) DIFF_RESIZE_H="$2"; DIFF_RESIZE_W="$3"; shift 3;;
        --skip-train) SKIP_TRAIN=1; shift;;
        -h|--help) sed -n '2,24p' "$0"; exit 0;;
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
    echo "❌ no dataset at $DATASET_ROOT (staged + drive mounted?)" >&2; exit 2
fi

# ---- environment -----------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh"
fi
export PYTHONPATH="${PYTHONPATH:-.:src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
PY="$REPO_ROOT/venv/bin/python"

OUT_ROOT="${SO101_OUTPUT_DIR:-$REPO_ROOT/outputs}"
RUN_DIR="$OUT_ROOT/vla_real_long/$RUN_NAME"
mkdir -p "$RUN_DIR/logs"

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
fi

echo "======================================================================"
echo " REAL-VLA LONG TRAINING"
echo "   dataset : $REPO_ID  ($DATASET_ROOT)"
echo "   policies: $ONLY   device: $DEVICE"
echo "   output  : $RUN_DIR"
echo "   NOTE: on-robot evaluation is a rig step (tool/run_policy_real.py)."
echo "======================================================================"

fail() { echo; echo "❌ Real-VLA long run FAILED during: $1"; exit 1; }
have() { case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }

train_cell() {
    local policy="$1"
    local out="$RUN_DIR/train/${policy}"
    if [ -d "$out/checkpoints/last/pretrained_model" ]; then
        echo "  ↷ reusing checkpoint $out"; return 0
    fi
    [ -d "$out" ] && rm -rf "$out"   # lerobot-train refuses an existing dir
    local args=(
        --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --output_dir="$out" --num_workers=4 --log_freq=100
        --env_eval_freq=0 --wandb.enable=false --policy.push_to_hub=false
        --policy.device="$DEVICE"
    )
    case "$policy" in
        act) args+=(--policy.type=act --steps="$ACT_STEPS" \
                    --batch_size="$ACT_BATCH" --save_freq="$ACT_SAVE");;
        diffusion) args+=(--policy.type=diffusion --steps="$DIFF_STEPS" \
                    --batch_size="$DIFF_BATCH" \
                    --policy.pretrained_backbone_weights=null \
                    --policy.resize_shape="[$DIFF_RESIZE_H,$DIFF_RESIZE_W]" \
                    --save_freq="$DIFF_SAVE");;
        *) fail "unknown policy '$policy' (want act|diffusion)";;
    esac
    echo; echo "### train $policy on $REPO_ID"
    lerobot-train "${args[@]}" 2>&1 | tee "$RUN_DIR/logs/train_${policy}.log" \
        || fail "train ($policy)"
    [ -d "$out/checkpoints/last/pretrained_model" ] || fail "train ($policy): no checkpoint"
}

if [ "$SKIP_TRAIN" = "0" ]; then
    have act && train_cell act
    have diffusion && train_cell diffusion
fi

# ---- report ----------------------------------------------------------
REPORT="$RUN_DIR/results.md"
{
    echo "# Real-VLA long training — $REPO_ID"
    echo
    echo "- dataset: \`$DATASET_ROOT\`"
    echo "- device: $DEVICE"
    echo
    echo "| policy | checkpoint | final train loss |"
    echo "|--------|------------|------------------|"
    for policy in act diffusion; do
        have "$policy" || continue
        ckpt="$RUN_DIR/train/${policy}/checkpoints/last/pretrained_model"
        loss="$(grep -oE 'loss:[0-9.]+' "$RUN_DIR/logs/train_${policy}.log" 2>/dev/null \
                | tail -1 | cut -d: -f2)"
        [ -d "$ckpt" ] && echo "| $policy | \`$ckpt\` | ${loss:-n/a} |" \
                       || echo "| $policy | (not trained) | - |"
    done
    echo
    echo "On-robot evaluation is not run here. Copy a checkpoint back to the rig"
    echo "and run \`tool/run_policy_real.py --checkpoint <ckpt> --task ...\`."
} > "$REPORT"

echo
echo "✅ REAL-VLA LONG RUN COMPLETE — see $REPORT"
cat "$REPORT"
