#!/usr/bin/env bash
# =====================================================================
# Real-data VLA LONG training  (train -> checkpoint -> report), sized for a
# big GPU. The real-world counterpart of test/system/long_vla_sim.sh, minus
# the parts that only exist in simulation:
#   * NO oracle gate / collection — the dataset was teleoperated on the rig;
#   * NO in-loop evaluation — a real policy is evaluated ON THE ROBOT with
#     tool/run_policy.py at the rig, not on the cluster.
# For each policy (act, diffusion by default) it trains a long run on the
# staged real dataset, saves checkpoints, and writes a results.md pointing at
# them and their final training loss.
#
# `pi05` is also trainable here, and differs from the other two in three ways
# that are all consequences of it being a FINETUNE of a 4.1B-param base rather
# than a policy trained from scratch:
#   * it starts from --policy.path (a pre-staged copy of lerobot/pi05_base;
#     compute nodes are offline, so the base must already be in the HF cache);
#   * it trains through LoRA, because full finetuning needs >24 GB just for
#     AdamW's optimiser state — --lora-r 0 turns that off if the GPU is big;
#   * its cameras are RENAMED onto the base's three pretrained slots rather
#     than derived from the dataset, since each slot carries what it learned
#     about that viewpoint. A slot with no camera behind it is padded and
#     masked by pi0.5 itself, which is exactly what a camera ablation wants.
#
#   bash test/system/long_vla_real.sh --dataset-root <ds>
#   bash test/system/long_vla_real.sh --dir /media/hdd/so101 --name towel_fold
#   bash test/system/long_vla_real.sh --dataset-root <ds> --only diffusion
#   bash test/system/long_vla_real.sh --dataset-root <ds> --act-steps 40000
#   bash test/system/long_vla_real.sh --dataset-root <ds> --only act --steps 40000
#   bash test/system/long_vla_real.sh --dataset-root <ds> --extra "--policy.optimizer_lr=5e-5"
#   bash test/system/long_vla_real.sh --dataset-root <ds> --keep-checkpoints 3
#
# --steps/--batch/--save-freq apply to whichever policy --only selects, so one
# cluster row per (dataset, policy) needs one column each; --extra is handed to
# lerobot-train verbatim, so a flag this script does not name is still reachable.
#
# --keep-checkpoints N (default 2) bounds what a run leaves on disk: lerobot-train
# writes a checkpoint every --save_freq steps and never removes one, so a 100k-step
# diffusion run parks ten ~3.3 GB copies. That filled CREATE's 200 GB scratch quota
# and killed four jobs mid-save; 0 disables the pruning.
#
# The script resumes: a finished checkpoint is reused, a partial train dir is
# cleared. Several invocations may share one --run-name (that is how the cluster
# array trains both policies of a dataset into one run directory), so each writes
# its own results_<policy>.md and rebuilds results.md from what is on disk.
# Run it under Slurm (hpc/create_real_vla.sbatch) or tmux/nohup.
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
# pi0.5 finetunes a pretrained base: far fewer steps than training from scratch,
# and a small batch because 4.1B params leave little room even under LoRA.
PI05_STEPS=30000; PI05_BATCH=8; PI05_SAVE=5000
PI05_BASE="${SO101_PI05_BASE:-lerobot/pi05_base}"
PI05_LORA_R=16                     # 0 => full finetuning (needs a very large GPU)
STEPS=""; BATCH=""; SAVE_FREQ=""   # per-run overrides for the selected policy
# Every sample decodes one video frame per camera, so the loader is the floor on
# training speed. Follow the CPUs the job was actually given rather than a fixed
# 4: reading a dataset over network scratch with too few workers is the likeliest
# explanation for the cube-pnp ACT run that hit its 24 h wall time at ~10 s/step.
WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}"
# How many step checkpoints to keep per policy once training finishes (see
# prune_checkpoints below for why this is not simply "all of them"). 0 => keep all.
KEEP_CKPTS=2
EXTRA=""                           # raw lerobot-train flags, appended last
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
        --pi05-steps) PI05_STEPS="$2"; shift 2;;
        --pi05-base) PI05_BASE="$2"; shift 2;;
        --lora-r) PI05_LORA_R="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --save-freq) SAVE_FREQ="$2"; shift 2;;
        --workers) WORKERS="$2"; shift 2;;
        --keep-checkpoints) KEEP_CKPTS="$2"; shift 2;;
        --extra) EXTRA="$2"; shift 2;;
        --skip-train) SKIP_TRAIN=1; shift;;
        -h|--help) sed -n '2,38p' "$0"; exit 0;;
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

# Episodes deleted in the review tool are only MARKED until that dataset is
# compacted: they are still on disk, so training would still learn from takes the
# operator threw away. Ask the marker itself rather than parsing it here.
"$PY" - "$DATASET_ROOT" <<'PY' || exit 2
import sys
from common.recording.dataset_edit import read_soft_deleted

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
echo "   policies: $ONLY   device: $DEVICE   loader workers: $WORKERS"
echo "   cameras : $("$PY" "$REPO_ROOT/tool/make_camera_view.py" --dataset "$DATASET_ROOT" --list \
                     | tail -n +2 | awk '{printf "%s ", $1}')"
echo "   output  : $RUN_DIR"
echo "   NOTE: on-robot evaluation is a rig step (tool/run_policy.py)."
echo "======================================================================"

fail() { echo; echo "❌ Real-VLA long run FAILED during: $1"; exit 1; }
have() { case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }

# lerobot-train writes a checkpoint every --save_freq steps and never removes an
# older one, so a 100k-step diffusion run parks ten ~3.3 GB copies and an 80k-step
# ACT run eight ~590 MB ones. On CREATE that is fatal rather than merely untidy:
# the scratch quota is a HARD 200 GB (ceph.quota.max_bytes), and in August 2026
# four array tasks ran for 1.5-17 h and then died inside save_pretrained with
# "OSError: [Errno 122] Disk quota exceeded" -- 182 GB of that scratch was
# superseded intermediates. Nothing downstream wants them: hpc/fetch_policies.sh
# only ever copies checkpoints/last/pretrained_model.
#
# Keeps the newest $KEEP_CKPTS step directories AND whatever `last` resolves to.
# Those are normally the same directory, but a run whose save was interrupted
# leaves a newest one that is a truncated stub while `last` still names the last
# checkpoint written whole -- keeping both means pruning can never orphan `last`,
# which is the only thing a resume or a fetch reads.
prune_checkpoints() {
    local policy="$1"
    local ck="$RUN_DIR/train/${policy}/checkpoints"
    [ "$KEEP_CKPTS" = "0" ] && return 0
    [ -d "$ck" ] || return 0

    local keep_last=""
    [ -e "$ck/last" ] && keep_last="$(basename "$(readlink -f "$ck/last")")"

    local steps=() d
    for d in "$ck"/[0-9]*/; do
        [ -d "$d" ] || continue          # an unmatched glob stays literal
        steps+=("$(basename "$d")")
    done
    [ "${#steps[@]}" -le "$KEEP_CKPTS" ] && return 0

    # Oldest first. `sort -n` rather than lexical, so a run whose --save-freq
    # produced differently-padded names still orders correctly.
    local drop=$(( ${#steps[@]} - KEEP_CKPTS )) s
    while read -r s; do
        [ -n "$s" ] || continue
        [ "$s" = "$keep_last" ] && continue
        echo "  ✂ pruning superseded checkpoint ${policy}/${s}"
        rm -rf "${ck:?}/${s}"
    done < <(printf '%s\n' "${steps[@]}" | sort -n | head -n "$drop")
}


train_cell() {
    local policy="$1"
    local out="$RUN_DIR/train/${policy}"
    if [ -d "$out/checkpoints/last/pretrained_model" ]; then
        echo "  ↷ reusing checkpoint $out"; return 0
    fi
    [ -d "$out" ] && rm -rf "$out"   # lerobot-train refuses an existing dir
    local steps batch save
    case "$policy" in
        act)       steps="$ACT_STEPS";  batch="$ACT_BATCH";  save="$ACT_SAVE";;
        diffusion) steps="$DIFF_STEPS"; batch="$DIFF_BATCH"; save="$DIFF_SAVE";;
        pi05)      steps="$PI05_STEPS"; batch="$PI05_BATCH"; save="$PI05_SAVE";;
        *) fail "unknown policy '$policy' (want act|diffusion|pi05)";;
    esac
    # A run may override the policy's sizing; --only selects the policy, so one
    # value each is enough and the cluster manifest carries one column each.
    [ -n "$STEPS" ] && steps="$STEPS"
    [ -n "$BATCH" ] && batch="$BATCH"
    [ -n "$SAVE_FREQ" ] && save="$SAVE_FREQ"
    # A save interval longer than the run writes no checkpoint at all, which
    # this script would then report as a failed train. Clamp instead.
    [ "$save" -gt "$steps" ] && save="$steps"

    local args=(
        --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --output_dir="$out" --num_workers="$WORKERS" --log_freq=100
        --env_eval_freq=0 --wandb.enable=false --policy.push_to_hub=false
        --policy.device="$DEVICE"
        --steps="$steps" --batch_size="$batch" --save_freq="$save"
    )
    if [ "$policy" = "pi05" ]; then
        # A finetune names its base instead of a policy type; the type comes
        # from the base's own config.
        args+=(--policy.path="$PI05_BASE")
        [ "$PI05_LORA_R" != "0" ] && args+=(--peft.r="$PI05_LORA_R")
        # Map THIS dataset's cameras onto the base's slots. Asking the dataset
        # rather than hard-coding it is what makes a camera ablation work: a
        # view with one camera produces a one-entry map, and pi0.5 pads and
        # masks the two slots left over.
        local rename
        rename="$("$PY" "$REPO_ROOT/tool/make_camera_view.py" \
            --dataset "$DATASET_ROOT" --print pi05-rename-map)" \
            || fail "pi05 rename map for $DATASET_ROOT"
        args+=(--rename_map="$rename")
        echo "  pi0.5 base   : $PI05_BASE"
        echo "  pi0.5 LoRA r : ${PI05_LORA_R} $([ "$PI05_LORA_R" = 0 ] && echo '(full finetune)')"
        echo "  camera slots : $rename"
    else
        args+=(--policy.type="$policy")
    fi
    if [ "$policy" = "diffusion" ]; then
        args+=(--policy.pretrained_backbone_weights=null
               --policy.resize_shape="[$DIFF_RESIZE_H,$DIFF_RESIZE_W]")
    fi
    # Deliberately unquoted: --extra is a string of flags to be word-split.
    # shellcheck disable=SC2206
    [ -n "$EXTRA" ] && args+=($EXTRA)
    echo; echo "### train $policy on $REPO_ID ($steps steps, batch $batch)"
    lerobot-train "${args[@]}" 2>&1 | tee "$RUN_DIR/logs/train_${policy}.log" \
        || fail "train ($policy)"
    [ -d "$out/checkpoints/last/pretrained_model" ] || fail "train ($policy): no checkpoint"
}

if [ "$SKIP_TRAIN" = "0" ]; then
    have act && { train_cell act; prune_checkpoints act; }
    have diffusion && { train_cell diffusion; prune_checkpoints diffusion; }
    have pi05 && { train_cell pi05; prune_checkpoints pi05; }
fi

# ---- report ----------------------------------------------------------
# Runs sharing a --run-name (the cluster array trains one policy per task into
# one dataset's run directory) would clobber each other's rows if the summary
# were written from $ONLY. Each run writes its own results_<policy>.md, and
# results.md is rebuilt from the checkpoints ON DISK -- so it converges to the
# complete table whichever task finishes last. Written via a temp file so a
# reader never catches it half-written.
final_loss() {  # policy -> last logged training loss, or empty
    # `\bloss:` so a metric whose name merely ENDS in loss is not mistaken for
    # it, and `|| true` because a run too short to log one is not a failure --
    # under `set -o pipefail` a grep that matches nothing would end the script
    # here, after the training it is reporting on has already succeeded.
    { grep -oE '\bloss:[0-9.]+' "$RUN_DIR/logs/train_${1}.log" 2>/dev/null \
        | tail -1 | cut -d: -f2; } || true
}

for policy in ${ONLY//,/ }; do
    ckpt="$RUN_DIR/train/${policy}/checkpoints/last/pretrained_model"
    [ -d "$ckpt" ] || continue
    {
        echo "# Real-VLA long training — $REPO_ID / $policy"
        echo
        echo "- dataset: \`$DATASET_ROOT\`"
        echo "- device: $DEVICE"
        echo "- checkpoint: \`$ckpt\`"
        loss="$(final_loss "$policy")"
        echo "- final train loss: ${loss:-n/a}"
    } > "$RUN_DIR/results_${policy}.md.tmp"
    mv "$RUN_DIR/results_${policy}.md.tmp" "$RUN_DIR/results_${policy}.md"
done

REPORT="$RUN_DIR/results.md"
{
    echo "# Real-VLA long training — $REPO_ID"
    echo
    echo "- dataset: \`$DATASET_ROOT\`"
    echo "- device: $DEVICE"
    echo
    echo "| policy | checkpoint | final train loss |"
    echo "|--------|------------|------------------|"
    for d in "$RUN_DIR"/train/*/; do
        [ -d "$d" ] || continue
        policy="$(basename "$d")"
        ckpt="${d}checkpoints/last/pretrained_model"
        loss="$(final_loss "$policy")"
        [ -d "$ckpt" ] && echo "| $policy | \`$ckpt\` | ${loss:-n/a} |" \
                       || echo "| $policy | (training unfinished) | - |"
    done
    echo
    echo "On-robot evaluation is not run here. Copy a checkpoint back to the rig"
    echo "and run \`tool/run_policy.py --checkpoint <ckpt> --task ...\`."
} > "$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"

echo
echo "✅ REAL-VLA LONG RUN COMPLETE — see $REPORT"
cat "$REPORT"
