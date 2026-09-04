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
# them and their final training loss. `pi05` and `fastwam` are also trainable
# here; both are finetunes of a large pretrained model rather than policies
# trained from scratch, and differ accordingly (see below).
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
#   bash test/system/long_vla_real.sh --dataset-root <ds> --only pi05 \
#        --slots central=base,left_arm_left_gripper=left_wrist
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
# The save interval is an exposure window, not a disk-space setting: a machine
# that goes down loses everything since its last checkpoint, and the resume path
# can only pick up from one that exists. thanos has gone down UNCLEANLY four
# times since 2026-08-23 -- no shutdown record, no journal entry, no OOM, no Xid
# -- and three of those killed a training run, the last of them 33 seconds after
# training began. Until that is fixed, 5000 steps caps the loss at roughly an
# hour instead of three. KEEP_CKPTS still prunes, so this costs no disk.
ACT_STEPS=80000;  ACT_BATCH=8;   ACT_SAVE=5000
DIFF_STEPS=100000; DIFF_BATCH=32; DIFF_SAVE=5000
DIFF_RESIZE_H=180; DIFF_RESIZE_W=240   # downsample cams for the diffusion encoder (3:4)
# pi0.5 finetunes a pretrained base: far fewer steps than training from scratch,
# and a small batch because 4.1B params leave little room even under LoRA.
PI05_STEPS=30000; PI05_BATCH=8; PI05_SAVE=5000
# FastWAM builds its visual world model from a Wan2.2 video backbone and then
# predicts actions directly, so like pi0.5 it is a finetune and wants far fewer
# steps than the docs' 300000, which is sized for a large multi-task corpus.
# Its image features are CONCATENATED into one frame of --fastwam-image-size,
# so every camera must be that high and their widths must sum to its width:
# two square views, or one wide one. A camera set that cannot is refused by
# tool/train_launch.py before anything is reserved.
FASTWAM_STEPS=30000; FASTWAM_BATCH=8; FASTWAM_SAVE=5000
FASTWAM_IMAGE_SIZE="[224,448]"
FASTWAM_HORIZON=32; FASTWAM_N_ACTION_STEPS=10
# Flow matching: pi0.5's objective trained from scratch on a small ResNet trunk,
# so it wants more steps than a finetune and fewer than ACT, and a batch this
# card has room for.
FLOWMATCH_STEPS=60000; FLOWMATCH_BATCH=16; FLOWMATCH_SAVE=10000
# The world action model predicts video as well as actions, so a step costs more
# than a plain policy's and the batch is smaller for the same card.
DREAMZERO_STEPS=80000; DREAMZERO_BATCH=8; DREAMZERO_SAVE=10000
# Off by default: the coupled schedule is the paper's DreamZero, and Flash is the
# variant measured against it.
DREAMZERO_FLASH=0
PI05_BASE="${SO101_PI05_BASE:-lerobot/pi05_base}"
PI05_LORA_R=16                     # 0 => full finetuning (needs a very large GPU)
# Which pi0.5 slot each camera is fed into, as camera=slot pairs. Empty means
# derive it from the view (named viewpoints first, then the free slots in
# order), which is right for most rows; an ablation that needs a particular
# assignment pins it here and the run matrix carries it in one column.
PI05_SLOTS=""
STEPS=""; BATCH=""; SAVE_FREQ=""   # per-run overrides for the selected policy
# Every sample decodes one video frame per camera, so the loader is the floor on
# training speed. Follow the CPUs the job was actually given rather than a fixed
# 4: reading a dataset over network scratch with too few workers is the likeliest
# explanation for the cube-pnp ACT run that hit its 24 h wall time at ~10 s/step.
#
# But a box can also have MORE cores than its power supply can feed alongside the
# card. thanos has 24 of them, and every diffusion run started there at 24
# workers took the whole machine down within minutes -- twice, at batch 32 and
# at batch 16, with nothing in the journal either time (see the SAVE comment
# above). At 8 workers and batch 8 the same run held 366 W of a 480 W limit and
# trained through. Which of the two reductions mattered is NOT isolated, so this
# knob exists to vary one of them without touching the other, and without
# pretending to be Slurm on a machine that has none.
WORKERS="${SO101_LOADER_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}}"
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
        --fastwam-steps) FASTWAM_STEPS="$2"; shift 2;;
        --fastwam-image-size) FASTWAM_IMAGE_SIZE="$2"; shift 2;;
        --slots) PI05_SLOTS="$2"; shift 2;;
        --lora-r) PI05_LORA_R="$2"; shift 2;;
        --dreamzero-flash) DREAMZERO_FLASH=1; shift;;
        --steps) STEPS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --save-freq) SAVE_FREQ="$2"; shift 2;;
        --workers) WORKERS="$2"; shift 2;;
        --keep-checkpoints) KEEP_CKPTS="$2"; shift 2;;
        --extra) EXTRA="$2"; shift 2;;
        --skip-train) SKIP_TRAIN=1; shift;;
        -h|--help) sed -n '2,40p' "$0"; exit 0;;
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
    # Reports WHY, not just what: "cpu unavailable" (torch cannot use any GPU)
    # and "cpu small" (there is one, but it is under MIN_GB) need opposite
    # responses, and the guard below can only tell them apart if we say so.
    DEVICE_WHY="$("$PY" - <<'PY'
import torch

MIN_GB = 8.0
if not torch.cuda.is_available():
    print("cpu unavailable")
elif torch.cuda.get_device_properties(0).total_memory / 1e9 < MIN_GB:
    print("cpu small")
else:
    print("cuda ok")
PY
)"
    DEVICE="${DEVICE_WHY%% *}"
    WHY="${DEVICE_WHY##* }"

    # A batch job that quietly falls back to the CPU is the worst outcome
    # available: a 4.1B-param finetune cannot finish on CPU, so the row burns
    # its entire wall-time reservation, writes no checkpoint, and logs nothing
    # that looks wrong. MEASURED 2026-08-25 on CREATE: three pi05 array tasks
    # landed on a node that HAD allocated them a GPU (CUDA_VISIBLE_DEVICES=0,
    # torch.cuda.device_count() == 1) which torch then could not use --
    # is_available() False, torch.cuda.init() raising "No CUDA GPUs are
    # available" -- and trained on CPU, reaching no steps in ten minutes where
    # the same run on a healthy node did a hundred in fifty seconds. That is a
    # sick node, not a configuration choice, so refuse it. Asking for CPU
    # explicitly with --device cpu still works, and a GPU merely too SMALL
    # still falls back quietly, which is what a laptop smoke test wants.
    if [ "$DEVICE" = "cpu" ] && [ "$WHY" = "unavailable" ] \
       && [ -n "${CUDA_VISIBLE_DEVICES:-}${SLURM_JOB_GPUS:-}" ]; then
        echo "❌ A GPU was allocated to this job (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')," >&2
        echo "   but torch cannot use it, so this run would train on the CPU:" >&2
        "$PY" - >&2 <<'PY'
import torch

print(f"     torch {torch.__version__}   device_count={torch.cuda.device_count()}")
try:
    torch.cuda.init()
except Exception as exc:
    print(f"     torch.cuda.init(): {type(exc).__name__}: {exc}")
PY
        echo "   That usually means the NODE's GPU is unhealthy. Resubmit without it:" >&2
        echo "     bash hpc/submit_real.sh --exclude ${SLURMD_NODENAME:-<node>} ..." >&2
        echo "   (--exclude, NOT \$SBATCH_EXCLUDE: CREATE's Slurm ignores that.)" >&2
        echo "   Or pass --device cpu if CPU training really is intended." >&2
        exit 1
    fi
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
# A policy implemented in THIS repo (src/so101_policies/) rather than in LeRobot.
# These are ports -- the upstream module tree moved, not rewritten -- so each one
# wants exactly the flags its upstream twin wants, plus the one that makes
# lerobot-train import our package before it parses anything. Deriving the base
# instead of repeating every branch is what keeps the two from drifting apart.
base_policy()  { case "$1" in so101_*) echo "${1#so101_}";; *) echo "$1";; esac; }
local_policy() { case "$1" in so101_*) return 0;; *) return 1;; esac; }
SO101_POLICY_PACKAGE="so101_policies"

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


# How far a run got, from the checkpoint it last wrote. Checkpoint directories
# are named for the step (`checkpoints/070000`), and `last` points at the newest
# one, so its target IS the step. Echoes 0 when there is no checkpoint at all.
checkpoint_step() {
    local link="$1/checkpoints/last"
    [ -e "$link/pretrained_model" ] || { echo 0; return 0; }
    local name
    name="$(basename "$(readlink -f "$link")")"
    case "$name" in
        ''|*[!0-9]*) echo 0;;      # not a step-named directory: treat as none
        *) echo "$((10#$name))";;  # 10# so 070000 is 70000, not an octal error
    esac
}

train_cell() {
    local policy="$1"
    local out="$RUN_DIR/train/${policy}"
    local steps batch save base
    # A ported policy takes its twin's budget on purpose: a run meant to compare
    # the two implementations must not also change the number of steps.
    base="$(base_policy "$policy")"
    case "$base" in
        act)       steps="$ACT_STEPS";  batch="$ACT_BATCH";  save="$ACT_SAVE";;
        diffusion) steps="$DIFF_STEPS"; batch="$DIFF_BATCH"; save="$DIFF_SAVE";;
        pi05)      steps="$PI05_STEPS"; batch="$PI05_BATCH"; save="$PI05_SAVE";;
        fastwam)   steps="$FASTWAM_STEPS"; batch="$FASTWAM_BATCH"; save="$FASTWAM_SAVE";;
        flowmatch) steps="$FLOWMATCH_STEPS"; batch="$FLOWMATCH_BATCH"; save="$FLOWMATCH_SAVE";;
        dreamzero) steps="$DREAMZERO_STEPS"; batch="$DREAMZERO_BATCH"; save="$DREAMZERO_SAVE";;
        *) fail "unknown policy '$policy' (want act|diffusion|pi05|fastwam|flowmatch|dreamzero, or so101_ prefixed)";;
    esac
    # A run may override the policy's sizing; --only selects the policy, so one
    # value each is enough and the cluster manifest carries one column each.
    [ -n "$STEPS" ] && steps="$STEPS"
    [ -n "$BATCH" ] && batch="$BATCH"
    [ -n "$SAVE_FREQ" ] && save="$SAVE_FREQ"
    # A save interval longer than the run writes no checkpoint at all, which
    # this script would then report as a failed train. Clamp instead.
    [ "$save" -gt "$steps" ] && save="$steps"

    # Three ways to meet an existing run directory, and only one of them is a
    # reason to delete it. A box that reboots mid-run (this one has, twice) used
    # to cost the WHOLE run, because a partial directory was wiped and started
    # again from zero -- while lerobot-train has supported resuming from the last
    # checkpoint all along.
    local resume=0 at
    at="$(checkpoint_step "$out")"
    if [ "$at" -ge "$steps" ] && [ "$at" -gt 0 ]; then
        echo "  ↷ reusing checkpoint $out (step $at)"; return 0
    elif [ "$at" -gt 0 ]; then
        # Every $save steps is what is at risk, never the run.
        echo "  ↻ resuming $out at step $at of $steps"
        resume=1
    else
        [ -d "$out" ] && rm -rf "$out"   # lerobot-train refuses an existing dir
    fi

    # A resume takes its whole configuration from the checkpoint's own
    # train_config.json -- model, optimiser, scheduler and schedule alike -- so
    # restating any of that here could only contradict it. What IS restated is
    # where things are: a staged dataset or a run directory can legitimately
    # have moved between the crash and the retry, and the step target can
    # legitimately have been raised.
    if [ "$resume" = "1" ]; then
        local args=(
            --config_path="$out/checkpoints/last/pretrained_model/train_config.json"
            --resume=true
            --output_dir="$out"
            --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT"
            --steps="$steps" --num_workers="$WORKERS"
        )
        echo; echo "### resume $policy on $REPO_ID (step $at -> $steps)"
        lerobot-train "${args[@]}" 2>&1 | tee -a "$RUN_DIR/logs/train_${policy}.log" \
            || fail "train ($policy, resumed)"
        [ -d "$out/checkpoints/last/pretrained_model" ] || fail "train ($policy): no checkpoint"
        return 0
    fi

    local args=(
        --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --output_dir="$out" --num_workers="$WORKERS" --log_freq=100
        --env_eval_freq=0 --wandb.enable=false --policy.push_to_hub=false
        --policy.device="$DEVICE"
        --steps="$steps" --batch_size="$batch" --save_freq="$save"
    )
    if [ "$base" = "pi05" ]; then
        # A finetune names its base instead of a policy type; the type comes
        # from the base's own config. Which is exactly why a REPO-LOCAL pi0.5
        # needs the base retargeted first: lerobot/pi05_base says "pi05", so
        # --policy.path to it would quietly load LeRobot's class no matter what
        # --only asked for. The retarget symlinks the 14.5 GB of weights and
        # rewrites one field.
        local base_path="$PI05_BASE"
        if local_policy "$policy"; then
            base_path="$("$PY" "$REPO_ROOT/tool/retarget_checkpoint.py" \
                --checkpoint "$PI05_BASE" --to "$policy" \
                --out "$RUN_DIR/base_${policy}" --print-path)" \
                || fail "retarget $PI05_BASE to $policy"
            echo "  base retargeted: $base_path"
        fi
        args+=(--policy.path="$base_path")
        [ "$PI05_LORA_R" != "0" ] && args+=(--peft.r="$PI05_LORA_R")
        # Map THIS dataset's cameras onto the base's slots. Asking the dataset
        # rather than hard-coding it is what makes a camera ablation work: a
        # view with one camera produces a one-entry map, and pi0.5 pads and
        # masks the two slots left over.
        local rename
        local slot_args=()
        [ -n "$PI05_SLOTS" ] && [ "$PI05_SLOTS" != "-" ] && slot_args=(--slots "$PI05_SLOTS")
        rename="$("$PY" "$REPO_ROOT/tool/make_camera_view.py" \
            --dataset "$DATASET_ROOT" "${slot_args[@]}" --print pi05-rename-map)" \
            || fail "pi05 rename map for $DATASET_ROOT"
        args+=(--rename_map="$rename")
        echo "  pi0.5 base   : $PI05_BASE"
        echo "  pi0.5 LoRA r : ${PI05_LORA_R} $([ "$PI05_LORA_R" = 0 ] && echo '(full finetune)')"
        echo "  camera slots : $rename"
    else
        args+=(--policy.type="$policy")
    fi
    # lerobot.configs.parser.wrap loads this package before draccus parses, which
    # is what puts our policy in the registry that --policy.type is looked up in.
    # Without it the run dies on an unknown policy type having reserved the GPU.
    if local_policy "$policy"; then
        args+=(--policy.discover_packages_path="$SO101_POLICY_PACKAGE")
        echo "  policy package: $SO101_POLICY_PACKAGE (implemented in this repo)"
    fi
    if [ "$base" = "diffusion" ]; then
        args+=(--policy.pretrained_backbone_weights=null
               --policy.resize_shape="[$DIFF_RESIZE_H,$DIFF_RESIZE_W]")
    fi
    if [ "$base" = "dreamzero" ] && [ "$DREAMZERO_FLASH" = "1" ]; then
        args+=(--policy.flash=true)
        echo "  dreamzero    : Flash (decoupled video/action noise schedules)"
    fi
    if [ "$base" = "fastwam" ]; then
        # The action and proprioception widths come from THIS dataset rather
        # than from the docs' 7 and 8: this rig has two arms, so both are 12,
        # and a mismatch is a shape error thousands of steps in.
        local dims
        dims="$("$PY" - "$DATASET_ROOT" <<'PYDIM'
import json, sys
info = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print(info["features"]["action"]["shape"][0], info["features"]["observation.state"]["shape"][0])
PYDIM
)" || fail "fastwam dimensions for $DATASET_ROOT"
        args+=(--policy.action_dim="${dims%% *}"
               --policy.proprio_dim="${dims##* }"
               --policy.action_horizon="$FASTWAM_HORIZON"
               --policy.n_action_steps="$FASTWAM_N_ACTION_STEPS"
               --policy.image_size="$FASTWAM_IMAGE_SIZE")
        echo "  fastwam dims : action=${dims%% *} proprio=${dims##* }"
        echo "  fastwam image: $FASTWAM_IMAGE_SIZE"
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
    # A loop rather than a ladder: a ladder silently IGNORED any policy nobody
    # had added a line for, so --only <typo> trained nothing and reported success.
    # train_cell's own case refuses a name it does not know.
    for policy in ${ONLY//,/ }; do
        train_cell "$policy"
        prune_checkpoints "$policy"
    done
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
