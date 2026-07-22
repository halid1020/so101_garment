#!/usr/bin/env bash
# =====================================================================
# Sim-VLA LONG experiment  (oracle gate -> collect -> train -> validate
# -> evaluate -> report), sized for a big GPU (validated target: RTX
# 3090 Ti, 24 GB, no sudo). See documents/long_vla_sim_guide.md.
#
# Two environment setups run in order (--modes to pick one):
#   simple : train/val/eval all on the seed-0 scenario (overfit sanity)
#   full   : ~1000 demos over TRAIN seeds; disjoint VAL/EVAL seed pools
# For each mode x task (single, handover) x policy (act, diffusion,
# pi05): collect teleop-oracle demonstrations (only verified successes
# are saved), train, validate every checkpoint on the VAL seeds, then
# evaluate the selected checkpoint on all 30 EVAL seeds with videos.
#
#   bash test/system/long_vla_sim.sh                  # both modes, all cells
#   bash test/system/long_vla_sim.sh --modes simple   # sanity half only
#   bash test/system/long_vla_sim.sh --only pi05      # one policy
#   bash test/system/long_vla_sim.sh --skip-collect   # reuse datasets
#   bash test/system/long_vla_sim.sh --skip-train     # re-validate/eval only
#   bash test/system/long_vla_sim.sh --pi05-steps 20000
#
# The script resumes: existing datasets, finished checkpoints, val
# results and eval results are reused; partial train dirs are cleared.
# Run it under tmux/nohup — the full matrix is multi-day.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults --------------------------------------------------------
MODES="simple,full"
ONLY=""                      # comma list of policies; empty = all three
ORACLE="teleop"              # collection oracle (direct = faster fallback)
SIMPLE_EPISODES=100
FULL_EPISODES=1000
GATE_EPISODES=30             # dry-run episodes per gate cell
SINGLE_GATE=95               # abort if teleop single success (%) is below
HANDOVER_GATE=70             # relay teleop measures ~75-86%/attempt; gate
                             # well below that band (but far above a broken
                             # <50% oracle) so sampling noise never aborts
ACT_STEPS=80000;  ACT_BATCH=8;   ACT_SAVE=10000
DIFF_STEPS=100000; DIFF_BATCH=32; DIFF_SAVE=10000
PI05_STEPS=10000; PI05_BATCH=8
VAL_TRIALS=5                 # VAL-seed rollouts per checkpoint (5-10)
CAM_W=640; CAM_H=480
DEVICE=""
RUN_NAME="long_$(date +%Y%m%d_%H%M%S)"
SKIP_COLLECT=0; SKIP_TRAIN=0; SKIP_GATE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --modes) MODES="$2"; shift 2;;
        --only) ONLY="$2"; shift 2;;
        --oracle) ORACLE="$2"; shift 2;;
        --simple-episodes) SIMPLE_EPISODES="$2"; shift 2;;
        --full-episodes) FULL_EPISODES="$2"; shift 2;;
        --act-steps) ACT_STEPS="$2"; shift 2;;
        --diffusion-steps) DIFF_STEPS="$2"; shift 2;;
        --pi05-steps) PI05_STEPS="$2"; shift 2;;
        --val-trials) VAL_TRIALS="$2"; shift 2;;
        --camera-width) CAM_W="$2"; shift 2;;
        --camera-height) CAM_H="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --skip-collect) SKIP_COLLECT=1; shift;;
        --skip-train) SKIP_TRAIN=1; shift;;
        --skip-gate) SKIP_GATE=1; shift;;
        -h|--help) sed -n '2,26p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

POLICIES="act diffusion pi05"
[ -n "$ONLY" ] && POLICIES="${ONLY//,/ }"
TASKS="single handover"

# ---- environment -----------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-.:src}"
PY="$REPO_ROOT/venv/bin/python"

OUT_ROOT="${SO101_OUTPUT_DIR:-$REPO_ROOT/outputs}"
RUN_DIR="$OUT_ROOT/vla_sim_long/$RUN_NAME"
mkdir -p "$RUN_DIR"/{logs,oracle_gate,collect_stats}
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# Device: this experiment is GPU-sized. Refuse silent CPU fallback.
if [ -z "$DEVICE" ]; then
    if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        DEVICE="cuda"
    else
        echo "❌ No CUDA GPU found — the long run is GPU-sized (multi-day even"
        echo "   on a 3090 Ti). Pass --device cpu only if you really mean it."
        exit 2
    fi
fi

cat <<BANNER

======================================================================
 SIM-VLA LONG EXPERIMENT
----------------------------------------------------------------------
 modes    : $MODES     tasks: $TASKS     policies: $POLICIES
 collect  : $ORACLE oracle, ${CAM_W}x${CAM_H}; simple=$SIMPLE_EPISODES, full=$FULL_EPISODES eps/task
 gate     : teleop >= ${SINGLE_GATE}% (single) / ${HANDOVER_GATE}% (handover), $GATE_EPISODES eps/cell
 train    : act ${ACT_STEPS}, diffusion ${DIFF_STEPS}, pi05 ${PI05_STEPS} (LoRA r=16 from pi05_base)
 validate : $VAL_TRIALS VAL-seed rollouts per checkpoint
 evaluate : all 30 EVAL seeds, videos on
 device   : $DEVICE
 output   : $RUN_DIR
 datasets : $LEROBOT_HOME/local/so101_sim_<task>_<mode>
======================================================================

BANNER

phase() { echo; echo "### [$(date +%H:%M:%S)] $1"; echo; }
fail() { echo; echo "❌ LONG RUN FAILED during: $1"; exit 1; }

# ---- preflight -------------------------------------------------------
phase "Preflight"
"$PY" -m sim_twin.verify >/dev/null 2>&1 || fail "twin assets (python -m sim_twin.verify)"
if grep -qw pi05 <<<"$POLICIES"; then
    "$PY" - <<'PY' || fail "pi0.5 prerequisites (hf auth login + accept the licence at https://huggingface.co/google/paligemma-3b-pt-224)"
from huggingface_hub import auth_check
auth_check("google/paligemma-3b-pt-224")
PY
fi
echo "  ✓ twin assets + model access OK"

# ---- phase 0: oracle gate --------------------------------------------
if [ "$SKIP_GATE" = "0" ]; then
    phase "Phase 0 — oracle gate ($GATE_EPISODES dry-run eps per cell; teleop is real-time)"
    for task in $TASKS; do
        for oracle in teleop direct; do
            out="$RUN_DIR/oracle_gate/${task}_${oracle}.json"
            [ -f "$out" ] && { echo "  ↷ reusing $out"; continue; }
            "$PY" tool/collect_sim_dataset.py \
                --task "$task" --oracle "$oracle" --episodes "$GATE_EPISODES" \
                --dry-run --stats-out "$out" \
                --camera-width "$CAM_W" --camera-height "$CAM_H" \
                > "$RUN_DIR/logs/gate_${task}_${oracle}.log" 2>&1 \
                || fail "oracle gate ($task/$oracle)"
            echo "  $task/$oracle: $("$PY" -c "import json;d=json.load(open('$out'));print(f\"{d['oracle_success_rate']*100:.0f}% ({d['episodes_collected']}/{d['episode_attempts']})\")")"
        done
    done
    "$PY" - "$RUN_DIR" "$SINGLE_GATE" "$HANDOVER_GATE" <<'PY' || fail "oracle gate (teleop below threshold — do not train on a broken oracle)"
import json, sys
from pathlib import Path
run, sgate, hgate = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
gates = {"single": sgate, "handover": hgate}
ok = True
for task, gate in gates.items():
    d = json.loads((run / "oracle_gate" / f"{task}_teleop.json").read_text())
    rate = 100 * d["oracle_success_rate"]
    mark = "✓" if rate >= gate else "✗"
    print(f"  {mark} teleop {task}: {rate:.0f}% (gate {gate:.0f}%)")
    ok &= rate >= gate
sys.exit(0 if ok else 1)
PY
else
    phase "Phase 0 — oracle gate SKIPPED (--skip-gate)"
fi

# ---- helpers ---------------------------------------------------------
collect_cell() {  # mode task
    local mode="$1" task="$2"
    local repo="local/so101_sim_${task}_${mode}"
    local root="$LEROBOT_HOME/$repo"
    local stats="$RUN_DIR/collect_stats/${mode}_${task}.json"
    if [ -d "$root" ]; then
        echo "  ↷ dataset exists: $root"; return 0
    fi
    [ "$SKIP_COLLECT" = "1" ] && fail "collect ($mode/$task): --skip-collect but no dataset at $root"
    local eps seeds
    if [ "$mode" = "simple" ]; then eps="$SIMPLE_EPISODES"; seeds="simple"
    else eps="$FULL_EPISODES"; seeds="full"; fi
    "$PY" tool/collect_sim_dataset.py \
        --task "$task" --oracle "$ORACLE" --seeds "$seeds" --episodes "$eps" \
        --repo-id "$repo" --stats-out "$stats" \
        --camera-width "$CAM_W" --camera-height "$CAM_H" \
        2>&1 | tee "$RUN_DIR/logs/collect_${mode}_${task}.log" \
        || fail "collect ($mode/$task)"
    [ -d "$root" ] || fail "collect ($mode/$task): no dataset written"
}

watch_speed() {  # logfile total_steps  — print a projection once ~step 50 lands
    local log="$1" total="$2"
    (
        for _ in $(seq 1 720); do
            line=$(grep -oE "step:[0-9]+ .*updt_s:[0-9.]+" "$log" 2>/dev/null | tail -1 || true)
            if [ -n "$line" ]; then
                step=$(sed -E 's/^step:([0-9]+).*/\1/' <<<"$line")
                if [ "${step:-0}" -ge 50 ]; then
                    upd=$(grep -oE "updt_s:[0-9.]+" <<<"$line" | cut -d: -f2)
                    "$REPO_ROOT/venv/bin/python" -c "print(f'\n⏱️  measured ~{$upd:.1f} s/step at step $step → ~{$upd*$total/3600:.1f} h projected for $total steps — trim --pi05-steps now if that is too long.')"
                    exit 0
                fi
            fi
            sleep 10
        done
    ) &
}

train_cell() {  # mode task policy
    local mode="$1" task="$2" policy="$3"
    local repo="local/so101_sim_${task}_${mode}"
    local out="$RUN_DIR/$mode/$task/$policy"
    if [ -d "$out/checkpoints/last/pretrained_model" ]; then
        echo "  ↷ reusing checkpoints in $out"; return 0
    fi
    [ "$SKIP_TRAIN" = "1" ] && fail "train ($mode/$task/$policy): --skip-train but no checkpoint"
    [ -d "$out" ] && rm -rf "$out"
    local args=(
        --dataset.repo_id="$repo" --dataset.root="$LEROBOT_HOME/$repo"
        --dataset.video_backend=pyav
        --output_dir="$out" --num_workers=4 --log_freq=100
        --env_eval_freq=0 --wandb.enable=false --policy.push_to_hub=false
        --policy.device="$DEVICE"
    )
    local log="$RUN_DIR/logs/train_${mode}_${task}_${policy}.log"
    case "$policy" in
        act)
            args+=(--policy.type=act --steps="$ACT_STEPS" --batch_size="$ACT_BATCH" --save_freq="$ACT_SAVE");;
        diffusion)
            args+=(--policy.type=diffusion --steps="$DIFF_STEPS" --batch_size="$DIFF_BATCH" --save_freq="$DIFF_SAVE");;
        pi05)
            # LoRA-finetune the published base — full finetuning OOMs 24 GB.
            args+=(--policy.path=lerobot/pi05_base --peft.r=16
                   --steps="$PI05_STEPS" --batch_size="$PI05_BATCH"
                   --save_freq=$(( PI05_STEPS / 5 )))
            watch_speed "$log" "$PI05_STEPS";;
    esac
    phase "Train $policy on $mode/$task"
    lerobot-train "${args[@]}" 2>&1 | tee "$log" || fail "train ($mode/$task/$policy)"
    [ -d "$out/checkpoints/last/pretrained_model" ] || fail "train ($mode/$task/$policy): no checkpoint"
}

validate_cell() {  # mode task policy — roll every checkpoint on the VAL seeds
    local mode="$1" task="$2" policy="$3"
    local out="$RUN_DIR/$mode/$task/$policy"
    mkdir -p "$out/val"
    local seeds_mode="val"
    [ "$mode" = "simple" ] && seeds_mode="simple"   # simple mode validates on its one scenario
    local found=0
    for d in "$out"/checkpoints/*/; do
        local step; step="$(basename "$d")"
        [ "$step" = "last" ] && continue
        [ -d "$d/pretrained_model" ] || continue
        found=1
        local vout="$out/val/step_${step}.json"
        [ -f "$vout" ] && { echo "  ↷ $vout"; continue; }
        "$PY" tool/eval_sim_policy.py \
            --task "$task" --checkpoint "$d/pretrained_model" \
            --seeds "$seeds_mode" --episodes "$VAL_TRIALS" \
            --camera-width "$CAM_W" --camera-height "$CAM_H" \
            --device "$DEVICE" --out "$vout" \
            > "$RUN_DIR/logs/val_${mode}_${task}_${policy}_${step}.log" 2>&1 \
            || fail "validate ($mode/$task/$policy step $step)"
        echo "  step $step: $("$PY" -c "import json;d=json.load(open('$vout'));print(f\"{d['success_rate']*100:.0f}%\")")"
    done
    [ "$found" = "1" ] || fail "validate ($mode/$task/$policy): no checkpoints found"
    # Select the best checkpoint (highest val success; ties -> latest step).
    "$PY" - "$out" <<'PY' || fail "checkpoint selection"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
best = None
for f in sorted(out.glob("val/step_*.json")):
    step = f.stem.split("_", 1)[1]
    d = json.loads(f.read_text())
    key = (d["success_rate"], int(step))
    if best is None or key >= best[0]:
        best = (key, step, d["success_rate"])
sel = {"step": best[1], "val_success": best[2],
       "checkpoint": str(out / "checkpoints" / best[1] / "pretrained_model")}
(out / "selected.json").write_text(json.dumps(sel, indent=2))
print(f"  selected step {best[1]} ({100*best[2]:.0f}% val)")
PY
}

eval_cell() {  # mode task policy — the selected checkpoint on all EVAL seeds
    local mode="$1" task="$2" policy="$3"
    local out="$RUN_DIR/$mode/$task/$policy"
    [ -f "$out/eval/results.json" ] && { echo "  ↷ $out/eval/results.json"; return 0; }
    local seeds_mode="full"
    [ "$mode" = "simple" ] && seeds_mode="simple"   # 30 trials of the seed-0 scenario
    local ckpt; ckpt="$("$PY" -c "import json;print(json.load(open('$out/selected.json'))['checkpoint'])")"
    "$PY" tool/eval_sim_policy.py \
        --task "$task" --checkpoint "$ckpt" \
        --seeds "$seeds_mode" \
        --camera-width "$CAM_W" --camera-height "$CAM_H" \
        --device "$DEVICE" --out "$out/eval/results.json" \
        --video-dir "$out/eval/videos" \
        2>&1 | tee "$RUN_DIR/logs/eval_${mode}_${task}_${policy}.log" \
        || fail "evaluate ($mode/$task/$policy)"
}

# ---- the matrix ------------------------------------------------------
for mode in ${MODES//,/ }; do
    phase "MODE: $mode — collect"
    for task in $TASKS; do collect_cell "$mode" "$task"; done
    for task in $TASKS; do
        for policy in $POLICIES; do
            train_cell "$mode" "$task" "$policy"
        done
    done
    phase "MODE: $mode — validate + evaluate"
    for task in $TASKS; do
        for policy in $POLICIES; do
            validate_cell "$mode" "$task" "$policy"
            eval_cell "$mode" "$task" "$policy"
        done
    done
done

# ---- report ----------------------------------------------------------
phase "Report"
"$PY" -m sim_datagen.report "$RUN_DIR" || fail "report"
echo
echo "✅ LONG RUN COMPLETE — see $RUN_DIR/results.md"
echo "   Send back: results.md, results.json, collect_stats/, and the"
echo "   */eval/videos directories (they feed the analysis notebook + paper)."
