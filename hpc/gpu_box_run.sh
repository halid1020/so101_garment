#!/usr/bin/env bash
# =====================================================================
# Run a training matrix on a PLAIN GPU box -- one with a card and an SSH login
# and no queue manager. The counterpart of hpc/create_real_vla.sbatch, which
# does the same work as one task of a Slurm array.
#
# Everything after the row is picked is identical at both destinations: the same
# camera view, the same test/system/long_vla_real.sh, the same run directory.
# Only two things differ, and they are the two Slurm was doing for us:
#
#   * NOTHING ELSE RESERVES THE CARD. Two runs started together do not queue,
#     they fight: MEASURED on this rig's 24 GB box, pi0.5 under LoRA already
#     needs 21.5 GiB, so a second run does not wait its turn, it takes the
#     first one down with an OOM hours in. So the whole matrix runs under one
#     `flock`, one row at a time, and a second invocation waits or is refused.
#   * NOTHING KEEPS IT ALIVE. The launcher's SSH connection goes away as soon
#     as it has started this, and a 36-hour run must not go with it -- so
#     --detach re-execs under setsid with the output redirected to a log, and
#     prints the pid and that log for the caller to remember.
#
#   bash hpc/gpu_box_run.sh --manifest runs.tsv --scratch ~/.cache/huggingface/lerobot
#   bash hpc/gpu_box_run.sh --manifest runs.tsv --detach
#   bash hpc/gpu_box_run.sh --manifest runs.tsv --dry-run
#
# Options:
#   --manifest F   the run matrix to work through (required). Same eight
#                  columns as hpc/runs.tsv; every data row is run, in order.
#   --scratch DIR  where HF_LEROBOT_HOME and SO101_OUTPUT_DIR live
#                  (default: $SO101_SCRATCH, else ~/.cache/huggingface/lerobot)
#   --detach       start in the background and print `pid=N log=PATH`, then
#                  return. Without it the rows run in the foreground.
#   --wait         with a run already going, wait for the lock instead of
#                  refusing. Off by default: a launcher that silently blocks
#                  for 36 hours looks exactly like one that has hung.
#   --dry-run      print the command for each row and run nothing.
#   --status       say whether a run is going here, and exit.
#   --stop         end the run going here (SIGTERM to its process group).
#
# The dataset must already be staged under <scratch>/local/<dataset>;
# tool/train_launch.py does that before it calls this.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST=""
SCRATCH="${SO101_SCRATCH:-$HOME/.cache/huggingface/lerobot}"
DETACH=0
WAIT=0
DRY_RUN=0
STATUS=0
STOP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2;;
        --scratch) SCRATCH="$2"; shift 2;;
        --detach) DETACH=1; shift;;
        --wait) WAIT=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --status) STATUS=1; shift;;
        --stop) STOP=1; shift;;
        -h|--help) sed -n '2,42p' "${BASH_SOURCE[0]}"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

OUT_ROOT="${SO101_OUTPUT_DIR:-$SCRATCH/so101_outputs}"
RUN_STATE="$OUT_ROOT/gpu_box"
LOCK="$RUN_STATE/run.lock"
PIDFILE="$RUN_STATE/run.pid"
mkdir -p "$RUN_STATE"

# ---- status / stop, which need no manifest ---------------------------
running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    echo "$pid"
}

if [ "$STATUS" = "1" ]; then
    if pid="$(running_pid)"; then
        echo "running pid=$pid log=$RUN_STATE/run.log"
    else
        echo "idle"
    fi
    exit 0
fi

if [ "$STOP" = "1" ]; then
    if pid="$(running_pid)"; then
        # The group, not the pid: this script starts lerobot-train as a child,
        # and killing only the shell would leave the training holding the card.
        kill -TERM -- "-$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill -TERM "$pid"
        echo "stopping pid=$pid"
    else
        echo "idle"
    fi
    exit 0
fi

[ -n "$MANIFEST" ] || { echo "❌ --manifest is required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "❌ no manifest at $MANIFEST" >&2; exit 1; }
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"

# ---- detach ----------------------------------------------------------
# Re-exec ourselves in a new session with the output on a file, so the caller's
# SSH connection closing cannot take a 36-hour run with it.
if [ "$DETACH" = "1" ]; then
    LOG="$RUN_STATE/run.log"
    args=(--manifest "$MANIFEST" --scratch "$SCRATCH")
    [ "$WAIT" = "1" ] && args+=(--wait)
    setsid nohup bash "${BASH_SOURCE[0]}" "${args[@]}" \
        >"$LOG" 2>&1 </dev/null &
    child=$!
    disown "$child" 2>/dev/null || true
    echo "pid=$child log=$LOG"
    exit 0
fi

# ---- one run at a time ----------------------------------------------
exec 9>"$LOCK"
if [ "$WAIT" = "1" ]; then
    flock 9
elif ! flock -n 9; then
    holder="$(cat "$PIDFILE" 2>/dev/null || echo '?')"
    echo "❌ a training run is already going on this machine (pid $holder)." >&2
    echo "   This card holds one run at a time — wait for it, pass --wait, or" >&2
    echo "   stop it with: bash hpc/gpu_box_run.sh --stop" >&2
    exit 3
fi
echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# ---- environment -----------------------------------------------------
# setup.sh activates the venv and sets PYTHONPATH; the caches are pointed at
# scratch AFTER sourcing, so these values win. No MUJOCO_GL: nothing renders.
# shellcheck disable=SC1091
source setup.sh
export HF_LEROBOT_HOME="$SCRATCH"
export SO101_OUTPUT_DIR="$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$SO101_OUTPUT_DIR" "$HF_LEROBOT_HOME/local"

echo "======================================================================"
echo " GPU-BOX TRAINING on $(hostname)"
echo "   manifest: $MANIFEST"
echo "   scratch : $SCRATCH"
echo "   outputs : $SO101_OUTPUT_DIR"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

FAILED=0
ROW_NO=0
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    ROW_NO=$((ROW_NO + 1))
    read -r DATASET POLICY CAMERAS STEPS BATCH HOURS SLOTS EXTRA <<<"$line"
    if [ -z "${SLOTS:-}" ]; then
        echo "❌ row $ROW_NO has 7 fields, not 8 — the 'slots' column was added." >&2
        echo "   Put a '-' before the last column: $line" >&2
        exit 2
    fi

    echo
    echo "----------------------------------------------------------------------"
    echo " row $ROW_NO: $DATASET / $POLICY   cameras=$CAMERAS slots=$SLOTS"
    echo "----------------------------------------------------------------------"

    DS="$HF_LEROBOT_HOME/local/$DATASET"
    if [ ! -f "$DS/meta/info.json" ]; then
        echo "❌ staged dataset missing at $DS — stage it first" >&2
        FAILED=$((FAILED + 1)); continue
    fi

    [ "${CAMERAS:--}" = "-" ] && CAMERAS="all"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  would build view: tool/make_camera_view.py --dataset $DS --cameras $CAMERAS"
        echo "  would train     : test/system/long_vla_real.sh --only $POLICY"
        continue
    fi

    VIEW="$("$PWD/venv/bin/python" tool/make_camera_view.py \
            --dataset "$DS" --cameras "$CAMERAS" \
            --out-dir "$HF_LEROBOT_HOME/local" --print path)" || {
        echo "❌ could not build the camera view for '$CAMERAS'" >&2
        FAILED=$((FAILED + 1)); continue
    }
    echo "  camera view: $VIEW"

    # The run name is the VIEW, exactly as on the cluster, so a rerun after a
    # crash reuses finished checkpoints and every policy trained on one camera
    # set shares a run directory and one report.
    args=(--dataset-root "$VIEW" --only "$POLICY" --run-name "$(basename "$VIEW")")
    [ "${STEPS:--}" != "-" ] && args+=(--steps "$STEPS")
    [ "${BATCH:--}" != "-" ] && args+=(--batch "$BATCH")
    [ "${SLOTS:--}" != "-" ] && args+=(--slots "$SLOTS")
    [ "${EXTRA:--}" != "-" ] && args+=(--extra "$EXTRA")

    # One failing row must not abandon the rest of the matrix: a machine left
    # idle overnight because row 2 of 6 had a typo is the whole cost of not
    # having a queue manager, and it is avoidable here.
    if bash test/system/long_vla_real.sh "${args[@]}"; then
        echo "✓ row $ROW_NO done"
    else
        echo "❌ row $ROW_NO FAILED ($DATASET / $POLICY)" >&2
        FAILED=$((FAILED + 1))
    fi
done < "$MANIFEST"

echo
if [ "$FAILED" -gt 0 ]; then
    echo "❌ $FAILED of $ROW_NO row(s) failed. Checkpoints for the rest are under"
    echo "   $SO101_OUTPUT_DIR/vla_real_long/"
    exit 1
fi
echo "✅ all $ROW_NO row(s) complete — $SO101_OUTPUT_DIR/vla_real_long/"
echo "   Fetch a checkpoint with hpc/fetch_policies.sh, evaluate it ON THE ROBOT."
