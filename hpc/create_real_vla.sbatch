#!/bin/bash
# =====================================================================
# KCL CREATE (Slurm) -- REAL-data VLA training. ONE array task = ONE row of a
# run manifest = one (dataset, policy) training run on a PRE-STAGED dataset
# collected on the rig. The real-world counterpart of create_sim_vla.sbatch:
# no MuJoCo (no simulation here) and no in-loop evaluation (a real policy is
# evaluated ON THE ROBOT with tool/run_policy_real.py, back at the rig).
#
# Submit it through the wrapper, which fills everything site-specific in and
# picks the array range from the manifest:
#
#     bash hpc/submit_real.sh              # see hpc/README.md
#
# Nothing here needs editing. It takes its paths from the environment the
# wrapper exports (SO101_REPO_ROOT, SO101_SCRATCH, SO101_MANIFEST), falling
# back to the submit directory, /scratch/users/$USER and hpc/runs.tsv, so a
# bare `sbatch hpc/create_real_vla.sbatch` from the repo still trains the
# manifest's first row.
# =====================================================================
#SBATCH --job-name=real_vla
#SBATCH --partition=gpu                     # e.g. gpu  -- check `sinfo -s`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                        # any >=12 GB GPU suffices for act+diffusion
#SBATCH --mem=48G
#SBATCH --time=24:00:00                     # overridden per group by submit_real.sh
#SBATCH --array=0                           # overridden by submit_real.sh
#SBATCH --output=%x-%A_%a.out               # <job-name>-<arrayjob>_<task>.out

set -euo pipefail

# ---- where things are (no site paths baked in) -----------------------
REPO_ROOT="${SO101_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH="${SO101_SCRATCH:-/scratch/users/${USER:-$(id -un)}}"
MANIFEST="${SO101_MANIFEST:-$REPO_ROOT/hpc/runs.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

[ -f "$REPO_ROOT/setup.sh" ] || { echo "❌ no repo at REPO_ROOT=$REPO_ROOT"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ no manifest at $MANIFEST"; exit 1; }

# ---- pick this task's row --------------------------------------------
# Comments and blank lines are skipped, so the index counts DATA rows only --
# the same numbering submit_real.sh wrote into its index_*.md.
ROW="$(awk 'NF && $1 !~ /^#/' "$MANIFEST" | sed -n "$((TASK_ID + 1))p")"
if [ -z "$ROW" ]; then
    echo "❌ manifest $MANIFEST has no row $TASK_ID"; exit 1
fi
# `extra` is the last column and may contain spaces: read takes the rest.
read -r DATASET POLICY STEPS BATCH HOURS EXTRA <<<"$ROW"

echo "======================================================================"
echo " REAL-VLA  array task $TASK_ID  ->  dataset=$DATASET policy=$POLICY"
echo "   steps=$STEPS batch=$BATCH hours=$HOURS extra=${EXTRA:--}"
echo "   manifest=$MANIFEST"
echo "======================================================================"

cd "$REPO_ROOT"

# ---- modules (verify for your site) ----------------------------------
# The pip torch wheel bundles its own CUDA runtime; the driver supplies
# libcuda. Load whatever CREATE requires, if anything:
# module load cuda/<version>

# ---- environment -----------------------------------------------------
# setup.sh activates the venv and sets PYTHONPATH. Point the big caches and
# outputs at scratch (home quotas are small) AFTER sourcing so our values win.
# No MUJOCO_GL: this job never renders a simulation.
source setup.sh
export HF_LEROBOT_HOME="${SCRATCH}/hf_lerobot"   # staged dataset: <here>/local/<DATASET>
export SO101_OUTPUT_DIR="${SCRATCH}/so101_outputs"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$SO101_OUTPUT_DIR" "${HF_LEROBOT_HOME}/local"

# ---- fail-fast checks (before hours of training) ---------------------
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== staged dataset present? =="
DS="${HF_LEROBOT_HOME}/local/${DATASET}"
if [ ! -f "${DS}/meta/info.json" ]; then
    echo "❌ staged real dataset missing at ${DS}"
    echo "   stage it from the collection box first (see hpc/README.md)."
    exit 1
fi

# ---- run the training ------------------------------------------------
# The run name is the DATASET, not the job id, so a requeue or a resubmission
# after a wall-time hit reuses the finished checkpoints instead of retraining;
# both policies of one dataset therefore share a run directory and one report.
RUN_NAME="${SO101_RUN_TAG:-$DATASET}"
args=(--dataset-root "$DS" --only "$POLICY" --run-name "$RUN_NAME")
[ "${STEPS:--}" != "-" ] && args+=(--steps "$STEPS")
[ "${BATCH:--}" != "-" ] && args+=(--batch "$BATCH")
[ "${EXTRA:--}" != "-" ] && args+=(--extra "$EXTRA")

bash test/system/long_vla_real.sh "${args[@]}"

echo "✓ Done. Checkpoints + results.md under:"
echo "  ${SO101_OUTPUT_DIR}/vla_real_long/${RUN_NAME}/"
echo "  Evaluate a checkpoint ON THE ROBOT with tool/run_policy_real.py."
