#!/bin/bash
# =====================================================================
# KCL CREATE (Slurm) -- REAL-data VLA training: policies = act + diffusion,
# trained from a PRE-STAGED real dataset collected on the rig. The real-world
# counterpart of create_sim_vla.sbatch: no MuJoCo (no simulation here) and no
# in-loop evaluation (a real policy is evaluated ON THE ROBOT with
# tool/run_policy_real.py, back at the rig). See hpc/README.md for the runbook.
#
# >>> FILL EVERY <...> PLACEHOLDER BELOW BEFORE SUBMITTING <<<
# =====================================================================
#SBATCH --job-name=real_vla
#SBATCH --partition=gpu        # e.g. gpu  -- check `sinfo -s`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                        # any >=12 GB GPU suffices for act+diffusion
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out                  # <job-name>-<jobid>.out in the submit dir

set -euo pipefail

# ---- paths (EDIT) ----------------------------------------------------
REPO_ROOT="/user/k2694428/so101_garment"         # where you cloned this repo on CREATE
SCRATCH="/scratch/users/k2694428"                # big, node-visible scratch
DATASET_NAME="<your-real-dataset>"               # e.g. towel_fold — staged below

# ---- modules (EDIT / verify) -----------------------------------------
# The pip torch wheel bundles its own CUDA runtime; the driver supplies
# libcuda. Load whatever CREATE requires, if anything:
# module load cuda/<version>

cd "$REPO_ROOT"

# ---- environment -----------------------------------------------------
# setup.sh activates the venv and sets PYTHONPATH. Point the big caches and
# outputs at scratch (home quotas are small) AFTER sourcing so our values win.
# No MUJOCO_GL: this job never renders a simulation.
source setup.sh
export HF_LEROBOT_HOME="${SCRATCH}/hf_lerobot"   # staged dataset: <here>/local/<DATASET_NAME>
export SO101_OUTPUT_DIR="${SCRATCH}/so101_outputs"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$SO101_OUTPUT_DIR" "${HF_LEROBOT_HOME}/local"

# ---- fail-fast checks (before hours of training) ---------------------
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== staged dataset present? =="
DS="${HF_LEROBOT_HOME}/local/${DATASET_NAME}"
if [ ! -f "${DS}/meta/info.json" ]; then
    echo "❌ staged real dataset missing at ${DS}"
    echo "   rsync it from the collection box first (see hpc/README.md)."
    exit 1
fi

# ---- run the long training ------------------------------------------
# Resumes: finished checkpoints are reused; partial train dirs are cleared.
bash test/system/long_vla_real.sh \
    --dataset-root "$DS" --only act,diffusion \
    --run-name "create_${SLURM_JOB_ID}"

echo "✓ Done. Checkpoints + results.md under:"
echo "  ${SO101_OUTPUT_DIR}/vla_real_long/create_${SLURM_JOB_ID}/"
echo "  Evaluate a checkpoint ON THE ROBOT with tool/run_policy_real.py."
