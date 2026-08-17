#!/bin/bash
# =====================================================================
# KCL CREATE (Slurm) -- sim-VLA training + inference, minimal working cell:
#   task = single (one arm), mode = simple (overfit sanity),
#   policies = act + diffusion.
# Trains from a PRE-STAGED local dataset (train -> validate -> eval) and
# skips the slow teleop-oracle collection. Mirrors the recipe that reaches
# 100% eval success on the local box. See hpc/README.md for the runbook.
#
# >>> FILL EVERY <...> PLACEHOLDER BELOW BEFORE SUBMITTING <<<
# =====================================================================
#SBATCH --job-name=sim_vla_single
#SBATCH --partition=gpu        # e.g. gpu  -- check `sinfo -s`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                        # any >=12 GB GPU suffices for this cell
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=%x-%j.out                  # <job-name>-<jobid>.out in the submit dir

set -euo pipefail

# ---- paths (EDIT) ----------------------------------------------------
REPO_ROOT="/users/k2694428/so101_garment"         # where you cloned this repo on CREATE
SCRATCH="/scratch/users/k2694428"                          # big, node-visible scratch, e.g. /scratch/users/<k-number>

# ---- modules (EDIT / verify) -----------------------------------------
# The pip torch wheel bundles its own CUDA runtime; the NVIDIA driver on
# the GPU node supplies libcuda + libEGL, so a CUDA module is usually
# unnecessary. Load whatever CREATE requires, if anything:
# module load cuda/<version>

cd "$REPO_ROOT"

# ---- environment -----------------------------------------------------
# setup.sh activates the venv, sets PYTHONPATH, pre-writes the LIBERO
# config, and defaults MUJOCO_GL=egl on a headless node. We then point the
# big caches/outputs at scratch (home quotas are small; outputs run to
# tens of GB) -- exported AFTER sourcing so our values win.
source setup.sh
export HF_LEROBOT_HOME="${SCRATCH}/hf_lerobot"      # staged dataset: <here>/local/so101_sim_single_simple
export SO101_OUTPUT_DIR="${SCRATCH}/so101_outputs"
export MUJOCO_GL=egl
# If the EGL probe below fails on a multi-GPU node, MuJoCo may be picking a
# GPU other than the one Slurm allocated. Uncomment to pin it:
# export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
mkdir -p "$SO101_OUTPUT_DIR" "${HF_LEROBOT_HOME}/local"

# ---- fail-fast checks (before hours of training) ---------------------
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== MuJoCo EGL offscreen render probe =="
python - <<'PY'
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

# A failed render raises here -> non-zero exit -> `set -e` aborts the job
# before hours of training are wasted on a node without working EGL.
xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)
renderer.render()
print("✓ MuJoCo EGL offscreen render OK")
# Exit before GC runs the renderer's finalizer: MuJoCo's EGL context
# teardown prints harmless-but-alarming "Exception ignored in __del__"
# tracebacks at interpreter shutdown. os._exit skips them; the render
# already succeeded, so the probe's verdict stands.
sys.stdout.flush()
os._exit(0)
PY

echo "== staged dataset present? =="
DS="${HF_LEROBOT_HOME}/local/so101_sim_single_simple"
if [ ! -d "$DS" ]; then
    echo "❌ staged dataset missing at ${DS}"
    echo "   rsync it from the collection box first (see hpc/README.md)."
    exit 1
fi

# ---- run the working cell -------------------------------------------
# The driver resumes: on a fresh output dir it trains from scratch, then
# validates every checkpoint and evaluates the best on the 30 EVAL seeds
# with the enhanced per-episode videos.
bash test/system/long_vla_sim.sh \
    --modes simple --tasks single --only act,diffusion \
    --skip-collect --run-name "create_${SLURM_JOB_ID}"

echo "✓ Done. Results under:"
echo "  ${SO101_OUTPUT_DIR}/vla_sim_long/create_${SLURM_JOB_ID}/simple/single/{act,diffusion}/eval/"
