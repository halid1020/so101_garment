#!/usr/bin/env bash
# =====================================================================
# One-time provisioning of the sim-VLA training/eval environment on the
# KCL CREATE cluster (or any Slurm HPC login node with internet access).
#
# This is the SIM-ONLY subset of ../install.sh: it builds the venv,
# installs LeRobot from source at the pinned commit with the training
# extras, adds this repo's requirements, and warms the torchvision
# backbone cache so that offline GPU compute nodes never try to download
# weights mid-train.
#
# It deliberately SKIPS install.sh steps 4-6 (meta_quest_teleop, adb,
# openscad, pre-commit): none are needed for simulation training/eval,
# and their `sudo apt install` steps abort on a cluster with no sudo.
#
# Run ONCE, on a CREATE *login* node (compute nodes have no internet):
#     module load <a Python >=3.12 module>     # see hpc/README.md
#     bash hpc/provision_create.sh
#
# Idempotent: safe to re-run. The pins below MUST match install.sh --
# keep them in sync if install.sh changes.
# =====================================================================
set -euo pipefail

# ---- pins (keep identical to ../install.sh) --------------------------
LEROBOT_COMMIT="3dd19d043e2f3fe5673b13ea0ebe4f31884c0797"
LEROBOT_EXTRAS="feetech,dataset,pi,libero,pusht,training,diffusion,peft"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# LeRobot requires Python >=3.12. On CREATE, `module load` a suitable
# Python before running this so that `python3` below resolves to >=3.12 --
# a venv built on an older python3 silently fails LeRobot's editable
# install (see CLAUDE.md: the Anaconda-python3.9-ahead-of-PATH pitfall).
py_ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "=> python3 = $(command -v python3) (Python ${py_ver})"
case "$py_ver" in
    3.1[2-9] | 3.[2-9][0-9]) : ;;
    *) echo "❌ Python ${py_ver} < 3.12 -- module load a newer Python first." >&2; exit 1 ;;
esac

# 1. Virtual environment ------------------------------------------------
if [ ! -d venv ]; then
    echo "=> Creating Python venv..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "=> Upgrading pip..."
pip install --upgrade pip

# 2. LeRobot from source (parallel ../lerobot), pinned commit + extras --
echo "=> Installing LeRobot ${LEROBOT_COMMIT:0:10} [${LEROBOT_EXTRAS}]..."
cd "$REPO_ROOT/.."
if [ ! -d lerobot ]; then
    echo "=> Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git
fi
cd lerobot
git fetch --quiet origin || true
git checkout --quiet "$LEROBOT_COMMIT"
# egl_probe / hf-egl-probe (LIBERO deps) need cmake<4 and a no-isolation
# pre-build -- identical to install.sh:71-75.
pip install "cmake<4" setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    pip install --no-build-isolation egl_probe hf-egl-probe ||
    echo "⚠️  egl_probe pre-build skipped (already satisfied)."
pip install -e ".[${LEROBOT_EXTRAS}]"

# 3. This repo's extra requirements ------------------------------------
cd "$REPO_ROOT"
echo "=> Installing project requirements..."
pip install -r requirements.txt

# 4. Warm the vision-backbone cache (CRITICAL for offline compute nodes)-
# Both ACT and Diffusion Policy build a torchvision ResNet18 whose default
# (ImageNet) weights download from a CDN on first use. Compute nodes have
# no internet, so fetch them now into ~/.cache/torch/hub/checkpoints,
# which is on shared home and therefore visible to the compute node.
echo "=> Warming torchvision ResNet18 backbone cache..."
python - <<'PY'
import torchvision.models as m

# ResNet18_Weights.DEFAULT == IMAGENET1K_V1, the file LeRobot's default
# ACT/Diffusion backbone config requests (resnet18-f37072fd.pth).
m.resnet18(weights=m.ResNet18_Weights.DEFAULT)
print("✓ ResNet18 weights cached under ~/.cache/torch/hub/checkpoints")
PY

echo "=================================================="
echo "✓ CREATE provisioning complete (sim-only subset)."
echo "  Next: stage the dataset to scratch, then submit the job."
echo "  See hpc/README.md."
echo "=================================================="
