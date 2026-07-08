#!/usr/bin/env bash
# One-shot installer for the Vision-Tactile World Model workspace:
#   - Python venv + project requirements (requirements.txt)
#   - LeRobot v0.5.x from source (parallel ../lerobot) with the extras
#     needed for SO-101 hardware, pi0.5 training, and the LIBERO + PushT
#     simulation benchmarks
#   - Meta Quest teleop reader + adb
#   - Pink IK / visualization deps + pre-commit hooks
#
# Safe to re-run: every step is idempotent. Run from the repo root:
#     bash install.sh
# Author: Abudureyimu Halite

set -euo pipefail

# Pin LeRobot to the commit this project was validated against.
LEROBOT_COMMIT="3dd19d043e2f3fe5673b13ea0ebe4f31884c0797"
# LeRobot install extras. libero/pusht give the simulation benchmarks;
# pi = pi0/pi0.5 policies; feetech = SO-101 bus; dataset = LeRobotDataset;
# training = accelerate + wandb, required by `lerobot-train` for ANY policy;
# diffusion = diffusers, required by the small diffusion policy used in
# test/smoke_test_pipeline.sh; peft = LoRA fine-tuning (`--peft.*`), needed
# to finetune pi0.5 (a ~4B-param model) on GPUs too small for full-parameter
# AdamW (full finetuning's optimizer state alone needs >30 GB — LoRA keeps
# only a small adapter's worth of params trainable).
# (libero is linux-only; pip silently skips its marker off-linux.)
LEROBOT_EXTRAS="feetech,dataset,pi,libero,pusht,training,diffusion,peft"

echo "=================================================="
echo "🚀 Initializing Workspace Installation..."
echo "=================================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# 1. Virtual environment ------------------------------------------------
if [ ! -d "venv" ]; then
    echo "=> Creating Python 3 virtual environment..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "=> Upgrading pip..."
pip install --upgrade pip

# 2. LeRobot from source (parallel ../lerobot) --------------------------
# Installed BEFORE requirements.txt so its tight pins win; requirements.txt
# only adds this repo's extra deps on top.
echo "=> Setting up local LeRobot (${LEROBOT_COMMIT:0:10}) with [${LEROBOT_EXTRAS}]..."
cd "$REPO_ROOT/.."
if [ ! -d "lerobot" ]; then
    echo "=> Cloning LeRobot repository..."
    git clone https://github.com/huggingface/lerobot.git
fi
cd lerobot
git fetch --quiet origin || true
git checkout --quiet "$LEROBOT_COMMIT"
# egl_probe / hf-egl-probe (LIBERO deps) shell out to `cmake` while
# building, and break two ways:
#  1. If the venv contains the pip `cmake` package, its venv/bin/cmake SHIM
#     shadows the system cmake on PATH — and under pip's build isolation
#     (PYTHONPATH pointing into the isolated env) the shim's `from cmake
#     import cmake` crashes, failing the wheels with "cmake --version
#     returned non-zero". Pre-building WITHOUT isolation runs the build in
#     the venv environment, where the shim actually works.
#  2. CMake 4.x removed compatibility with the packages' ancient
#     `cmake_minimum_required(<3.5)`. Pin the venv cmake to 3.x and set
#     CMAKE_POLICY_VERSION_MINIMUM as belt-and-braces for machines whose
#     system cmake is already 4.x.
# Idempotent: no-ops once the wheels are installed/cached.
pip install "cmake<4" setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    pip install --no-build-isolation egl_probe hf-egl-probe || \
    echo "⚠️  egl_probe pre-build skipped (non-linux or already satisfied)."
pip install -e ".[${LEROBOT_EXTRAS}]"

# 3. Project requirements ----------------------------------------------
cd "$REPO_ROOT"
if [ -f "requirements.txt" ]; then
    echo "=> Installing project requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found — skipping project deps."
fi

# 4. Meta Quest teleop reader ------------------------------------------
echo "=> Setting up Meta Quest teleop reader..."
cd "$REPO_ROOT/.."
if [ ! -d "meta_quest_teleop" ]; then
    echo "=> Cloning meta_quest_teleop repository..."
    git clone https://github.com/NeuracoreAI/meta_quest_teleop.git
fi
cd meta_quest_teleop
pip install -e .

# adb (Android Debug Bridge) — Meta Quest USB link
if ! command -v adb &> /dev/null; then
    echo "=> Installing adb (Meta Quest USB, sudo may prompt)..."
    sudo apt install -y android-tools-adb
fi

# 5. OpenSCAD for the digital twin (optional but recommended) ----------
if ! command -v openscad &> /dev/null; then
    echo "=> Installing OpenSCAD (digital-twin part export, sudo may prompt)..."
    sudo apt install -y openscad || echo "⚠️  OpenSCAD install skipped; sim_twin export needs it."
fi

# 6. Pre-commit hooks ---------------------------------------------------
cd "$REPO_ROOT"
echo "=> Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

echo "=================================================="
echo "✓ Installation complete."
echo "  Before each working session run:  source setup.sh"
echo "  Smoke-test the training pipeline:  bash test/smoke_test_pipeline.sh"
echo "=================================================="
