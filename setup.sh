#!/usr/bin/env bash
# Source this before every working session:  source setup.sh
#
# Activates the venv, puts the project + src/ on PYTHONPATH, prepares the
# simulation render backend and cache/output locations, and (for the real
# rig) grants serial access and checks the Meta Quest link. Safe to source
# repeatedly.

source venv/bin/activate

# Absolute path of the directory containing this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# --- Simulation / training environment --------------------------------
# MuJoCo render backend. On a headless box (no DISPLAY) the LIBERO
# offscreen renderer needs EGL; on a machine with a display we leave the
# default so the interactive teleop viewer (glfw) still works. Override by
# exporting MUJOCO_GL yourself before sourcing.
if [ -z "${DISPLAY:-}" ]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

# Video decoding: LeRobotDataset videos are AV1-encoded. The default
# 'torchcodec' backend dlopen's system FFmpeg, which this box either lacks
# or (via a stray conda copy) is too old to decode AV1, and there's no
# sudo here to fix it system-wide. LeRobot has no env-var override for
# this, so every lerobot-train/lerobot-eval invocation in this project
# (see test/smoke_test_pipeline.sh, README) passes
# --dataset.video_backend=pyav explicitly — PyAV's wheel bundles its own
# modern, AV1-capable FFmpeg (libdav1d), no system install required.

# hf_libero prompts interactively on first import ("custom dataset path?
# Y/N") if ~/.libero/config.yaml doesn't exist yet, which hangs any
# non-interactive script (EOFError). Pre-write it with the package's own
# defaults so env.type=libero never blocks waiting for stdin. LIBERO's
# object/asset meshes (not bundled in the pip package) auto-download from
# HF Hub the first time a LiberoEnv is built.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    python - <<'PY'
import importlib.util
import os
import yaml

# Locate the package without importing it — importing libero.libero runs
# module-level code that itself prompts on stdin when this config is
# missing, which is exactly the chicken-and-egg problem we're avoiding.
spec = importlib.util.find_spec("libero.libero")
if spec and spec.submodule_search_locations:
    root = list(spec.submodule_search_locations)[0]
    cfg_dir = os.path.expanduser("~/.libero")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        yaml.dump(
            {
                "benchmark_root": root,
                "bddl_files": os.path.join(root, "bddl_files"),
                "init_states": os.path.join(root, "init_files"),
                "datasets": os.path.join(root, "../datasets"),
                "assets": os.path.join(root, "assets"),
            },
            f,
        )
PY
fi

# Where LeRobot caches datasets + Hub downloads (the LIBERO dataset alone
# is ~35 GB — point this at a big disk if ~/.cache is small).
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# Single root for all training/eval outputs (see README "Pipeline output").
export SO101_OUTPUT_DIR="${SO101_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

# Quieter, more reproducible tokenizer/threading behavior for training.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --- Real-rig access (harmless when no arms/headset are attached) ------
# Serial ports for the SO-101 buses (resets on reboot/replug).
for port in /dev/ttyACM0 /dev/ttyACM1; do
    if [ -e "$port" ] && [ ! -w "$port" ]; then
        echo "=> Enabling access to ${port} (sudo may prompt for password)..."
        sudo chmod 666 "$port"
    fi
done

# Meta Quest over adb (unauthorized until you accept the on-headset prompt).
if command -v adb &> /dev/null; then
    adb start-server &> /dev/null
    quest_status="$(adb devices | sed -n '2p' | awk '{print $2}')"
    case "$quest_status" in
        device)       echo "✓ Meta Quest connected and authorized." ;;
        unauthorized) echo "⚠️  Meta Quest connected but UNAUTHORIZED — accept 'Allow USB debugging' in the headset." ;;
        *)            echo "ℹ️  No Meta Quest over USB (fine for sim/training; pass --ip-address for network)." ;;
    esac
else
    echo "ℹ️  adb not installed (only needed for Meta Quest teleop)."
fi

# --- Readout ----------------------------------------------------------
python - <<'PY'
import shutil
try:
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    vram = ""
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram = f" ({gb:.1f} GB VRAM)"
    print(f"✓ Compute: {dev}{vram}")
except Exception as e:
    print(f"ℹ️  torch not importable yet: {e}")
print(f"✓ Disk free at HF cache: {shutil.disk_usage('.').free/1e9:.0f} GB")
PY

echo "✓ Environment ready."
echo "  PYTHONPATH set · MUJOCO_GL=${MUJOCO_GL:-<default>} · outputs -> ${SO101_OUTPUT_DIR}"
echo "  Run tools from tool/, or smoke-test training with:"
echo "     bash test/smoke_test_pipeline.sh"
