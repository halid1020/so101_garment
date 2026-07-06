#!/usr/bin/env bash
# Hardware & Software Installation Script for Vision-Tactile World Model
# Author: Abudureyimu Halite

set -e

echo "=================================================="
echo "🚀 Initializing Workspace Installation..."
echo "=================================================="

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "=> Creating Python 3 virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "=> Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "=> Installing local requirements..."
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found."
fi

# SURGICAL CHANGE: Build LeRobot from source in the parallel directory
echo "=> Setting up local LeRobot v0.5.1 environment..."
cd ..
if [ ! -d "lerobot" ]; then
    echo "=> Cloning LeRobot repository..."
    git clone https://github.com/huggingface/lerobot.git
fi

cd lerobot
echo "=> Checking out git log hash 3dd19d043e2f3fe5673b13ea0ebe4f31884c0797.."
git checkout 3dd19d043e2f3fe5673b13ea0ebe4f31884c0797

echo "=> Installing LeRobot in editable mode with [dataset,pi] extras..."
pip install -e ".[feetech, dataset,pi]"

# Clone and install the Meta Quest reader
echo "=> Setting up Meta Quest teleop reader..."
cd ..
if [ ! -d "meta_quest_teleop" ]; then
    echo "=> Cloning meta_quest_teleop repository..."
    git clone https://github.com/NeuracoreAI/meta_quest_teleop.git
fi

cd meta_quest_teleop
pip install -e .

# Install adb (Android Debug Bridge) — required to talk to the Meta Quest over USB
if ! command -v adb &> /dev/null; then
    echo "=> Installing adb (needed for Meta Quest USB connection, sudo may prompt)..."
    sudo apt install -y android-tools-adb
fi

# IK + visualization dependencies:
#   pin-pink   - Pink IK solver on Pinocchio (pulls pin/eigenpy)
#   quadprog   - QP solver used by Pink (SOLVER_NAME in configs)
#   viser      - web-based 3D visualizer for the tuning UI
#   yourdfpy   - URDF loading for the visualizer
echo "=> Installing IK and visualization dependencies..."
pip install pin-pink quadprog viser yourdfpy

# Return to the main project directory
cd ../so101_garment

# Install Pre-commit hooks for industrial-grade commits
echo "=> Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

echo "=================================================="
echo "✓ Installation Complete."
echo "Please run: source venv/bin/activate"
echo "=================================================="
