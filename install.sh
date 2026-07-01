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
pip install -e ".[dataset,pi]"

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
