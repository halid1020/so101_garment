#!/usr/bin/env bash
# Does a ported policy train like the LeRobot one it came from?
#
# The cheap, permanent half of that question. The expensive half is a full-scale
# pair on a real card, which is a launch rather than a test; this one runs a few
# hundred steps on the CPU and fails the moment the two diverge, which is enough
# to catch a port that stopped being one.
#
# What it covers that the unit and integration tests do not: everything
# lerobot-train assembles around the model. test/unit/test_policy_ports.py
# compares the SOURCE, test/integration/test_policy_ports_checkpoints.py
# compares actions, loss and gradients on shared WEIGHTS -- neither one builds a
# processor pipeline, an optimiser preset, a scheduler or a dataloader, and
# neither exercises the plugin discovery that has to resolve a type this repo
# defines before draccus parses anything.
#
#   bash test/system/test_port_parity.sh --dataset-root /media/hdd/so101/cube-pnp-new
#   bash test/system/test_port_parity.sh --dataset-root <ds> --policy diffusion
#
# Needs a dataset on this machine. Skips -- rather than fails -- without one, so
# the tier still runs on a laptop with the collection drive unplugged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/venv/bin/python"

DATASET_ROOT=""
POLICY="act"
STEPS=200
BATCH=2
DEVICE="cpu"

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2;;
        --policy)       POLICY="$2";       shift 2;;
        --steps)        STEPS="$2";        shift 2;;
        --batch)        BATCH="$2";        shift 2;;
        --device)       DEVICE="$2";       shift 2;;
        -h|--help)      sed -n '2,22p' "$0"; exit 0;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done

if [ -z "$DATASET_ROOT" ] || [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "⏭  no dataset given or found — pass --dataset-root <dir>"
    echo "   (skipped, not failed: this tier runs on machines with no drive)"
    exit 0
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

"$PY" "$REPO_ROOT/tool/compare_port_training.py" \
    --dataset-root "$DATASET_ROOT" \
    --policy "$POLICY" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --device "$DEVICE"
