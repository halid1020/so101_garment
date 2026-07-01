#!/usr/bin/env bash
# Sourcing this script adds the project root and src/ to the PYTHONPATH
# allowing scripts in tool/ to be run from anywhere without import errors.

source venv/bin/activate

# Get the absolute path of the directory containing this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "✓ Environment ready. PYTHONPATH configured for Actoris Harena / VLA tools." # TODO: environment should be independent of Actoris Harena.
echo "  You can now run scripts from the tool/ directory."
