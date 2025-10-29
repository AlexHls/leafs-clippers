#!/bin/bash
# Script to run tests for leafs-clippers

# Get the directory where this script is located (repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add src directory to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# Run pytest with all arguments passed to this script
python3 -m pytest "${REPO_ROOT}/tests" "$@"
