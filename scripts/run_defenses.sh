#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
export PYTHONPATH="src:${PYTHONPATH:-}"

# Use dataset registry
python -m pipeline \
  --dataset test1 \
  --defense erosion \
  --params "radius: 9" \
  --seed 1234

echo "✔ Successfully executed run_defenses.sh"
