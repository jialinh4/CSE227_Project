#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
export PYTHONPATH="src:${PYTHONPATH:-}"

python -m pipeline \
  --video-id test1 \
  --masks-dir data/interim/masks \
  --frames-dir data/interim/frames \
  --defense erosion \
  --params "radius: 9" \
  --out-alpha-root outputs/defenses \
  --out-frames-root outputs/frames_defended

echo "✔ Successfully executed run_defenses.sh"
