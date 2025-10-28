#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

python -m vaderx.cli \
  --in_path data/raw/test1.mp4 \
  --out_path outputs/test1_edit.mp4 \
  --start 0.0 --end -1 \
  --resize 1280x720 \
  --gray

echo "✔ Wrote outputs/test1_edit.mp4"
