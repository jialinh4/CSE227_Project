#!/usr/bin/env bash
#
# Convenience wrapper to run segmentation + leakage measurement in one go.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Optional environment overrides
VIDEO_PATH="${VIDEO_PATH:-${1:-data/raw/test1.mp4}}"
VIDEO_ID="${VIDEO_ID:-${2:-$(basename "${VIDEO_PATH%.*}")}}"
MASK_ROOT="${MASK_ROOT:-masks}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEG_BACKEND="${SEG_BACKEND:-}"
SEG_STRIDE="${SEG_STRIDE:-}"
SEG_SMOOTH="${SEG_SMOOTH:-}"
RING_WIDTH="${RING_WIDTH:-}"
DIFF_THRESHOLD="${DIFF_THRESHOLD:-}"
STABILITY_WINDOW="${STABILITY_WINDOW:-}"
STABILITY_THRESHOLD="${STABILITY_THRESHOLD:-}"
MASK_THRESHOLD="${MASK_THRESHOLD:-}"
WARMUP_FRAMES="${WARMUP_FRAMES:-}"
if [[ -z "${SNAPSHOT_DIR+x}" ]]; then
  SNAPSHOT_DIR="results/snapshots"
fi
SNAPSHOT_ELR_THRESHOLD="${SNAPSHOT_ELR_THRESHOLD:-}"
SNAPSHOT_BER_THRESHOLD="${SNAPSHOT_BER_THRESHOLD:-}"
SNAPSHOT_MAX_COUNT="${SNAPSHOT_MAX_COUNT:-}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "✖ Video not found: $VIDEO_PATH" >&2
  exit 1
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("vaderx") else 1)
PY
  echo "Installing local package into current interpreter (pip install -e .)..."
  python -m pip install -e .
fi

SEG_ARGS=()
[[ -n "$SEG_BACKEND" ]] && SEG_ARGS+=(--backend "$SEG_BACKEND")
[[ -n "$SEG_STRIDE" ]] && SEG_ARGS+=(--stride "$SEG_STRIDE")
[[ -n "$SEG_SMOOTH" ]] && SEG_ARGS+=(--smooth_kernel "$SEG_SMOOTH")

echo "▶ Running segmentation for ${VIDEO_PATH} (id=${VIDEO_ID})"
python -m vaderx.cli segment \
  --in_path "$VIDEO_PATH" \
  --mask_root "$MASK_ROOT" \
  --video_id "$VIDEO_ID" \
  "${SEG_ARGS[@]}"

MASK_DIR="${MASK_ROOT}/${VIDEO_ID}"

MEASURE_ARGS=()
[[ -n "$RING_WIDTH" ]] && MEASURE_ARGS+=(--ring_width "$RING_WIDTH")
[[ -n "$DIFF_THRESHOLD" ]] && MEASURE_ARGS+=(--diff_threshold "$DIFF_THRESHOLD")
[[ -n "$STABILITY_WINDOW" ]] && MEASURE_ARGS+=(--stability_window "$STABILITY_WINDOW")
[[ -n "$STABILITY_THRESHOLD" ]] && MEASURE_ARGS+=(--stability_threshold "$STABILITY_THRESHOLD")
[[ -n "$MASK_THRESHOLD" ]] && MEASURE_ARGS+=(--mask_threshold "$MASK_THRESHOLD")
[[ -n "$WARMUP_FRAMES" ]] && MEASURE_ARGS+=(--warmup_frames "$WARMUP_FRAMES")
[[ -n "$SNAPSHOT_DIR" ]] && MEASURE_ARGS+=(--snapshot_dir "$SNAPSHOT_DIR")
[[ -n "$SNAPSHOT_ELR_THRESHOLD" ]] && MEASURE_ARGS+=(--snapshot_elr_threshold "$SNAPSHOT_ELR_THRESHOLD")
[[ -n "$SNAPSHOT_BER_THRESHOLD" ]] && MEASURE_ARGS+=(--snapshot_ber_threshold "$SNAPSHOT_BER_THRESHOLD")
[[ -n "$SNAPSHOT_MAX_COUNT" ]] && MEASURE_ARGS+=(--snapshot_max_count "$SNAPSHOT_MAX_COUNT")

echo "▶ Measuring ELR/BER from masks in ${MASK_DIR}"
python scripts/measure_leakage.py \
  --video "$VIDEO_PATH" \
  --mask_dir "$MASK_DIR" \
  --video_id "$VIDEO_ID" \
  --results_dir "$RESULTS_DIR" \
  "${MEASURE_ARGS[@]}"

echo "✔ Complete. Results at ${RESULTS_DIR}/baseline_${VIDEO_ID}.csv|.png"
