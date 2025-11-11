#!/usr/bin/env bash
# Clean generated artifacts while preserving data/raw and required directories.
# Only English comments and print messages.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
OUTPUTS="${ROOT}/outputs"

DRY_RUN=false
ASSUME_YES=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    *) echo "[error] Unknown option: $arg"; exit 2 ;;
  esac
done

say() { echo "[clean] $*"; }
do_or_echo() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

ensure_dir() {
  local d="$1"
  do_or_echo "mkdir -p \"$d\""
  # keep directory in Git if user wants; safe no-op otherwise
  do_or_echo "touch \"$d/.gitkeep\""
}

purge_dir_contents() {
  # Remove all contents under the given directory, but keep the directory itself.
  local d="$1"
  if [ -d "$d" ]; then
    # Avoid accidentally expanding to root if pattern empty
    do_or_echo "find \"$d\" -mindepth 1 -maxdepth 1 -exec rm -rf {} +"
  fi
}

confirm() {
  $ASSUME_YES && return 0
  read -r -p "[confirm] This will remove generated files under data/interim, data/processed, outputs (and legacy folders). Continue? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) return 0 ;;
    *) echo "[abort] No changes made."; exit 0 ;;
  esac
}

# --- Show plan ---------------------------------------------------------------
say "Repository root: $ROOT"
say "Target folders:"
echo "  - ${DATA}/interim/**"
echo "  - ${DATA}/processed/**"
echo "  - ${OUTPUTS}/**"
echo "  - (legacy) ${DATA}/frames_defended/**"
echo "  - (legacy) ${ROOT}/defenses/**"
echo
say "Preserved:"
echo "  - ${DATA}/raw/** (ALL files kept)"
echo "  - Directory structure will be recreated with .gitkeep"

confirm

# --- Perform cleanup ---------------------------------------------------------
say "Cleaning generated artifacts..."

# Ensure base directories exist
do_or_echo "mkdir -p \"$DATA\" \"$OUTPUTS\""

# 1) data/interim/** and data/processed/**
purge_dir_contents "${DATA}/interim"
purge_dir_contents "${DATA}/processed"

# Also remove nested subdirs just in case users created deeper trees
# but keep top-level directories themselves intact.
# (Already handled by purge_dir_contents)

# 2) outputs/**
purge_dir_contents "${OUTPUTS}"

# 3) legacy locations
# data/frames_defended/** (legacy before outputs/)
if [ -d "${DATA}/frames_defended" ]; then
  do_or_echo "rm -rf \"${DATA}/frames_defended\""
fi
# top-level defenses/** (legacy before outputs/defenses)
if [ -d "${ROOT}/defenses" ]; then
  do_or_echo "rm -rf \"${ROOT}/defenses\""
fi

# --- Recreate required structure --------------------------------------------
say "Recreating minimal directory structure..."

# Data side
ensure_dir "${DATA}/raw"                    # keep raw videos, untouched
ensure_dir "${DATA}/interim"
ensure_dir "${DATA}/interim/frames"
ensure_dir "${DATA}/interim/masks"
ensure_dir "${DATA}/processed"

# Outputs side
ensure_dir "${OUTPUTS}"
ensure_dir "${OUTPUTS}/defenses"
ensure_dir "${OUTPUTS}/frames_defended"
ensure_dir "${OUTPUTS}/videos"

say "Done."
$DRY_RUN && echo "[note] This was a dry run. Re-run without --dry-run to apply changes."
