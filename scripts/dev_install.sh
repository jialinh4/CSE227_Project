#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .  
echo "✔ venv ready. Run: source .venv/bin/activate"
