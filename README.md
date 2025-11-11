# CSE227 Video Editing Pipeline (OpenCV-based)

This repository provides a minimal yet extensible **video editing pipeline** built on **OpenCV**.  
It is designed as a clean, modular starting point for future expansion—allowing the team to progressively add features such as stabilization, effects, segmentation, or sound overlays.

---

## 🎯 Purpose

The project aims to:
- Provide a reproducible, Python-based environment for basic video editing and processing.  
- Serve as a foundation for CSE227 project development (e.g., Poisson blending, visual analysis, and future ML components).  
- Maintain a clean Git structure—only code/configs are versioned, while all videos and data outputs are ignored.

---

## ⚙️ Current Features

- Load and edit sample videos (`test1.mp4`) using OpenCV.
- Apply simple transformations: grayscale conversion, resizing, and trimming by timestamp.
- Generate per-frame human alpha masks (MediaPipe Selfie Segmentation with DeepLabV3 fallback) under `masks/<video_id>/`.
- Compute baseline leakage metrics (ELR / BER) + plots from masks and source frames, with optional flagged-frame snapshots.
- Command-line interface powered by **Typer** for flexible use.
- Fully structured pipeline with isolated data stages (`raw`, `interim`, `processed`, `outputs`).
- `.gitignore` optimized to exclude all large/binary media files automatically.

---

## 🧩 Repository Structure

```
CSE227_Project/
├─ README.md               # Documentation (this file)
├─ requirements.txt        # Python dependencies
├─ pyproject.toml          # Package metadata
├─ .gitignore              # Ignores all video/data outputs
├─ configs/                # YAML configs for pipeline parameters
├─ scripts/                # Bash utilities for setup and runs
│  ├─ dev_install.sh       # Create virtual env & install dependencies
│  └─ run_edit.sh          # Example: edit and export test1.mp4
├─ src/
│  └─ vaderx/              # Main source code (OpenCV pipeline)
│     ├─ io.py             # Video I/O utilities
│     ├─ ops.py            # Frame-level operations
│     ├─ pipeline.py       # Main video processing flow
│     └─ cli.py            # Command-line interface entry
├─ data/
│  ├─ raw/                 # Original sample videos (e.g., test1.mp4)
│  ├─ interim/             # Intermediate outputs (e.g., frames, masks)
│  └─ processed/           # Preprocessed stable data for reuse
├─ masks/                  # Generated alpha mattes per video (auto-created)
├─ outputs/                # Final exported results (e.g., test1_edit.mp4)
├─ results/                # CSV + plots for leakage measurements (snapshots under results/snapshots/)
└─ tests/                  # Unit and smoke tests
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
bash scripts/dev_install.sh
```
Creates `.venv` and installs all dependencies.

### 2. Add Your Sample Video
Place your file in:
```
data/raw/test1.mp4
```

### 3. Run the Example Edit
```bash
bash scripts/run_edit.sh
```
or run manually:
```bash
source .venv/bin/activate
python -m vaderx.cli edit   --in_path data/raw/test1.mp4   --out_path outputs/test1_edit.mp4   --resize 1280x720   --gray
```

This reads `test1.mp4`, converts to grayscale, resizes to 1280×720, and saves to `outputs/test1_edit.mp4`.

### 4. Optional: Use Config File
Edit `configs/edit.sample.yaml`, then run:
```bash
python -m vaderx.cli edit --config configs/edit.sample.yaml
```

### 5. End-to-end leakage baseline via helper script
```bash
bash scripts/run_leakage_pipeline.sh data/raw/test1.mp4 test1_zoom
```
This single command activates the virtualenv (if present), ensures the `vaderx` package is installed locally, runs segmentation, and then measures ELR/BER. It writes:
- Masks: `masks/test1_zoom/frame_*.png`
- Metrics: `results/baseline_test1_zoom.csv` + `results/baseline_test1_zoom.png`
- Snapshots: `results/snapshots/test1_zoom/frame_*.png`

Tweak behavior by exporting environment variables before invoking, e.g. `SEG_BACKEND=deeplabv3 SNAPSHOT_ELR_THRESHOLD=0.6 bash scripts/run_leakage_pipeline.sh ...`. Leaving them unset lets the Python tooling use its built-in defaults.

### 6. Human Segmentation (alpha masks)
```bash
python -m vaderx.cli segment \
  --in_path data/raw/test1.mp4 \
  --mask_root masks \
  --backend auto \
  --smooth_kernel 5
```
Outputs `masks/test1/frame_*.png` alpha mattes plus metadata. `--backend auto` prefers MediaPipe and falls back to DeepLabV3 if MediaPipe is unavailable.

### 7. Leakage Metrics (ELR / BER)
```bash
python scripts/measure_leakage.py \
  --video data/raw/test1.mp4 \
  --mask_dir masks/test1 \
  --results_dir results \
  --warmup_frames 30 \
  --snapshot_dir results/snapshots \
  --snapshot_elr_threshold 0.85 \
  --snapshot_ber_threshold 0.6
```
Produces `results/baseline_test1.csv` (per-frame ELR/BER + summary) and `results/baseline_test1.png` (line plot).
The optional `--warmup_frames` flag suppresses the first N frames to avoid initialization spikes. Supplying `--snapshot_dir` also saves annotated frames under `results/snapshots/test1/` whenever ELR/BER spikes beyond the provided thresholds; those timestamps are highlighted on the plot.

> `bash scripts/run_leakage_pipeline.sh data/raw/test1.mp4 test1_zoom` already wires snapshots on by default, writing them to `results/snapshots/test1_zoom/`. Override paths/thresholds with `SNAPSHOT_DIR`, `SNAPSHOT_ELR_THRESHOLD`, etc. environment variables before running the script.


# defenses实验前baseline准备：
01: 对目标视频生成掩码
python -m vaderx.cli segment \
  --in_path data/raw/test1.mp4 \
  --mask_root masks \
  --backend auto \
  --smooth_kernel 5
  
02：利用掩码生成elr,ber
python scripts/measure_leakage.py \
  --video data/raw/test1.mp4 \
  --mask_dir masks/test1 \
  --results_dir results \
  --warmup_frames 30 \
  --snapshot_dir results/snapshots \
  --snapshot_elr_threshold 0.85 \ 
  --snapshot_ber_threshold 0.6
  
# Defenses
1.Jitter
01:对目标视频生成防御掩码：
python scripts/defenses.py --mask_dir masks/test1 --out_dir defenses/test1/jitter_s2_p0.6 --method jitter --max_shift 2 --prob 0.6

02：利用防御掩码计算elr,ber:
python scripts/measure_leakage.py \
  --video data/raw/test1.mp4 \
  --mask_dir defenses/test1/jitter_s2_p0.6 \
  --results_dir results \
  --video_id test1_jitter_s2_p0.6 \
  --warmup_frames 30 \
  --snapshot_dir results/snapshots \
  --snapshot_elr_threshold 0.85 \
  --snapshot_ber_threshold 0.6











### 8. Deactivate the virtual environment

To exit the virtual environment created by `scripts/dev_install.sh`:
```bash
deactivate
```

To activate it again later:
```bash
source .venv/bin/activate
```


---

## 🧰 Next Steps

- Extend `ops.py` with new effects (crop, fade, stabilize).  
- Add metrics logging and visualization under `outputs/logs/`.  
- Implement ML-assisted or Houdini-inspired pipelines for advanced editing.

---

**Course:** CSE 227 — Fall 2025
