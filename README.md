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
├─ outputs/                # Final exported results (e.g., test1_edit.mp4)
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

---

## 📂 Directory Guide (Simplified)

| Directory | Purpose | Version Control |
|------------|----------|----------------|
| `src/` | All source code for pipeline and CLI | ✅ |
| `scripts/` | Helper scripts for environment & runs | ✅ |
| `configs/` | Config files for batch editing | ✅ |
| `data/raw/` | Original videos (e.g., `test1.mp4`) | 🚫 ignored |
| `data/interim/` | Temporary intermediate outputs | 🚫 ignored |
| `data/processed/` | Stable processed data | 🚫 ignored |
| `outputs/` | Final exported videos | 🚫 ignored |

---

## 🧰 Next Steps

- Extend `ops.py` with new effects (crop, fade, stabilize).  
- Add metrics logging and visualization under `outputs/logs/`.  
- Implement ML-assisted or Houdini-inspired pipelines for advanced editing.

---

**Course:** CSE 227 — Fall 2025
