# CSE227 Defense-Integrated Video Pipeline (OpenCV-based)

This repository now provides a **modular, defense-oriented video processing pipeline** built on **OpenCV + Typer**, designed for the **CSE227 Fall 2025** project.

It preserves the original `vaderx` editing pipeline for reference while introducing a new, independent **defense pipeline** and **dataset management system** that together form the foundation for later attack–defense experiments.

---

## 🎯 Purpose

The updated pipeline aims to:

- Provide a **reproducible defense baseline** for video-leakage and privacy research.  
- Support modular extensions for **attack baselines (T3)** and **defense strategies (T4)** without breaking legacy code.  
- Maintain strict Git hygiene—only code and configs are versioned; all videos, frames, and generated results stay local.

---

## ⚙️ Current Features

✅ **Defense Pipeline (New)**
- Implements three core *defense mechanisms* for alpha masks:
  - **Erosion:** morphological shrinking of mask regions.  
  - **Boundary Noise:** adds Gaussian noise to boundary bands.  
  - **Temporal Jitter:** randomly shifts masks slightly across frames.  
- Outputs both *defended alpha masks* and *composited preview frames*.  
- Fully CLI-driven and dataset-configurable.  
- Compatible with downstream evaluation (e.g., leakage measurement).

✅ **Dataset Registry**
- Centralized config file (`configs/datasets.yml`) manages all video sources, frame paths, and output directories.
- Allows batch preparation (frame extraction + background-subtraction mask generation).

✅ **Automation Tools**
- `scripts/prepare_data.py` — auto-extract frames & generate masks from raw `.mp4`.  
- `scripts/run_defenses.sh` — run a single defense on one dataset.  
- `scripts/bench_defenses.py` — run a full parameter grid across all defenses and export CSV summary.  
- `scripts/export_defended_video.sh` — combine defended frames into final `.mp4` previews.

✅ **Legacy Editing Pipeline**
- The original OpenCV `vaderx/` editing system remains intact and runnable (`scripts/run_edit.sh`).

---

## 🧩 Repository Structure

```
CSE227_Project/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
│
├─ configs/
│  ├─ datasets.yml           # Central dataset registry
│  └─ defenses_grid.yml      # Parameter grid for batch defense evaluation
│
├─ scripts/
│  ├─ dev_install.sh         # Create venv & install deps
│  ├─ prepare_data.py        # Extract frames & auto-generate masks
│  ├─ run_defenses.sh        # Run one defense on a dataset
│  ├─ bench_defenses.py      # Batch all defense configs → CSV
│  ├─ export_defended_video.sh  # Export defended frames → .mp4
│  └─ run_edit.sh            # Legacy edit demo
│
├─ src/
│  ├─ pipeline.py            # New defense pipeline (CLI)
│  ├─ config.py              # Dataset registry loader
│  └─ defense/
│      ├─ __init__.py
│      ├─ defenses.py        # Core defense algorithms
│      ├─ io.py              # I/O helpers
│
│  └─ vaderx/                # Legacy OpenCV editing module (unchanged)
│      ├─ io.py
│      ├─ ops.py
│      ├─ pipeline.py
│      └─ cli.py
│
├─ data/
│  ├─ raw/                   # Original .mp4 files (e.g., test1.mp4)
│  ├─ interim/
│  │   ├─ frames/<video_id>/  # Extracted frames
│  │   └─ masks/<video_id>/   # Auto-generated or real alpha masks
│  └─ processed/              # Reserved for future preprocessing
│
├─ outputs/
│  ├─ defenses/<video_id>/<variant>/        # Defended alpha masks
│  ├─ frames_defended/<video_id>/<variant>/ # Composited preview frames
│  ├─ videos/<video_id>_<variant>.mp4       # Exported defended videos
│  └─ bench_<video_id>.csv                  # Batch-run results summary
│
└─ tests/
   └─ test_defenses.py        # Unit tests for erosion and I/O
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
bash scripts/dev_install.sh
```
Creates `.venv` and installs dependencies.

### 2. Register Your Dataset
Edit or extend `configs/datasets.yml`:
```yaml
datasets:
  test1:
    raw_video: data/raw/test1.mp4
    frames_dir: data/interim/frames/test1
    masks_dir: data/interim/masks/test1
    outputs_root: outputs
    fps: 24
```

### 3. Prepare Data (extract frames & generate masks)
```bash
python scripts/prepare_data.py --dataset test1
```
This:
- Extracts frames from `data/raw/test1.mp4` to `data/interim/frames/test1/`
- Generates background-subtraction masks to `data/interim/masks/test1/`

### 4. Run a Defense
```bash
python -m pipeline --dataset test1 --defense erosion --params "radius: 9" --seed 1234
```
Outputs:
```
outputs/defenses/test1/erosion_r9/*.png
outputs/frames_defended/test1/erosion_r9/*.png
```

### 5. Export Video
```bash
python scripts/export_defended_video.py test1 erosion_r9 24
```
Creates:
```
outputs/videos/test1_erosion_r9.mp4
```

### 6. Batch All Defenses
```bash
python scripts/bench_defenses.py --dataset test1
```
Generates:
```
outputs/bench_test1.csv
```
containing all defense configurations and output directories.

---

## 🧪 Run with Dummy Data

Use a tiny synthetic dataset (no `data/raw` needed) to smoke-test the defense pipeline.

### 1) Generate dummy frames & masks
Save as `scripts/gen_dummy_data.py` and run:
```bash
source .venv/bin/activate
python scripts/gen_dummy_data.py
# Outputs:
# data/interim/frames/test1/%06d.png
# data/interim/masks/test1/%06d.png
```

### 2) Run a defense (no dataset mode)
```bash
export PYTHONPATH="src:${PYTHONPATH:-}"
python -m pipeline   --video-id test1   --masks-dir data/interim/masks   --frames-dir data/interim/frames   --defense erosion   --params "radius: 9"   --seed 1234
# Outputs:
# outputs/defenses/test1/erosion_r9/*.png
# outputs/frames_defended/test1/erosion_r9/*.png
```

### 3) Export a preview video (optional)
```bash
python scripts/export_defended_video.py test1 erosion_r9 24
# Output:
# outputs/videos/test1_erosion_r9.mp4
```

**Notes**
- Ensure `export PYTHONPATH="src:${PYTHONPATH:-}"` before running the pipeline.
- You can switch to dataset mode later by adding a `dummy` entry in `configs/datasets.yml` and calling `python -m pipeline --dataset dummy ...`.


---


## 🧹 Cleaning Generated Files

To keep the repository clean and lightweight, you can automatically remove all generated intermediate and output files while preserving your **raw videos** and required directory structure.

Run:
```bash
bash scripts/clean_generated.sh
```

This will:

- **Delete all generated content** under:
  - `data/interim/**`
  - `data/processed/**`
  - `outputs/**`
  - (legacy) `data/frames_defended/**`
  - (legacy) top-level `defenses/**`
- **Preserve**:
  - `data/raw/**` (your original videos)
  - Required folders (with `.gitkeep` placeholders)

Optional flags:
```bash
--dry-run   # Preview actions without deleting anything
--yes       # Skip confirmation prompt and clean directly
```

Example:
```bash
bash scripts/clean_generated.sh --dry-run   # just preview
bash scripts/clean_generated.sh --yes       # clean immediately
```

After cleaning, the minimal directory tree is automatically recreated:
```
data/
  ├─ raw/
  ├─ interim/
  │   ├─ frames/
  │   └─ masks/
  └─ processed/
outputs/
  ├─ defenses/
  ├─ frames_defended/
  └─ videos/
```

This keeps your repository tidy while maintaining all necessary structure for subsequent runs.

---

## 🧠 Defense Mechanisms (Summary)

| Defense Type | Description | Key Params | Effect |
|---------------|--------------|-------------|---------|
| **Erosion** | Morphological shrink of alpha region | `radius` | Removes edge detail |
| **Boundary Noise** | Adds Gaussian noise within edge band | `sigma`, `band_width` | Breaks clean boundary predictability |
| **Temporal Jitter** | Randomly shifts alpha masks across frames | `shift_px`, `prob` | Reduces temporal consistency |

---

## 🧰 Development Notes

- All generated outputs (`data/interim/**`, `outputs/**`) are **ignored by `.gitignore`** — safe to keep local.  
- The legacy `vaderx/` pipeline remains runnable for testing independent editing operations.  
- Defense pipeline is fully modular: new methods can be added to `src/defense/defenses.py` and automatically picked up by the CLI.

---

## 💡 Next Steps

- Integrate **attack baseline (T3)** outputs into the same dataset registry for cross-comparison.  
- Add **metrics evaluation (ELR / BER)** script to measure leakage per variant.  
- Optionally add GPU acceleration for batch defenses.

---

**Course:** CSE 227 — Fall 2025  
**Contributors:** Team Defense (Jialin He et al.)
