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


**Course:** CSE 227 — Fall 2025
# 实验准备
实验前没有  .venu 的话先激活
```bash
source .venv/bin/activate
```
# defenses实验前baseline准备：
01: 对目标baseline视频生成掩码
```bash
python -m vaderx.cli segment \
  --in_path data/raw/test1.mp4 \
  --mask_root masks \
  --backend auto \
  --smooth_kernel 1
```
02：利用掩码生成Baseline的 elr,ber
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
03：Baseline的重建攻击,生成效果图
```bash
  python scripts/attack_reconstruct.py \
  --video data/raw/test1.mp4 \
  --mask_dir masks/test1 \
  --out results/attack/blur1_baseline_ring_attack.png \
  --use_ring \
  --mask_threshold 0.5
```
# Defenses
1.Jitter
01:对目标视频生成防御掩码：
```bash
python scripts/defenses.py --mask_dir masks/test1 --out_dir defenses/test1/jitter_s2_p0.6 --method jitter --max_shift 2 --prob 0.6
```

02：用jitter掩码生成blackout 防御视频：
```bash
python scripts/compose_defense_video.py \
  --raw_video data/raw/test1.mp4 \
  --raw_mask_dir masks/test1 \
  --defense_mask_dir defenses/test1/jitter_s2_p0.6 \
  --output defense_videos/blur1_jitter_black.mp4
```

03: 对防御视频生成新掩码
```bash
python -m vaderx.cli segment \
  --in_path defense_videos/blur1_jitter_black.mp4 \
  --mask_root masks \
  --backend auto \
  --smooth_kernel 1
```

04：对防御视频测ELR，BER
```bash
python scripts/measure_leakage.py \
  --video defense_videos/blur1_jitter_black.mp4 \
  --mask_dir masks/blur1_jitter_black \
  --results_dir results \
  --video_id blur1_jitter_black \
  --warmup_frames 30 \
  --snapshot_dir results/snapshots \
  --snapshot_elr_threshold 0.85 \
  --snapshot_ber_threshold 0.6
```

05：重建攻击防御视频，生成效果图：
```bash
python scripts/attack_reconstruct.py \
  --video defense_videos/blur1_jitter_black.mp4 \
  --mask_dir masks/blur1_jitter_black \
  --out results/attack/blur1_jitter_black_attack.png \
  --use_ring \
  --mask_threshold 0.5
```



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

