#!/usr/bin/env python3
# Only English comments and print messages.
import os, sys, glob, subprocess, shlex
import cv2, numpy as np
import argparse
from typing import List
from pathlib import Path

sys.path.insert(0, "src")
from config import load_datasets_cfg, get_dataset  # noqa

def run_ffmpeg_extract(raw_mp4: str, out_frames_dir: str, fps: int | None = None):
    Path(out_frames_dir).mkdir(parents=True, exist_ok=True)
    if fps and fps > 0:
        cmd = f'ffmpeg -y -i "{raw_mp4}" -r {fps} -vsync 0 "{out_frames_dir}/%06d.png"'
    else:
        cmd = f'ffmpeg -y -i "{raw_mp4}" -vsync 0 "{out_frames_dir}/%06d.png"'
    print(f"[ffmpeg] {cmd}")
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg extract failed")

def gen_bgsub_masks(frames_dir: str, masks_dir: str, max_samples: int = 200, thr: int = 30, open_kernel: int = 5):
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No frames found: {frames_dir}")

    sample_paths = files if len(files) <= max_samples else files[::max(1, len(files)//max_samples)]
    sample_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sample_paths]
    sample_imgs = [img for img in sample_imgs if img is not None]
    stack = np.stack(sample_imgs, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    for i, fp in enumerate(files):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        diff = cv2.absdiff(img, bg)
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        outp = os.path.join(masks_dir, f"{i:06d}.png")
        cv2.imwrite(outp, mask)
    print(f"[done] Wrote masks to {masks_dir}")

def main():
    ap = argparse.ArgumentParser(description="Prepare datasets: extract frames and generate masks by background subtraction.")
    ap.add_argument("--dataset", type=str, help="Dataset name as defined in configs/datasets.yml")
    ap.add_argument("--all", action="store_true", help="Process all datasets")
    ap.add_argument("--thr", type=int, default=30, help="Threshold for foreground")
    ap.add_argument("--open-kernel", type=int, default=5, help="Morphological open kernel size")
    ap.add_argument("--max-samples", type=int, default=200, help="Max frames sampled to estimate median background")
    ap.add_argument("--skip-extract", action="store_true", help="Skip ffmpeg extraction if frames already exist")
    ap.add_argument("--skip-masks", action="store_true", help="Skip mask generation")
    args = ap.parse_args()

    cfg_all = load_datasets_cfg()
    names = list(cfg_all.keys()) if args.all else [args.dataset]
    if not names or names == [None]:
        ap.error("Please provide --dataset <name> or --all")

    for name in names:
        cfg = cfg_all[name]
        raw_video = cfg.get("raw_video")
        frames_dir = cfg.get("frames_dir")
        masks_dir = cfg.get("masks_dir")
        fps = int(cfg.get("fps", 0)) or None
        if not raw_video or not frames_dir or not masks_dir:
            print(f"[skip] Incomplete config for dataset {name}")
            continue

        # Extract frames
        if args.skip_extract:
            print(f"[skip] extract for {name}")
        else:
            # Skip extract if frames already exist
            if glob.glob(os.path.join(frames_dir, "*.png")):
                print(f"[info] frames exist: {frames_dir} (skipping extract)")
            else:
                run_ffmpeg_extract(raw_video, frames_dir, fps=fps)

        # Generate masks
        if args.skip_masks:
            print(f"[skip] masks for {name}")
        else:
            gen_bgsub_masks(frames_dir, masks_dir, max_samples=args.max_samples, thr=args.thr, open_kernel=args.open_kernel)

if __name__ == "__main__":
    main()
