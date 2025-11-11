#!/usr/bin/env python3
# Only English comments and print messages.
import os, glob, cv2
import numpy as np

VIDEO_ID = "test1"
frames_dir = f"data/interim/frames/{VIDEO_ID}"
mask_dir = f"data/interim/masks/{VIDEO_ID}"
os.makedirs(mask_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
if not files:
    print("[error] No frames found. Run ffmpeg extract first.")
    raise SystemExit(2)

# sample frames (avoid huge mem usage)
sample_paths = files if len(files) <= 200 else files[::max(1, len(files)//200)]
sample_imgs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in sample_paths]
sample_imgs = [img for img in sample_imgs if img is not None]
stack = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8) for img in sample_imgs], axis=0)
bg = np.median(stack, axis=0).astype(np.uint8)

# create masks by thresholding absolute diff
for i, fp in enumerate(files):
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg)
    # adaptive thresholding parameters can be tuned
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # optional morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    outp = os.path.join(mask_dir, f"{i:06d}.png")
    cv2.imwrite(outp, mask)

print(f"[done] Wrote {len(files)} masks to {mask_dir}")
