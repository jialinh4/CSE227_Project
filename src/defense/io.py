# coding: utf-8
"""
I/O helpers for defense pipeline.
Only English comments and print messages.
"""
from __future__ import annotations
import os, glob
from typing import List, Optional, Tuple
import numpy as np
import cv2

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _sorted_pngs(dirpath: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dirpath, "*.png")))

def load_alpha_sequence(dirpath: str) -> List[np.ndarray]:
    files = _sorted_pngs(dirpath)
    seq: List[np.ndarray] = []
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        a = img.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        seq.append(a)
    return seq

def save_alpha_sequence(seq: List[np.ndarray], out_dir: str):
    ensure_dir(out_dir)
    for idx, a in enumerate(seq):
        a = np.clip(a, 0.0, 1.0)
        img = (a * 255.0).astype(np.uint8)
        fp = os.path.join(out_dir, f"{idx:06d}.png")
        cv2.imwrite(fp, img)

def _sorted_frames(dirpath: str) -> List[str]:
    # supports png/jpg
    files = sorted(glob.glob(os.path.join(dirpath, "*.png")) + glob.glob(os.path.join(dirpath, "*.jpg")))
    return files

def load_video_frames_optional(dirpath: str) -> List[np.ndarray]:
    files = _sorted_frames(dirpath)
    seq: List[np.ndarray] = []
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        seq.append(img)
    return seq

def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 2:
        a3 = np.stack([alpha, alpha, alpha], axis=2)
    else:
        a3 = alpha
    a3 = a3.astype(np.float32)
    if a3.max() > 1.0:
        a3 = a3 / 255.0
    fg = fg.astype(np.float32) / 255.0
    bg = bg.astype(np.float32) / 255.0
    out = a3 * fg + (1.0 - a3) * bg
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

def save_frames_sequence(frames: List[np.ndarray], alphas: List[np.ndarray], out_dir: str, resize_to: Optional[Tuple[int,int]]=None):
    ensure_dir(out_dir)
    n = min(len(frames), len(alphas))
    for i in range(n):
        f = frames[i]
        a = alphas[i]
        if resize_to:
            f = cv2.resize(f, resize_to, interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, resize_to, interpolation=cv2.INTER_NEAREST)
        # Here we simply darken background to visualize alpha; replace with real bg if needed
        bg = np.zeros_like(f)
        out = composite(f, bg, a)
        fp = os.path.join(out_dir, f"{i:06d}.png")
        cv2.imwrite(fp, out)
