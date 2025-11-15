#!/usr/bin/env python
"""
Compute Reconstructed Percentage for a reconstruction attack:

Reconstructed Percentage =
  (# pixels where reconstructed background is close to GT background)
/ (total pixels)

This measures how much of the real background the attack successfully
recovers, analogous to the reconstruction area in Weissberg et al.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def compute_reconstructed_percentage(
    gt_bg_path: str,
    recon_path: str,
    recon_threshold: float = 20.0,
) -> float:
    """
    Args:
        gt_bg_path: ground-truth background (no person).
        recon_path: reconstructed background image from attack.
        recon_threshold: max |recon - gt| (gray) to treat as "successfully reconstructed".

    Returns:
        ratio in [0, 1].
    """
    gt_bg = load_img(gt_bg_path)
    recon = load_img(recon_path)

    h, w = gt_bg.shape[:2]
    if recon.shape[:2] != (h, w):
        recon = cv2.resize(recon, (w, h), interpolation=cv2.INTER_LINEAR)

    gt_gray = cv2.cvtColor(gt_bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    recon_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = cv2.absdiff(recon_gray, gt_gray)
    recon_mask = diff <= recon_threshold

    total_pixels = float(recon_mask.size)
    recon_pixels = float(np.count_nonzero(recon_mask))

    return recon_pixels / total_pixels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Reconstructed Percentage using GT and reconstructed background."
    )
    ap.add_argument("--gt_bg", required=True, help="Ground-truth background image (no person).")
    ap.add_argument("--recon", required=True, help="Reconstructed background image from attack.")
    ap.add_argument("--recon_threshold", type=float, default=20.0,
                    help="Threshold on |recon-gray - gt-gray| to treat as 'reconstructed' (default: 20).")
    ap.add_argument("--out_txt", type=str, default="",
                    help="Optional text file to save Reconstructed Percentage.")
    args = ap.parse_args()

    ratio = compute_reconstructed_percentage(
        gt_bg_path=args.gt_bg,
        recon_path=args.recon,
        recon_threshold=args.recon_threshold,
    )

    pct = ratio * 100.0
    print("=== Reconstructed Percentage ===")
    print(f"GT background:  {args.gt_bg}")
    print(f"Reconstruction: {args.recon}")
    print(f"Reconstructed Percentage: {pct:.4f} %")

    if args.out_txt:
        out_p = Path(args.out_txt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            f.write(f"Reconstructed Percentage: {pct:.6f}\n")
        print(f"[INFO] saved to {out_p}")


if __name__ == "__main__":
    main()
