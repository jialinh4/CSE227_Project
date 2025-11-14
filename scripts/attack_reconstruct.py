#!/usr/bin/env python
"""
Reconstruction attack:
RGB temporal fusion of leaked background regions.

相比简单 median，我们对每个像素做“带权平均”：
只在 region_mask == 1 的帧上累加，再除以出现次数，
避免少量泄露被大量 0 冲掉。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from vaderx.io import open_video  # noqa: E402
from vaderx.leakage import LeakageMetricsConfig  # noqa: E402
from vaderx.leakage import _load_mask_index  # noqa: E402  # type: ignore


def reconstruct_background(
    video_path: str,
    mask_dir: str,
    out_path: str,
    mask_threshold: float = 0.5,
    use_ring: bool = False,
) -> None:
    mask_root = Path(mask_dir)
    mask_index = _load_mask_index(mask_root)

    cap, fps, (w, h), total = open_video(video_path)

    cfg = LeakageMetricsConfig(mask_threshold=mask_threshold)

    # 累加器：sum / count
    sum_B = None
    sum_G = None
    sum_R = None
    cnt    = None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mask_path = mask_index.get(frame_idx)
            if mask_path is None:
                frame_idx += 1
                continue

            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(mask_path)
            if mask_img.shape[:2] != frame.shape[:2]:
                mask_img = cv2.resize(
                    mask_img,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            alpha = mask_img.astype(np.float32) / 255.0
            fg_mask = alpha >= cfg.mask_threshold
            bg_mask = ~fg_mask

            if use_ring and cfg.ring_width > 0:
                kernel_size = max(1, cfg.ring_width * 2 + 1)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                fg_uint = fg_mask.astype(np.uint8)
                dilated = cv2.dilate(fg_uint, kernel, iterations=1)
                ring = cv2.subtract(dilated, fg_uint)
                region_mask = (ring.astype(bool) & bg_mask).astype(np.float32)
            else:
                region_mask = bg_mask.astype(np.float32)

            if not np.any(region_mask):
                frame_idx += 1
                continue

            frame_f = frame.astype(np.float32)
            B = frame_f[:, :, 0]
            G = frame_f[:, :, 1]
            R = frame_f[:, :, 2]

            # 初始化累加器
            if sum_B is None:
                sum_B = np.zeros_like(B, dtype=np.float32)
                sum_G = np.zeros_like(G, dtype=np.float32)
                sum_R = np.zeros_like(R, dtype=np.float32)
                cnt   = np.zeros_like(B, dtype=np.float32)

            sum_B += B * region_mask
            sum_G += G * region_mask
            sum_R += R * region_mask
            cnt   += region_mask

            frame_idx += 1
    finally:
        cap.release()

    if sum_B is None or cnt is None:
        raise RuntimeError(f"No frames contributed to reconstruction for {video_path}")

    eps = 1e-6
    B_rec = sum_B / (cnt + eps)
    G_rec = sum_G / (cnt + eps)
    R_rec = sum_R / (cnt + eps)

    fused = np.stack([B_rec, G_rec, R_rec], axis=2)
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fused)
    print(f"✔ Reconstruction attack image saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RGB temporal reconstruction attack.")
    p.add_argument("--video", required=True, help="Input video path (Zoom output).")
    p.add_argument("--mask_dir", required=True, help="Directory with frame_*.png alpha masks.")
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--use_ring", action="store_true",
                   help="Only reconstruct on edge ring region.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct_background(
        video_path=args.video,
        mask_dir=args.mask_dir,
        out_path=args.out,
        mask_threshold=args.mask_threshold,
        use_ring=args.use_ring,
    )


if __name__ == "__main__":
    main()
