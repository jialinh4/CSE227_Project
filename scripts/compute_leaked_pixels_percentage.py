#!/usr/bin/env python
"""
Compute Leaked Pixels Percentage for a video, using:
- original video frames
- segmentation masks (from your pipeline)
- ground-truth background image (no person)

Definition (aligned with Weissberg-style intuition):
A pixel is considered "leaked" if at least in one frame:
  1) it looks like true background (frame is close to GT background), AND
  2) the segmentation mask labels it as foreground.

Such pixels correspond to real background that is incorrectly treated
as part of the caller, and would therefore bypass the virtual background.

Leaked Pixels Percentage = (# leaked pixels over all frames, union)
/ (total pixels in the frame)
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


def compute_leaked_pixels_percentage(
    video_path: str,
    mask_dir: str,
    gt_bg_path: str,
    bg_threshold: float = 25.0,
    seg_threshold: float = 0.5,
    warmup_frames: int = 0,
) -> float:
    """
    Args:
        video_path: path to the (baseline or defense) video.
        mask_dir: directory containing frame_000000.png style masks.
        gt_bg_path: ground-truth background image (no person).
        bg_threshold: max |frame - GT| (gray) to treat as "background visible".
        seg_threshold: alpha threshold to decide foreground from mask (0~1).
        warmup_frames: how many initial frames to skip.

    Returns:
        leaked_ratio: Leaked Pixels Percentage as a fraction in [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    gt_bg = load_img(gt_bg_path)
    gt_h, gt_w = gt_bg.shape[:2]
    gt_gray = cv2.cvtColor(gt_bg, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask_root = Path(mask_dir)
    if not mask_root.is_dir():
        raise FileNotFoundError(mask_root)

    leaked_union = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask_path = mask_root / f"frame_{frame_idx:06d}.png"
        if not mask_path.exists():
            # 一般意味着掩码结束了，直接停止
            break

        if frame_idx < warmup_frames:
            frame_idx += 1
            continue

        # resize 到 GT 大小（假设 GT 决定参考坐标系）
        frame = cv2.resize(frame, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(mask_path)
        mask_img = cv2.resize(mask_img, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

        alpha = mask_img.astype(np.float32) / 255.0
        fg_mask = alpha >= seg_threshold          # segmentation 认为是前景的地方

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = cv2.absdiff(frame_gray, gt_gray)
        # GT 看来是背景的地方（与 GT 足够接近）
        bg_visible = diff <= bg_threshold

        # 泄露条件：GT 说这里是背景，而 segmentation 把它当作前景
        leak_mask = fg_mask & bg_visible

        if leaked_union is None:
            leaked_union = leak_mask.copy()
        else:
            leaked_union |= leak_mask

        frame_idx += 1

    cap.release()

    if leaked_union is None:
        # 没有有效帧，就认为没有泄露
        return 0.0

    total_pixels = float(leaked_union.size)
    leaked_pixels = float(np.count_nonzero(leaked_union))
    leaked_ratio = leaked_pixels / total_pixels
    return leaked_ratio


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Leaked Pixels Percentage using video, masks, and GT background."
    )
    ap.add_argument("--video", required=True, help="Input video path (baseline or defense).")
    ap.add_argument("--mask_dir", required=True, help="Directory with frame_XXXXXX.png masks.")
    ap.add_argument("--gt_bg", required=True, help="Ground-truth background image (no person).")
    ap.add_argument("--bg_threshold", type=float, default=25.0,
                    help="Threshold on |frame-gray - gt-gray| to treat as background visible (default: 25).")
    ap.add_argument("--seg_threshold", type=float, default=0.5,
                    help="Alpha threshold (0~1) to treat mask as foreground (default: 0.5).")
    ap.add_argument("--warmup_frames", type=int, default=0,
                    help="Number of initial frames to skip (default: 0).")
    ap.add_argument("--out_txt", type=str, default="",
                    help="Optional text file to save Leaked Pixels Percentage.")
    args = ap.parse_args()

    leaked_ratio = compute_leaked_pixels_percentage(
        video_path=args.video,
        mask_dir=args.mask_dir,
        gt_bg_path=args.gt_bg,
        bg_threshold=args.bg_threshold,
        seg_threshold=args.seg_threshold,
        warmup_frames=args.warmup_frames,
    )

    leaked_pct = leaked_ratio * 100.0
    print("=== Leaked Pixels Percentage ===")
    print(f"Video:   {args.video}")
    print(f"Masks:   {args.mask_dir}")
    print(f"GT bg:   {args.gt_bg}")
    print(f"Leaked Pixels Percentage: {leaked_pct:.4f} %")

    if args.out_txt:
        out_p = Path(args.out_txt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            f.write(f"Leaked Pixels Percentage: {leaked_pct:.6f}\n")
        print(f"[INFO] saved to {out_p}")


if __name__ == "__main__":
    main()
