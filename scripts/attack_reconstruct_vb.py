#!/usr/bin/env python
"""
Virtual-background-aware reconstruction attack.

假设攻击者：
- 可以获取 Zoom 输出视频（带虚拟背景的会议录制）；
- 可以获得每帧的人像分割掩码（通过自己的 segmentation 模型）；
- 知道 Zoom 使用的虚拟背景图片 V（默认图是公开的，或者会前截一张“无人画面”即可）。

攻击目标：
- 在人物轮廓附近 / 背景侧，找出那些“与虚拟背景图 V 明显不同且在时间上稳定”的像素，
  这些像素很可能是真实背景或其 halo。
- 输出两张图：
  1) recon：这些泄露像素的平均颜色（大致重建图）
  2) heatmap：泄露强度热力图，用于高亮泄露位置
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_mask(mask_dir: Path, frame_idx: int, size_hw: tuple[int, int]) -> np.ndarray | None:
    """读取 masks/frame_XXXXXX.png，返回 float32 alpha in [0,1]，大小匹配视频。"""
    mask_path = mask_dir / f"frame_{frame_idx:06d}.png"
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = size_hw
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = m.astype(np.float32) / 255.0
    return alpha


def compute_region_mask(
    alpha: np.ndarray,
    use_ring: bool,
    mask_threshold: float,
    ring_width: int = 8,
) -> np.ndarray:
    """
    根据 alpha 生成“候选泄露区域”二值掩码 (bool)。

    - 前景：alpha >= mask_threshold
    - 背景：alpha < mask_threshold
    - use_ring=False: 直接用背景侧作为候选区域
    - use_ring=True : 只取前景外侧的一圈 ring ∩ 背景侧
    """
    fg = alpha >= mask_threshold
    bg = ~fg

    if not use_ring:
        return bg

    fg_uint = fg.astype(np.uint8)
    ksize = max(1, ring_width * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(fg_uint, kernel, iterations=1)
    ring = (dilated - fg_uint).clip(0, 1).astype(bool)

    region_mask = ring & bg
    return region_mask


def reconstruct_with_vb(
    video_path: str,
    mask_dir: str,
    vb_image: str,
    out_recon: str,
    out_heatmap: str,
    mask_threshold: float = 0.5,
    use_ring: bool = True,
    ring_width: int = 8,
    residual_threshold: float = 25.0,
    min_count: int = 3,
) -> None:
    """
    利用已知虚拟背景 V 的重建攻击。

    参数：
    - video_path: 含虚拟背景的 Zoom 录屏
    - mask_dir  : 自己算的人像掩码目录 (frame_XXXXXX.png)
    - vb_image  : 虚拟背景图片路径（攻击者已知）
    - out_recon : 输出重建图路径 (PNG)
    - out_heatmap: 输出泄露热力图路径 (PNG)
    - mask_threshold: alpha 判前景阈值
    - use_ring      : 是否只在前景外圈 ring 内考虑泄露
    - ring_width    : ring 半宽
    - residual_threshold: |frame - V| 的阈值，越大越“保守”
    - min_count     : 某像素至少在多少帧被检测为泄露，才算稳定泄露
    """

    video_path = Path(video_path)
    mask_dir = Path(mask_dir)
    vb_image = Path(vb_image)
    out_recon = Path(out_recon)
    out_heatmap = Path(out_heatmap)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not vb_image.exists():
        raise FileNotFoundError(f"Virtual background image not found: {vb_image}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame.")
    h, w = frame0.shape[:2]

    # 读虚拟背景图，并 resize 到视频大小
    vb = cv2.imread(str(vb_image), cv2.IMREAD_COLOR)
    if vb is None:
        raise RuntimeError(f"Failed to read vb image: {vb_image}")
    if vb.shape[:2] != (h, w):
        vb = cv2.resize(vb, (w, h), interpolation=cv2.INTER_LINEAR)
    vb = vb.astype(np.float32)

    # 累积真实泄露的像素 & 残差
    sum_B = np.zeros((h, w), dtype=np.float32)
    sum_G = np.zeros((h, w), dtype=np.float32)
    sum_R = np.zeros((h, w), dtype=np.float32)
    sum_mag = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.int32)

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[INFO] VB-aware reconstruction attack running...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        alpha = load_mask(mask_dir, frame_idx, (h, w))
        if alpha is None:
            print(f"[WARN] mask missing at frame {frame_idx}, stop.")
            break

        region_mask = compute_region_mask(
            alpha,
            use_ring=use_ring,
            mask_threshold=mask_threshold,
            ring_width=ring_width,
        )
        if not np.any(region_mask):
            frame_idx += 1
            continue

        F = frame.astype(np.float32)

        # 计算与虚拟背景图 V 的差异
        diff = F - vb                # (h,w,3)
        mag = np.sum(np.abs(diff), axis=2)  # L1 距离 (h,w)

        # 泄露候选：在背景/ring 区域，且与 V 的差异大于一定阈值
        leak_mask = (mag > residual_threshold) & region_mask

        if not np.any(leak_mask):
            frame_idx += 1
            continue

        B, G, R = cv2.split(F)

        sum_B[leak_mask] += B[leak_mask]
        sum_G[leak_mask] += G[leak_mask]
        sum_R[leak_mask] += R[leak_mask]
        sum_mag[leak_mask] += mag[leak_mask]
        cnt[leak_mask] += 1

        frame_idx += 1

    cap.release()

    eps = 1e-6
    # 只保留在足够多帧中被标记为泄露的像素
    stable = cnt >= min_count
    if not np.any(stable):
        print("[WARN] No stable leakage pixels found; outputs will be blank.")
        recon = np.zeros((h, w, 3), dtype=np.uint8)
        heat = np.zeros((h, w, 3), dtype=np.uint8)
        out_recon.parent.mkdir(parents=True, exist_ok=True)
        out_heatmap.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_recon), recon)
        cv2.imwrite(str(out_heatmap), heat)
        print(f"[INFO] Saved blank recon to {out_recon} and heatmap to {out_heatmap}")
        return

    # 重建图：泄露像素的平均颜色
    rec_B = np.zeros_like(sum_B)
    rec_G = np.zeros_like(sum_G)
    rec_R = np.zeros_like(sum_R)

    rec_B[stable] = sum_B[stable] / (cnt[stable].astype(np.float32) + eps)
    rec_G[stable] = sum_G[stable] / (cnt[stable].astype(np.float32) + eps)
    rec_R[stable] = sum_R[stable] / (cnt[stable].astype(np.float32) + eps)

    recon = np.stack([rec_B, rec_G, rec_R], axis=2)
    recon[~stable] = 0.0
    recon = np.clip(recon, 0, 255).astype(np.uint8)

    # 泄露热力图：平均残差
    mean_mag = np.zeros_like(sum_mag)
    mean_mag[stable] = sum_mag[stable] / (cnt[stable].astype(np.float32) + eps)

    max_mag = float(mean_mag.max())
    if max_mag > 0:
        norm = (mean_mag / max_mag * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(mean_mag, dtype=np.uint8)

    heat_gray = norm
    heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)
    # 把非 stable 区域抹黑
    heat_color[~stable] = 0

    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_heatmap.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_recon), recon)
    cv2.imwrite(str(out_heatmap), heat_color)

    n_leak = int(stable.sum())
    print(f"[INFO] Reconstruction saved to {out_recon}")
    print(f"[INFO] Heatmap saved to {out_heatmap}")
    print(f"[INFO] Stable leakage pixels: {n_leak}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VB-aware leakage reconstruction attack.")
    p.add_argument("--video", required=True, help="Zoom video with virtual background.")
    p.add_argument("--mask_dir", required=True, help="Directory with frame_*.png masks.")
    p.add_argument("--vb_image", required=True, help="Known virtual background image path.")
    p.add_argument("--out_recon", required=True, help="Output reconstructed leakage image (PNG).")
    p.add_argument("--out_heatmap", required=True, help="Output leakage heatmap image (PNG).")
    p.add_argument("--mask_threshold", type=float, default=0.5, help="Alpha FG threshold in [0,1].")
    p.add_argument("--use_ring", action="store_true", help="Only consider ring around FG boundary.")
    p.add_argument("--ring_width", type=int, default=8, help="Ring half-width in pixels.")
    p.add_argument(
        "--residual_threshold",
        type=float,
        default=25.0,
        help="L1 color distance threshold |frame-V| to treat as leakage (default: 25.0).",
    )
    p.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum #frames a pixel must be detected as leakage to be kept (default: 3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct_with_vb(
        video_path=args.video,
        mask_dir=args.mask_dir,
        vb_image=args.vb_image,
        out_recon=args.out_recon,
        out_heatmap=args.out_heatmap,
        mask_threshold=args.mask_threshold,
        use_ring=args.use_ring,
        ring_width=args.ring_width,
        residual_threshold=args.residual_threshold,
        min_count=args.min_count,
    )


if __name__ == "__main__":
    main()
