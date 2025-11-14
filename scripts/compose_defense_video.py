#!/usr/bin/env python
import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def compose_defense_video(
    raw_video: str,
    raw_mask_dir: str,
    defense_mask_dir: str,
    output_path: str,
    bg_color=(0, 0, 0)
) -> None:
    """Use original masks to extract foreground, defense masks to recomposite with safe background."""

    cap = cv2.VideoCapture(raw_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {raw_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0
    print(f"[INFO] composing defense video -> {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_mask_path = Path(raw_mask_dir) / f"frame_{frame_idx:06d}.png"
        defense_mask_path = Path(defense_mask_dir) / f"frame_{frame_idx:06d}.png"

        if not raw_mask_path.exists() or not defense_mask_path.exists():
            # 没有更多掩码就停
            print(f"[WARN] mask missing at frame {frame_idx}, stop.")
            break

        raw_alpha = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
        defense_alpha = cv2.imread(str(defense_mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_alpha is None or defense_alpha is None:
            print(f"[WARN] failed to read mask at frame {frame_idx}, stop.")
            break

        raw_alpha = raw_alpha.astype(np.float32) / 255.0
        defense_alpha = defense_alpha.astype(np.float32) / 255.0

        # 用原始 mask 提取前景
        fg = (frame.astype(np.float32) * raw_alpha[..., None]).astype(np.uint8)

        # 安全背景（纯色，你可以改成别的）
        bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # 用“防御后 alpha”重新合成视频
        out_frame = (
            fg.astype(np.float32) * defense_alpha[..., None]
            + bg.astype(np.float32) * (1.0 - defense_alpha[..., None])
        )
        out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)

        out.write(out_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[INFO] done. total frames: {frame_idx}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compose defense video using raw + defense masks."
    )
    p.add_argument("--raw_video", required=True, help="Path to original Zoom video.")
    p.add_argument("--raw_mask_dir", required=True, help="Dir with baseline masks.")
    p.add_argument("--defense_mask_dir", required=True, help="Dir with defense masks.")
    p.add_argument("--output", required=True, help="Output defended video path.")
    p.add_argument(
        "--bg_color",
        type=int,
        nargs=3,
        default=(0, 0, 0),
        help="Background B G R color (default: 0 0 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compose_defense_video(
        raw_video=args.raw_video,
        raw_mask_dir=args.raw_mask_dir,
        defense_mask_dir=args.defense_mask_dir,
        output_path=args.output,
        bg_color=tuple(args.bg_color),
    )


if __name__ == "__main__":
    main()
