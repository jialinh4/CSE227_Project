"""
Segmentation utilities for extracting per-frame human alpha mattes.

Primary backend uses MediaPipe Selfie Segmentation; automatically falls back
to DeepLabV3 (torchvision) if MediaPipe is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .io import open_video


class Segmenter:
    """Abstract base class for per-frame segmentation backends."""

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class MediaPipeSegmenter(Segmenter):
    """MediaPipe selfie segmentation backend."""

    def __init__(self, model_selection: int = 1):
        import mediapipe as mp  # type: ignore

        self._module = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._module.process(rgb)
        mask = np.asarray(result.segmentation_mask, dtype=np.float32)
        return np.clip(mask, 0.0, 1.0)


class DeepLabSegmenter(Segmenter):
    """torchvision DeepLab human segmentation fallback."""

    def __init__(self, device: Optional[str] = None):
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        self._model.to(self._device).eval()
        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        tensor = tensor.to(self._device)
        with self._torch.inference_mode():
            logits = self._model(tensor)["out"]
            probs = self._torch.softmax(logits, dim=1)
            person_prob = probs[0, 15, :, :]  # Pascal VOC class index for person
        mask = person_prob.cpu().numpy()
        mask = cv2.resize(
            mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        return np.clip(mask, 0.0, 1.0)


def build_segmenter(backend: str) -> Segmenter:
    backend = backend.lower()
    errors: Dict[str, Exception] = {}

    if backend in ("auto", "mediapipe"):
        try:
            return MediaPipeSegmenter()
        except Exception as exc:  # pragma: no cover - triggered when mediapipe missing
            errors["mediapipe"] = exc
            if backend != "auto":
                raise

    if backend in ("auto", "deeplab", "deeplabv3"):
        try:
            return DeepLabSegmenter()
        except Exception as exc:
            errors["deeplabv3"] = exc
            if backend != "auto":
                raise

    details = ", ".join(f"{k}: {v}" for k, v in errors.items()) or "no backends available"
    raise RuntimeError(f"Unable to initialize segmenter backend ({details})")


@dataclass
class SegmentationResult:
    mask_dir: Path
    total_frames: int
    masks_written: int
    backend: str


def segment_video(
    in_path: str,
    mask_root: str = "masks",
    backend: str = "auto",
    video_id: Optional[str] = None,
    stride: int = 1,
    smooth_kernel: int = 0,
    progress: bool = False,
) -> SegmentationResult:
    """
    Generate alpha masks for each frame and save as PNGs.
    """

    if stride < 1:
        raise ValueError("stride must be >= 1")

    video_id = video_id or Path(in_path).stem
    mask_dir = Path(mask_root) / video_id
    mask_dir.mkdir(parents=True, exist_ok=True)

    cap, fps, (width, height), total = open_video(in_path)

    segmenter = build_segmenter(backend)

    frame_idx = 0
    mask_count = 0
    kernel = (smooth_kernel // 2) * 2 + 1 if smooth_kernel and smooth_kernel > 0 else 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            alpha = segmenter.segment(frame)
            if kernel >= 3:
                alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)

            alpha = np.clip(alpha, 0.0, 1.0)
            mask_img = (alpha * 255).astype(np.uint8)
            out_path = mask_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), mask_img)
            mask_count += 1

            if progress and mask_count % 50 == 0:
                print(f"[segment] processed {mask_count} frames...", flush=True)

            frame_idx += 1
    finally:
        cap.release()

    metadata = {
        "video_path": in_path,
        "video_id": video_id,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total,
        "stride": stride,
        "backend": type(segmenter).__name__,
    }
    (mask_dir / "metadata.json").write_text(json_dumps(metadata), encoding="utf-8")

    return SegmentationResult(
        mask_dir=mask_dir,
        total_frames=total,
        masks_written=mask_count,
        backend=metadata["backend"],
    )


def json_dumps(payload: Dict) -> str:
    import json

    return json.dumps(payload, indent=2)
