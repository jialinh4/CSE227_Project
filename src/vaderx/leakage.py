"""
Leakage metrics (ELR/BER) derived from segmentation masks.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .io import open_video


@dataclass
class LeakageMetricsConfig:
    mask_threshold: float = 0.5
    ring_width: int = 8
    diff_threshold: float = 15.0
    stability_window: int = 5
    stability_threshold: float = 12.0
    ber_region: str = "ring"  # ring|background
    warmup_frames: int = 8


@dataclass
class FrameLeakage:
    frame_index: int
    timestamp_sec: float
    elr: float
    ber: float


@dataclass
class LeakageSummary:
    elr_mean: float
    elr_peak: float
    ber_mean: float
    ber_peak: float


@dataclass
class SnapshotInfo:
    frame_index: int
    timestamp_sec: float
    elr: float
    ber: float
    reason: str
    path: Path


@dataclass
class MeasurementResult:
    frames: List[FrameLeakage]
    summary: LeakageSummary
    fps: float
    video_path: str
    mask_dir: Path
    snapshots: List[SnapshotInfo]


def _load_mask_index(mask_dir: Path) -> Dict[int, Path]:
    pattern = re.compile(r"frame_(\d+)\.png$")
    index: Dict[int, Path] = {}
    for mask_path in sorted(mask_dir.glob("*.png")):
        match = pattern.match(mask_path.name)
        if not match:
            continue
        index[int(match.group(1))] = mask_path
    if not index:
        raise FileNotFoundError(f"No masks found under {mask_dir}")
    return index


def _prepare_masks(alpha: np.ndarray, cfg: LeakageMetricsConfig):
    fg_mask = alpha >= cfg.mask_threshold
    bg_mask = ~fg_mask
    if not cfg.ring_width:
        ring_mask = np.zeros_like(fg_mask, dtype=bool)
    else:
        kernel_size = max(1, cfg.ring_width * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fg_uint = fg_mask.astype(np.uint8)
        dilated = cv2.dilate(fg_uint, kernel, iterations=1)
        ring = cv2.subtract(dilated, fg_uint)
        ring_mask = ring.astype(bool) & bg_mask
    return fg_mask, bg_mask, ring_mask


def measure_leakage(
    video_path: str,
    mask_dir: str,
    cfg: LeakageMetricsConfig | None = None,
    snapshot_dir: Optional[str] = None,
    snapshot_elr_threshold: float = 0.85,
    snapshot_ber_threshold: float = 0.6,
    snapshot_max_count: int = 10,
) -> MeasurementResult:
    cfg = cfg or LeakageMetricsConfig()
    region_mode = cfg.ber_region.lower()
    if region_mode not in {"ring", "background"}:
        raise ValueError("ber_region must be 'ring' or 'background'")
    warmup_frames = max(0, int(cfg.warmup_frames))
    mask_root = Path(mask_dir)
    mask_index = _load_mask_index(mask_root)
    snapshot_root = Path(snapshot_dir) if snapshot_dir else None
    snapshots: List[SnapshotInfo] = []

    cap, fps, (width, height), total = open_video(video_path)
    frames: List[FrameLeakage] = []

    prev_gray: np.ndarray | None = None
    recent_frames: List[np.ndarray] = []

    try:
        frame_idx = 0
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
                raise FileNotFoundError(f"Cannot read mask {mask_path}")
            if mask_img.shape[:2] != frame.shape[:2]:
                mask_img = cv2.resize(
                    mask_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            alpha = mask_img.astype(np.float32) / 255.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.zeros_like(gray) if prev_gray is None else cv2.absdiff(gray, prev_gray)
            prev_gray = gray

            _, bg_mask, ring_mask = _prepare_masks(alpha, cfg)
            region_mask = ring_mask if region_mode == "ring" else bg_mask
            ring_pixels = int(np.count_nonzero(ring_mask))
            if ring_pixels == 0:
                elr = 0.0
            else:
                high_motion = diff > cfg.diff_threshold
                elr = float(np.count_nonzero(high_motion & ring_mask) / ring_pixels)

            region_pixels = int(np.count_nonzero(region_mask))
            ber = 0.0
            recent_frames.append(gray)
            if len(recent_frames) > cfg.stability_window:
                recent_frames.pop(0)
            if len(recent_frames) == cfg.stability_window and region_pixels > 0:
                stack = np.stack(recent_frames, axis=0)
                range_map = stack.max(axis=0) - stack.min(axis=0)
                unstable = range_map >= cfg.stability_threshold
                ber = float(np.count_nonzero(unstable & region_mask) / region_pixels)

            if frame_idx < warmup_frames:
                elr = 0.0
                ber = 0.0

            if snapshot_root and len(snapshots) < snapshot_max_count:
                reasons: List[str] = []
                if snapshot_elr_threshold >= 0 and elr >= snapshot_elr_threshold:
                    reasons.append("ELR")
                if snapshot_ber_threshold >= 0 and ber >= snapshot_ber_threshold:
                    reasons.append("BER")
                if reasons:
                    snapshot_root.mkdir(parents=True, exist_ok=True)
                    snap_path = snapshot_root / f"frame_{frame_idx:06d}.png"
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        f"ELR:{elr:.3f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        f"BER:{ber:.3f}",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 128, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        "+".join(reasons),
                        (12, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (40, 255, 40),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imwrite(str(snap_path), overlay)
                    snapshots.append(
                        SnapshotInfo(
                            frame_index=frame_idx,
                            timestamp_sec=frame_idx / fps if fps else 0.0,
                            elr=elr,
                            ber=ber,
                            reason="+".join(reasons),
                            path=snap_path,
                        )
                    )

            frames.append(
                FrameLeakage(
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / fps if fps else 0.0,
                    elr=elr,
                    ber=ber,
                )
            )

            frame_idx += 1
    finally:
        cap.release()

    elrs = np.array([f.elr for f in frames], dtype=np.float32) if frames else np.array([0.0])
    bers = np.array([f.ber for f in frames], dtype=np.float32) if frames else np.array([0.0])
    summary = LeakageSummary(
        elr_mean=float(elrs.mean()),
        elr_peak=float(elrs.max()),
        ber_mean=float(bers.mean()),
        ber_peak=float(bers.max()),
    )

    return MeasurementResult(
        frames=frames,
        summary=summary,
        fps=fps,
        video_path=video_path,
        mask_dir=mask_root,
        snapshots=snapshots,
    )


def write_metrics_csv(result: MeasurementResult, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["kind", "frame_index", "timestamp_sec", "elr", "ber", "elr_mean", "elr_peak", "ber_mean", "ber_peak"]
        )
        for frame in result.frames:
            writer.writerow(["frame", frame.frame_index, f"{frame.timestamp_sec:.4f}", f"{frame.elr:.6f}", f"{frame.ber:.6f}", "", "", "", ""])
        summary = result.summary
        writer.writerow(
            [
                "summary",
                "",
                "",
                f"{summary.elr_mean:.6f}",
                f"{summary.ber_mean:.6f}",
                f"{summary.elr_mean:.6f}",
                f"{summary.elr_peak:.6f}",
                f"{summary.ber_mean:.6f}",
                f"{summary.ber_peak:.6f}",
            ]
        )
        for snap in result.snapshots:
            writer.writerow(
                [
                    "snapshot",
                    snap.frame_index,
                    f"{snap.timestamp_sec:.4f}",
                    f"{snap.elr:.6f}",
                    f"{snap.ber:.6f}",
                    "",
                    "",
                    snap.reason,
                    str(snap.path),
                ]
            )


def save_metrics_plot(result: MeasurementResult, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = [f.timestamp_sec for f in result.frames]
    elrs = [f.elr for f in result.frames]
    bers = [f.ber for f in result.frames]

    plt.figure(figsize=(10, 4))
    plt.plot(times, elrs, label="ELR", color="#d62728")
    plt.plot(times, bers, label="BER", color="#1f77b4")
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio")
    plt.title("Edge Leakage / Background Exposure")
    plt.legend()
    if result.snapshots:
        snap_elr = [s.elr for s in result.snapshots if "ELR" in s.reason]
        snap_elr_t = [s.timestamp_sec for s in result.snapshots if "ELR" in s.reason]
        snap_ber = [s.ber for s in result.snapshots if "BER" in s.reason]
        snap_ber_t = [s.timestamp_sec for s in result.snapshots if "BER" in s.reason]
        if snap_elr:
            plt.scatter(snap_elr_t, snap_elr, marker="o", color="#d62728", edgecolors="k", zorder=5, label="ELR snapshot")
        if snap_ber:
            plt.scatter(snap_ber_t, snap_ber, marker="s", color="#1f77b4", edgecolors="k", zorder=5, label="BER snapshot")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
