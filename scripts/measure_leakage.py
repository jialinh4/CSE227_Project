#!/usr/bin/env python
"""
Compute baseline edge-leakage metrics (ELR/BER) for a segmented video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vaderx.leakage import (  # noqa: E402
    LeakageMetricsConfig,
    measure_leakage,
    save_metrics_plot,
    write_metrics_csv,
)

DEFAULT_CFG = LeakageMetricsConfig()
DEFAULT_SNAPSHOT_ELR = 0.85
DEFAULT_SNAPSHOT_BER = 0.6
DEFAULT_SNAPSHOT_MAX = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure ELR/BER leakage baselines.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing alpha masks.")
    parser.add_argument("--video_id", help="Optional identifier (defaults to mask_dir name).")
    parser.add_argument("--results_dir", default="results", help="Output directory for CSV/plots.")
    parser.add_argument(
        "--ring_width",
        type=int,
        default=DEFAULT_CFG.ring_width,
        help=f"Boundary ring width in pixels (default: {DEFAULT_CFG.ring_width}).",
    )
    parser.add_argument(
        "--diff_threshold",
        type=float,
        default=DEFAULT_CFG.diff_threshold,
        help=f"Frame-diff threshold for ELR (default: {DEFAULT_CFG.diff_threshold}).",
    )
    parser.add_argument(
        "--stability_window",
        type=int,
        default=DEFAULT_CFG.stability_window,
        help=f"Window size (frames) for BER stability check (default: {DEFAULT_CFG.stability_window}).",
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=DEFAULT_CFG.stability_threshold,
        help=f"Max intensity range to treat as unstable (default: {DEFAULT_CFG.stability_threshold}).",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=DEFAULT_CFG.mask_threshold,
        help=f"Alpha threshold to split foreground/background (default: {DEFAULT_CFG.mask_threshold}).",
    )
    parser.add_argument(
        "--ber_region",
        choices=["ring", "background"],
        default=DEFAULT_CFG.ber_region,
        help="Region considered for BER (ring keeps focus near subject edges).",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=DEFAULT_CFG.warmup_frames,
        help=f"Ignore the first N frames when summarizing ELR/BER (default: {DEFAULT_CFG.warmup_frames}).",
    )
    parser.add_argument(
        "--snapshot_dir",
        help="Directory root for saving flagged frames (per-video subfolders are auto-created).",
    )
    parser.add_argument(
        "--snapshot_elr_threshold",
        type=float,
        default=DEFAULT_SNAPSHOT_ELR,
        help=f"Save a snapshot when ELR exceeds this value (default: {DEFAULT_SNAPSHOT_ELR}; set <0 to disable).",
    )
    parser.add_argument(
        "--snapshot_ber_threshold",
        type=float,
        default=DEFAULT_SNAPSHOT_BER,
        help=f"Save a snapshot when BER exceeds this value (default: {DEFAULT_SNAPSHOT_BER}; set <0 to disable).",
    )
    parser.add_argument(
        "--snapshot_max_count",
        type=int,
        default=DEFAULT_SNAPSHOT_MAX,
        help=f"Maximum number of snapshots to save (default: {DEFAULT_SNAPSHOT_MAX}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LeakageMetricsConfig(
        mask_threshold=args.mask_threshold,
        ring_width=args.ring_width,
        diff_threshold=args.diff_threshold,
        stability_window=args.stability_window,
        stability_threshold=args.stability_threshold,
        ber_region=args.ber_region,
        warmup_frames=args.warmup_frames,
    )

    video_id = args.video_id or Path(args.mask_dir).name
    snap_dir = None
    if args.snapshot_dir:
        snap_dir = str(Path(args.snapshot_dir) / video_id)
    result = measure_leakage(
        args.video,
        args.mask_dir,
        cfg,
        snapshot_dir=snap_dir,
        snapshot_elr_threshold=args.snapshot_elr_threshold,
        snapshot_ber_threshold=args.snapshot_ber_threshold,
        snapshot_max_count=args.snapshot_max_count,
    )

    results_dir = Path(args.results_dir)
    csv_path = results_dir / f"baseline_{video_id}.csv"
    plot_path = results_dir / f"baseline_{video_id}.png"

    write_metrics_csv(result, csv_path)
    save_metrics_plot(result, plot_path)

    summary = result.summary
    print(
        "✔ Leakage metrics\n"
        f"  Video: {args.video}\n"
        f"  Masks: {args.mask_dir}\n"
        f"  Frames measured: {len(result.frames)}\n"
        f"  ELR mean/peak: {summary.elr_mean:.4f} / {summary.elr_peak:.4f}\n"
        f"  BER mean/peak: {summary.ber_mean:.4f} / {summary.ber_peak:.4f}\n"
        f"  CSV: {csv_path}\n"
        f"  Plot: {plot_path}"
    )
    if result.snapshots:
        snap_dir_print = result.snapshots[0].path.parent
        print(f"  Snapshots: {len(result.snapshots)} saved under {snap_dir_print}")


if __name__ == "__main__":
    main()
