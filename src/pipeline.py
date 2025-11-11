#!/usr/bin/env python3
# coding: utf-8
"""
CLI pipeline for defense stage (single-command version, no subcommands).
Only English comments and print text.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict
import yaml
import typer

from defense.io import (
    load_alpha_sequence,
    save_alpha_sequence,
    load_video_frames_optional,
    save_frames_sequence,
)
from defense.defenses import (
    erosion_alpha_seq,
    boundary_noise_seq,
    temporal_jitter_seq,
)

def parse_size(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    import re
    m = re.match(r"^(\d+)x(\d+)$", s.strip())
    if not m:
        raise typer.BadParameter("Use WxH, e.g., 1280x720")
    return (int(m.group(1)), int(m.group(2)))

def main(
    # core ids/paths
    video_id: str = typer.Option(..., "--video-id", help="Video id used to locate inputs/outputs"),
    masks_dir: str = typer.Option("data/interim/masks", "--masks-dir", "--masks_dir", help="Dir of alpha masks per video id"),
    frames_dir: str = typer.Option("data/interim/frames", "--frames-dir", "--frames_dir", help="Optional frames dir to export defended frames"),
    # UPDATED DEFAULTS: write everything inside outputs/
    out_alpha_root: str = typer.Option("outputs/defenses", "--out-alpha-root", "--out_alpha_root", help="Output root for defended alpha"),
    out_frames_root: str = typer.Option("outputs/frames_defended", "--out-frames-root", "--out_frames_root", help="Output root for defended frames"),
    # defense & params
    defense: str = typer.Option(..., "--defense", help="erosion|boundary_noise|temporal_jitter"),
    params: Optional[str] = typer.Option(None, "--params", help="YAML string or path to YAML with params"),
    # optional render size
    resize: Optional[str] = typer.Option(None, "--resize", help="Optional resize WxH for saving frames"),
):
    # parse params
    cfg: Dict = {}
    if params:
        if os.path.isfile(params):
            with open(params, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = yaml.safe_load(params) or {}

    # load alphas
    alpha_in_dir = os.path.join(masks_dir, video_id)
    alphas = load_alpha_sequence(alpha_in_dir)
    if not alphas:
        typer.secho(f"[warn] No masks found under {alpha_in_dir}", fg=typer.colors.YELLOW)
        raise SystemExit(2)

    # apply defense
    dname = defense.lower().strip()
    if dname == "erosion":
        radius = int(cfg.get("radius", 9))
        alphas_def = erosion_alpha_seq(alphas, radius=radius)
        variant = f"erosion_r{radius}"
    elif dname == "boundary_noise":
        sigma = float(cfg.get("sigma", 0.04))
        band = int(cfg.get("band_width", 5))
        alphas_def = boundary_noise_seq(alphas, sigma=sigma, band_width=band)
        variant = f"bnoise_s{sigma}_b{band}"
    elif dname == "temporal_jitter":
        shift_px = int(cfg.get("shift_px", 2))
        prob = float(cfg.get("prob", 0.5))
        alphas_def = temporal_jitter_seq(alphas, shift_px=shift_px, prob=prob)
        variant = f"tjit_s{shift_px}_p{prob}"
    else:
        raise typer.BadParameter("Unknown --defense")

    # save defended alphas
    out_alpha_dir = os.path.join(out_alpha_root, video_id, variant)
    save_alpha_sequence(alphas_def, out_alpha_dir)

    # optional defended frames
    frames_in_dir = os.path.join(frames_dir, video_id)
    frames = load_video_frames_optional(frames_in_dir)
    if frames:
        size = parse_size(resize)
        save_frames_sequence(frames, alphas_def, os.path.join(out_frames_root, video_id, variant), resize_to=size)

    typer.echo(f"✔ Defended masks saved to {out_alpha_dir}")
    if frames:
        typer.echo(f"✔ Defended frames saved to {os.path.join(out_frames_root, video_id, variant)}")

if __name__ == "__main__":
    typer.run(main)
