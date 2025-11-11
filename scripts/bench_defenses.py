#!/usr/bin/env python3
# Only English comments and print messages.
import os, sys, csv, yaml, subprocess, shlex
from itertools import product

GRID_PATH = "configs/defenses_grid.yml"
VIDEO_ID = os.environ.get("VIDEO_ID", "test1")
MASKS_DIR = os.environ.get("MASKS_DIR", "data/interim/masks")
FRAMES_DIR = os.environ.get("FRAMES_DIR", "data/interim/frames")
OUT_ALPHA_ROOT = os.environ.get("OUT_ALPHA_ROOT", "outputs/defenses")
OUT_FRAMES_ROOT = os.environ.get("OUT_FRAMES_ROOT", "outputs/frames_defended")
SEED = os.environ.get("SEED", "1234")
RESULTS_CSV = os.environ.get("RESULTS_CSV", f"outputs/bench_{VIDEO_ID}.csv")

MEASURE_SCRIPT = "scripts/measure_leakage.py"  # optional; adapt CLI if needed

def run_cmd(cmd: str):
    print(f"[run] {cmd}")
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(f"[warn] command failed: {cmd}", file=sys.stderr)
    return p.returncode == 0

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def write_row(writer, defense, params_str, variant):
    out_alpha_dir = f"{OUT_ALPHA_ROOT}/{VIDEO_ID}/{variant}"
    out_frames_dir = f"{OUT_FRAMES_ROOT}/{VIDEO_ID}/{variant}"
    writer.writerow([defense, params_str, variant, out_alpha_dir, out_frames_dir, SEED])

def main():
    with open(GRID_PATH, "r") as f:
        grid = yaml.safe_load(f)

    ensure_dir(os.path.dirname(RESULTS_CSV) or ".")
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense", "params", "variant", "out_alpha_dir", "out_frames_dir", "seed"])

        # Erosion
        for r in grid.get("erosion", {}).get("radius", []):
            params = f"radius: {r}"
            variant = f"erosion_r{r}"
            cmd = (
                f"python -m pipeline --video-id {VIDEO_ID} "
                f"--masks-dir {MASKS_DIR} --frames-dir {FRAMES_DIR} "
                f"--defense erosion --params '{params}' "
                f"--out-alpha-root {OUT_ALPHA_ROOT} --out-frames-root {OUT_FRAMES_ROOT} "
                f"--seed {SEED}"
            )
            if run_cmd(cmd):
                write_row(w, "erosion", f"radius={r}", variant)

        # Temporal jitter
        tj = grid.get("temporal_jitter", {})
        for s, p in product(tj.get("shift_px", []), tj.get("prob", [])):
            params = f"shift_px: {s}\nprob: {p}"
            variant = f"tjit_s{s}_p{p}"
            cmd = (
                f"python -m pipeline --video-id {VIDEO_ID} "
                f"--masks-dir {MASKS_DIR} --frames-dir {FRAMES_DIR} "
                f"--defense temporal_jitter --params '{params}' "
                f"--out-alpha-root {OUT_ALPHA_ROOT} --out-frames-root {OUT_FRAMES_ROOT} "
                f"--seed {SEED}"
            )
            if run_cmd(cmd):
                write_row(w, "temporal_jitter", f"shift_px={s},prob={p}", variant)

        # Boundary noise
        bn = grid.get("boundary_noise", {})
        for sigma, band in product(bn.get("sigma", []), bn.get("band_width", [])):
            params = f"sigma: {sigma}\nband_width: {band}"
            variant = f"bnoise_s{sigma}_b{band}"
            cmd = (
                f"python -m pipeline --video-id {VIDEO_ID} "
                f"--masks-dir {MASKS_DIR} --frames-dir {FRAMES_DIR} "
                f"--defense boundary_noise --params '{params}' "
                f"--out-alpha-root {OUT_ALPHA_ROOT} --out-frames-root {OUT_FRAMES_ROOT} "
                f"--seed {SEED}"
            )
            if run_cmd(cmd):
                write_row(w, "boundary_noise", f"sigma={sigma},band_width={band}", variant)

    # Optional: measurement step (adapt to your script)
    if os.path.isfile(MEASURE_SCRIPT):
        print("[info] Running leakage measurement...")
        cmd = f"python {MEASURE_SCRIPT} --video-id {VIDEO_ID} --inputs {OUT_FRAMES_ROOT}/{VIDEO_ID}"
        run_cmd(cmd)

    print(f"[done] Wrote grid summary to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
