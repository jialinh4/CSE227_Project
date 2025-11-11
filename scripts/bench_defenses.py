#!/usr/bin/env python3
# Only English comments and print messages.
import os, sys, csv, yaml, subprocess, shlex, argparse
from itertools import product

sys.path.insert(0, "src")
from config import get_dataset  # dataset registry

GRID_PATH = "configs/defenses_grid.yml"
MEASURE_SCRIPT = "scripts/measure_leakage.py"  # optional

def run_cmd(cmd: str):
    print(f"[run] {cmd}")
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(f"[warn] command failed: {cmd}", file=sys.stderr)
    return p.returncode == 0

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name in configs/datasets.yml")
    ap.add_argument("--seed", default="1234")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    video_id = ds.video_id
    masks_dir = ds.get("masks_dir", "data/interim/masks")
    frames_dir = ds.get("frames_dir", "data/interim/frames")
    out_root = ds.get("outputs_root", "outputs")
    out_alpha_root = os.path.join(out_root, "defenses")
    out_frames_root = os.path.join(out_root, "frames_defended")

    with open(GRID_PATH, "r") as f:
        grid = yaml.safe_load(f)

    results_csv = os.path.join(out_root, f"bench_{video_id}.csv")
    ensure_dir(out_root)
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense","params","variant","out_alpha_dir","out_frames_dir","seed"])

        # erosion
        for r in grid.get("erosion",{}).get("radius",[]):
            variant = f"erosion_r{r}"
            cmd = (f"python -m pipeline --dataset {args.dataset} "
                   f"--defense erosion --params 'radius: {r}' --seed {args.seed}")
            if run_cmd(cmd):
                w.writerow(["erosion", f"radius={r}",
                            variant,
                            f"{out_alpha_root}/{video_id}/{variant}",
                            f"{out_frames_root}/{video_id}/{variant}",
                            args.seed])

        # temporal_jitter
        tj = grid.get("temporal_jitter",{})
        for s,p in product(tj.get("shift_px",[]), tj.get("prob",[])):
            variant = f"tjit_s{s}_p{p}"
            cmd = (f"python -m pipeline --dataset {args.dataset} "
                   f"--defense temporal_jitter --params 'shift_px: {s}\nprob: {p}' --seed {args.seed}")
            if run_cmd(cmd):
                w.writerow(["temporal_jitter", f"shift_px={s},prob={p}",
                            variant,
                            f"{out_alpha_root}/{video_id}/{variant}",
                            f"{out_frames_root}/{video_id}/{variant}",
                            args.seed])

        # boundary_noise
        bn = grid.get("boundary_noise",{})
        for sigma,band in product(bn.get("sigma",[]), bn.get("band_width",[])):
            variant = f"bnoise_s{sigma}_b{band}"
            cmd = (f"python -m pipeline --dataset {args.dataset} "
                   f"--defense boundary_noise --params 'sigma: {sigma}\nband_width: {band}' --seed {args.seed}")
            if run_cmd(cmd):
                w.writerow(["boundary_noise", f"sigma={sigma},band_width={band}",
                            variant,
                            f"{out_alpha_root}/{video_id}/{variant}",
                            f"{out_frames_root}/{video_id}/{variant}",
                            args.seed])

    # Optional measure script
    if os.path.isfile(MEASURE_SCRIPT):
        print("[info] Running leakage measurement...")
        cmd = f"python {MEASURE_SCRIPT} --video-id {video_id} --inputs {out_frames_root}/{video_id}"
        run_cmd(cmd)

    print(f"[done] Wrote grid summary to {results_csv}")

if __name__ == "__main__":
    main()
