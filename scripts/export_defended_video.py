#!/usr/bin/env python3
# Only English comments and print messages.
import os, sys, glob, cv2

def main():
    if len(sys.argv) < 3:
        print("Usage: export_defended_video.py <video_id> <variant> [fps]")
        sys.exit(2)
    video_id = sys.argv[1]
    variant = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0

    in_dir = f"outputs/frames_defended/{video_id}/{variant}"
    files = sorted(glob.glob(os.path.join(in_dir, "*.png")))
    if not files:
        print(f"[error] No frames found in {in_dir}")
        sys.exit(2)

    os.makedirs("outputs/videos", exist_ok=True)
    out_path = f"outputs/videos/{video_id}_{variant}.mp4"

    # read first frame to get size
    first = cv2.imread(files[0], cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        print("[error] Failed to open VideoWriter")
        sys.exit(1)

    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed to read {fp}, skip")
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(img)

    vw.release()
    print(f"✔ Wrote {out_path}")

if __name__ == "__main__":
    main()
