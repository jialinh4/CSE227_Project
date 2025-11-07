from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from .io import open_video, make_writer
from .ops import to_gray_bgr, resize

@dataclass
class EditConfig:
    start_sec: float = 0.0
    end_sec: float = -1.0        # -1 means till the end
    resize_to: Optional[Tuple[int,int]] = None
    gray: bool = False

def process_video(in_path: str, out_path: str, cfg: EditConfig):
    cap, fps, (w, h), total = open_video(in_path)

    # Compute frame range
    start_f = int(max(0.0, cfg.start_sec) * fps)
    end_f   = total - 1 if cfg.end_sec < 0 else int(cfg.end_sec * fps)
    end_f   = min(end_f, total - 1)
    if end_f < start_f:
        raise ValueError(f"Invalid time range: start={cfg.start_sec}, end={cfg.end_sec}")

    # Decide output size
    out_size = cfg.resize_to if cfg.resize_to else (w, h)
    writer = make_writer(out_path, fps, out_size)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    fidx = start_f
    try:
        while fidx <= end_f:
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.gray:
                frame = to_gray_bgr(frame)
            if cfg.resize_to:
                frame = resize(frame, cfg.resize_to)

            writer.write(frame)
            fidx += 1
    finally:
        writer.release()
        cap.release()
