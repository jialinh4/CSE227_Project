from typing import Tuple
import cv2

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, (w, h), total

def make_writer(out_path: str, fps: float, size: Tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, float(fps), size, isColor=True)
