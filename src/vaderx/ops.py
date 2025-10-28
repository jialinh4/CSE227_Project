import cv2
import numpy as np
from typing import Tuple

def to_gray_bgr(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def resize(frame: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    w, h = size
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
