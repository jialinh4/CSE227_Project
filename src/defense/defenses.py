# coding: utf-8
"""
Core defenses for alpha masks.
All functions use float32 alpha in [0,1].
Only English comments and print messages.
"""
from __future__ import annotations
from typing import List
import numpy as np
import cv2

def _ensure_float01(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    # if looks like 0..255 convert
    if a.max() > 1.0:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)

def erosion_alpha(a: np.ndarray, radius: int = 9) -> np.ndarray:
    a = _ensure_float01(a)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(a, kernel, iterations=1)
    return np.clip(eroded, 0.0, 1.0)

def erosion_alpha_seq(seq: List[np.ndarray], radius: int = 9) -> List[np.ndarray]:
    return [erosion_alpha(a, radius=radius) for a in seq]

def boundary_noise(a: np.ndarray, sigma: float = 0.04, band_width: int = 5) -> np.ndarray:
    a = _ensure_float01(a)
    # build boundary band via dilate - erode
    k = 2 * band_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(a, kernel, iterations=1)
    ero = cv2.erode(a, kernel, iterations=1)
    band = (dil - ero) > 1e-6
    noise = np.random.normal(0.0, sigma, size=a.shape).astype(np.float32)
    out = a.copy()
    out[band] = np.clip(out[band] + noise[band], 0.0, 1.0)
    return out

def boundary_noise_seq(seq: List[np.ndarray], sigma: float = 0.04, band_width: int = 5) -> List[np.ndarray]:
    return [boundary_noise(a, sigma=sigma, band_width=band_width) for a in seq]

def temporal_jitter_seq(seq: List[np.ndarray], shift_px: int = 2, prob: float = 0.5) -> List[np.ndarray]:
    out = []
    h, w = seq[0].shape[:2]
    for i, a in enumerate(seq):
        a = _ensure_float01(a)
        if np.random.rand() < prob:
            dx = np.random.randint(-shift_px, shift_px + 1)
            dy = np.random.randint(-shift_px, shift_px + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            out.append(moved)
        else:
            out.append(a)
    return out
