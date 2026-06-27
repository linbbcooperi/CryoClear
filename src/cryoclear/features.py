"""Per-candidate image features for the junk classifier.

Given a micrograph (2D array) and candidate particle centres, crop a box around
each centre and compute cheap, interpretable features that separate real
particles from junk (ice = bright/high-contrast blobs, carbon edges = strong
straight gradients, aggregates = large high-variance regions).

Start here with a RandomForest (see junk_classifier.py); upgrade to a CNN on the
raw crops only once M1–M3 are solid.

Depends only on numpy → unit-testable offline.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

FEATURE_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "contrast",        # (max-min)/(max+min+eps)
    "grad_mean",       # mean gradient magnitude (edge density)
    "grad_std",
    "central_ratio",   # center brightness vs border (blobiness)
)


def _crop(image: np.ndarray, x: float, y: float, box: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = box // 2
    x0, x1 = int(round(x)) - half, int(round(x)) + half
    y0, y1 = int(round(y)) - half, int(round(y)) + half
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    return image[y0c:y1c, x0c:x1c]


def crop_features(crop: np.ndarray) -> np.ndarray:
    """Compute the FEATURE_NAMES vector for a single crop."""
    eps = 1e-6
    if crop.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    c = crop.astype(np.float32)
    mean = float(c.mean())
    std = float(c.std())
    cmin = float(c.min())
    cmax = float(c.max())
    contrast = (cmax - cmin) / (cmax + cmin + eps)
    gy, gx = np.gradient(c)
    gmag = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(gmag.mean())
    grad_std = float(gmag.std())
    # central brightness vs border → "blobiness" of a real particle
    hh, ww = c.shape
    qy, qx = max(hh // 4, 1), max(ww // 4, 1)
    center = c[qy: hh - qy, qx: ww - qx]
    center_mean = float(center.mean()) if center.size else mean
    border_sum = float(c.sum()) - float(center.sum())
    border_n = max(c.size - center.size, 1)
    border_mean = border_sum / border_n
    central_ratio = (center_mean + eps) / (border_mean + eps)
    return np.array(
        [mean, std, cmin, cmax, contrast, grad_mean, grad_std, central_ratio],
        dtype=float,
    )


def extract_features(image: np.ndarray, coords: Sequence, box: int = 64) -> np.ndarray:
    """Return an (N, n_features) array for N candidate centres `coords` (x, y)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    feats = np.zeros((len(coords), len(FEATURE_NAMES)), dtype=float)
    for i, (x, y) in enumerate(coords):
        feats[i] = crop_features(_crop(image, x, y, box))
    return feats
