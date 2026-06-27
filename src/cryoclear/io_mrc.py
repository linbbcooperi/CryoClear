"""MRC micrograph I/O and preprocessing (Tony).

Reads .mrc files, normalizes to a viewable 8-bit image, optional downsample.
`mrcfile` is imported lazily so `import cryoclear` works without it installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mrc(path: str | Path) -> np.ndarray:
    """Load a micrograph as a float32 2D array."""
    import mrcfile

    with mrcfile.open(str(path), permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    return data


def normalize_8bit(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile-clip then scale to 0-255 uint8 for display/feature extraction."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def downsample(img: np.ndarray, factor: int = 4) -> np.ndarray:
    """Fast box downsample by an integer factor (keeps things real-time)."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h - (h % factor), w - (w % factor)
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def load_for_pipeline(path: str | Path, factor: int = 4) -> np.ndarray:
    """Convenience: load -> downsample -> 8-bit, ready for picker + features."""
    return normalize_8bit(downsample(load_mrc(path), factor))
