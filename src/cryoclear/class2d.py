"""Lightweight reference-free 2D classification of kept particles (M4).

Produces class-average images ("2D classes") — the cryo-EM sanity check that the
kept particles are real protein views. Pure numpy + scipy (no RELION/cryoSPARC).
ASPIRE is the heavier alternative; this is the always-works fallback.

Pipeline: extract crops -> normalise + circular mask -> multi-reference alignment
(rotational alignment by angular cross-correlation in polar coords) -> class means.
"""
from __future__ import annotations

import numpy as np


def extract_particles(image: np.ndarray, coords, box: int, out_size: int = 64) -> np.ndarray:
    """Crop `box`-px particles at `coords` (full-res x,y), resize to out_size, normalise+mask."""
    from scipy.ndimage import zoom

    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    half = box // 2
    h, w = image.shape[:2]
    crops = []
    for x, y in coords:
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        c = image[yi - half:yi + half, xi - half:xi + half].astype(np.float32)
        if c.shape[0] != box or c.shape[1] != box:
            continue
        c = zoom(c, out_size / box, order=1)
        c = (c - c.mean()) / (c.std() + 1e-6)
        crops.append(c)
    if not crops:
        return np.zeros((0, out_size, out_size), dtype=np.float32)
    stack = np.stack(crops)
    return stack * _circular_mask(out_size)


def _circular_mask(s: int) -> np.ndarray:
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2
    return ((xx - c) ** 2 + (yy - c) ** 2 <= (s / 2) ** 2).astype(np.float32)


def _to_polar(img: np.ndarray, n_r: int = 32, n_theta: int = 64) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    s = img.shape[0]
    c = (s - 1) / 2
    r = np.linspace(1, s / 2 - 1, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, TH = np.meshgrid(r, th, indexing="ij")
    return map_coordinates(img, [c + R * np.sin(TH), c + R * np.cos(TH)], order=1)


def _best_angle_bin(img_polar: np.ndarray, ref_polar: np.ndarray) -> tuple[int, float]:
    """Best angular shift (rotation) of img to match ref, by FFT cross-correlation."""
    f = np.fft.fft(img_polar, axis=1)
    g = np.fft.fft(ref_polar, axis=1)
    cc = np.fft.ifft(f * np.conj(g), axis=1).real.sum(axis=0)
    k = int(np.argmax(cc))
    return k, float(cc[k])


def classify_2d(crops: np.ndarray, n_classes: int = 8, n_iter: int = 5,
                n_theta: int = 64, seed: int = 0):
    """Reference-free 2D classification -> (class_averages, labels, counts)."""
    from scipy.ndimage import rotate

    n = len(crops)
    if n == 0:
        return np.zeros((0,) + crops.shape[1:]), np.zeros(0, int), np.zeros(0, int)
    k = min(n_classes, n)
    s = crops.shape[1]
    polars = np.stack([_to_polar(c, n_theta=n_theta) for c in crops])

    rng = np.random.default_rng(seed)
    refs = crops[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, int)
    for _ in range(n_iter):
        ref_polars = np.stack([_to_polar(r, n_theta=n_theta) for r in refs])
        best_k = np.zeros(n, int)
        best_bin = np.zeros(n, int)
        for i in range(n):
            scores, bins = [], []
            for j in range(k):
                b, sc = _best_angle_bin(polars[i], ref_polars[j])
                scores.append(sc)
                bins.append(b)
            j = int(np.argmax(scores))
            best_k[i] = j
            best_bin[i] = bins[j]
        labels = best_k
        new_refs = np.zeros_like(refs)
        counts = np.zeros(k, int)
        for i in range(n):
            ang = -best_bin[i] * 360.0 / n_theta
            new_refs[best_k[i]] += rotate(crops[i], ang, reshape=False, order=1)
            counts[best_k[i]] += 1
        for j in range(k):
            if counts[j] > 0:
                refs[j] = new_refs[j] / counts[j]
    order = np.argsort(-counts)
    return refs[order], labels, counts[order]


def montage(class_avgs: np.ndarray, counts=None, cols: int = 4):
    """Arrange class averages into a single 2D image for display."""
    if len(class_avgs) == 0:
        return np.zeros((1, 1))
    s = class_avgs.shape[1]
    rows = int(np.ceil(len(class_avgs) / cols))
    pad = 2
    out = np.zeros((rows * (s + pad) + pad, cols * (s + pad) + pad), dtype=np.float32)
    for idx, avg in enumerate(class_avgs):
        r, c = divmod(idx, cols)
        a = (avg - avg.min()) / (avg.ptp() + 1e-6)
        out[pad + r * (s + pad):pad + r * (s + pad) + s,
            pad + c * (s + pad):pad + c * (s + pad) + s] = a
    return out
