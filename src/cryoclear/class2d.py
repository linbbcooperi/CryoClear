"""Reference-free 2D classification of kept particles (M4) — cryoSPARC-style.

Produces class-average images ("2D classes"), the cryo-EM check that the kept
particles are real protein views. Pure numpy + scipy (no RELION/cryoSPARC).

What makes the averages crisp (the cryoSPARC-like ingredients we added):
  1. **Translational + rotational** alignment. The old version aligned rotation only,
     so the picker's ±several-px centring error blurred every average — the #1 fix.
     Translation uses **phase correlation** (sharp peak) and is applied with a
     **Fourier shift** (no spline-interpolation blur).
  2. **Resolution annealing** (frequency marching): align coarse first (heavy
     low-pass) and tighten each iteration — the biggest crispness trick after (1).
  3. **Band-pass** alignment images (high-pass removes the ice/carbon ramp, low-pass
     removes noise) kept separate from the lightly-filtered average accumulator.
  4. **Soft cosine mask** (a hard binary mask rings the FFT).
  5. **Rotation-invariant PCA + k-means++** initialisation (no random seeds → classes
     stop being blends of views).
  6. **Outlier rejection** (drop the lowest-scoring members before averaging) and
     dropping tiny classes; uses many particles (averages sharpen as ~1/sqrt(N)).
"""
from __future__ import annotations

import numpy as np


def _soft_mask(s: int, radius: float | None = None, edge: float = 5.0) -> np.ndarray:
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2
    rr = np.hypot(xx - c, yy - c)
    radius = radius if radius is not None else s * 0.46
    return np.clip((radius - rr) / edge + 0.5, 0.0, 1.0).astype(np.float32)


def extract_particles(image: np.ndarray, coords, box: int, out_size: int = 96) -> np.ndarray:
    """Crop `box`-px particles at `coords` (full-res x,y), resize to out_size, normalise."""
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
    return np.stack(crops)


def _lowpass(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma)


def _bandpass(img: np.ndarray, lp: float, hp: float = 8.0) -> np.ndarray:
    """High-pass (remove ramp) + low-pass (denoise) — the alignment image."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, lp) - gaussian_filter(img, hp)


def _to_polar(img: np.ndarray, n_r: int = 36, n_theta: int = 72) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    s = img.shape[0]
    c = (s - 1) / 2
    r = np.linspace(1, s / 2 - 1, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, TH = np.meshgrid(r, th, indexing="ij")
    return map_coordinates(img, [c + R * np.sin(TH), c + R * np.cos(TH)], order=1)


def _best_angle_bin(img_polar_f: np.ndarray, ref_polar: np.ndarray) -> tuple[int, float]:
    """Best rotation bin + score. `img_polar_f` is the pre-computed FFT of the polar image."""
    cc = np.fft.ifft(img_polar_f * np.conj(np.fft.fft(ref_polar, axis=1)), axis=1).real.sum(axis=0)
    k = int(np.argmax(cc))
    return k, float(cc[k])


def _phase_shift(img: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Translation aligning img→ref by phase correlation (sharp peak) + peak score."""
    F, G = np.fft.fft2(img), np.fft.fft2(ref)
    R = F * np.conj(G)
    R /= np.abs(R) + 1e-8                       # phase correlation
    cc = np.fft.fftshift(np.fft.ifft2(R).real)
    h, w = img.shape
    py, px = np.unravel_index(int(np.argmax(cc)), cc.shape)
    return float(py - h // 2), float(px - w // 2), float(cc.max())


def _apply(img: np.ndarray, ang: float, shift) -> np.ndarray:
    from scipy.ndimage import rotate, fourier_shift
    out = rotate(img, ang, reshape=False, order=1, mode="nearest") if ang else img
    if shift[0] or shift[1]:
        out = np.fft.ifftn(fourier_shift(np.fft.fftn(out), shift)).real
    return out


def _rot_invariant_features(crops_bp, n_theta=72):
    """Rotation-invariant descriptor: |FFT over theta| of each polar image."""
    feats = []
    for c in crops_bp:
        pol = _to_polar(c, n_theta=n_theta)
        feats.append(np.abs(np.fft.fft(pol, axis=1))[:, :n_theta // 4].ravel())
    return np.asarray(feats, np.float32)


def _kmeanspp(feat, k, rng):
    n = len(feat)
    idx = [int(rng.integers(n))]
    d2 = ((feat - feat[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        nxt = int(rng.choice(n, p=d2 / (d2.sum() + 1e-12)))
        idx.append(nxt)
        d2 = np.minimum(d2, ((feat - feat[nxt]) ** 2).sum(1))
    return idx


def classify_2d(crops: np.ndarray, n_classes: int = 8, n_iter: int = 6,
                n_theta: int = 72, seed: int = 0):
    """Reference-free 2D classification (translation+rotation E-M, resolution-annealed)."""
    n = len(crops)
    if n == 0:
        return np.zeros((0,) + crops.shape[1:]), np.zeros(0, int), np.zeros(0, int)
    k = min(n_classes, n)
    s = crops.shape[1]
    mask = _soft_mask(s)
    crops = crops * mask

    # rotation-invariant PCA + k-means++ init (no random seeds)
    init_bp = np.stack([_bandpass(c, 2.0) for c in crops]) * mask
    feat = _rot_invariant_features(init_bp, n_theta)
    feat = feat - feat.mean(0)
    _, _, vt = np.linalg.svd(feat, full_matrices=False)
    pca = feat @ vt[:min(16, vt.shape[0])].T
    idx = _kmeanspp(pca, k, np.random.default_rng(seed))
    refs = crops[idx].copy()

    labels = np.zeros(n, int)
    counts = np.zeros(k, int)
    for it in range(n_iter):
        lp = max(3.0 - 2.0 * it / max(n_iter - 1, 1), 1.0)        # anneal: coarse→fine
        crops_bp = np.stack([_bandpass(c, lp) for c in crops]) * mask
        polar_f = [np.fft.fft(_to_polar(c, n_theta=n_theta), axis=1) for c in crops_bp]
        refs_bp = np.stack([_bandpass(r, lp) for r in refs]) * mask
        ref_polars = [_to_polar(r, n_theta=n_theta) for r in refs_bp]

        # E-step: assign class by cheap rotational cross-correlation (no Cartesian rotate)
        bk = np.zeros(n, int); bang = np.zeros(n); bscore = np.zeros(n)
        for i in range(n):
            best_j, best_s, best_b = 0, -1e30, 0
            for j in range(k):
                b, sc = _best_angle_bin(polar_f[i], ref_polars[j])
                if sc > best_s:
                    best_s, best_j, best_b = sc, j, b
            bk[i] = best_j
            bang[i] = -best_b * 360.0 / n_theta
            bscore[i] = best_s
        labels = bk

        # M-step: rotate + translate (phase correlation) each member ONCE, then average
        new = np.zeros_like(refs); counts = np.zeros(k, int)
        for j in range(k):
            members = np.where(bk == j)[0]
            if len(members) == 0:
                continue
            if len(members) >= 8:        # outlier rejection: drop lowest-scoring 15%
                members = members[bscore[members] >= np.quantile(bscore[members], 0.15)]
            for i in members:
                rbp = _apply(crops_bp[i], bang[i], (0, 0))           # rotated band-pass → translation
                dy, dx, _ = _phase_shift(rbp, refs_bp[j])
                new[j] += _apply(crops[i], bang[i], (dy, dx))        # rotate+shift the original
            counts[j] = len(members)
            refs[j] = new[j] / max(counts[j], 1)

    from scipy.ndimage import gaussian_filter
    refs = np.stack([gaussian_filter(r, 0.6) * mask for r in refs])
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
        a = (avg - avg.min()) / (np.ptp(avg) + 1e-6)
        out[pad + r * (s + pad):pad + r * (s + pad) + s,
            pad + c * (s + pad):pad + c * (s + pad) + s] = a
    return out
