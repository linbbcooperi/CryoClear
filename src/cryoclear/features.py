"""Per-candidate image features for the junk classifier.

Given a micrograph and candidate centres, crop a fixed box around each centre and
compute interpretable, *intensity-normalised* features that separate real particles
from the three junk classes:

  - ice       → bright, high-frequency, often asymmetric blobs
  - carbon    → strong, coherent (single-direction) edges
  - aggregate → large, heavy-tailed, high-variance regions

plus a matched-filter score for "looks like a centred, radially-symmetric particle".

Most features are normalised by the per-crop mean/std so they are comparable *across*
micrographs (raw intensity drifts between exposures — that drift is what made the
old raw mean/min/max features memorise individual micrographs instead of generalise).

Pure numpy → unit-testable offline, and parallelises across CPU cores in precompute.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

FEATURE_NAMES = (
    # --- coarse intensity / gradient stats (kept for interpretability) ---
    "mean", "std", "min", "max",
    "contrast",         # (max-min)/(max+min)
    "grad_mean", "grad_std",
    "central_ratio",    # centre brightness vs border (signed blobiness)
    # --- radial profile: a particle is a symmetric blob decaying to background ---
    "ring0", "ring1", "ring2", "ring3",   # mean intensity in 4 concentric annuli (z-scored)
    "core_outer",       # |core - outer| contrast magnitude
    "radial_mono",      # monotonic decay core->outer (|corr| of ring index vs ring mean)
    "radial_sym",       # angular symmetry of the core (low = symmetric particle)
    # --- matched filter & local SNR ---
    "gauss_ncc",        # |NCC| with a centred Gaussian template (matched filter)
    "center_snr",       # |core - background| / background std
    "peak_prom",        # central peak prominence
    # --- edge structure (carbon edges) ---
    "grad_coherence",   # structure-tensor anisotropy (high = one dominant edge direction)
    "grad_center",      # core vs border gradient-magnitude ratio
    # --- distribution shape (aggregates / ice) & sharpness ---
    "skew", "kurtosis",
    "lap_var",          # variance-of-Laplacian (high-frequency / sharpness)
)

_GEOM: dict = {}


def _geom(shape: tuple[int, int]) -> dict:
    """Cached geometry (radius rings, angular sectors, Gaussian template) per crop shape."""
    g = _GEOM.get(shape)
    if g is not None:
        return g
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = max(min(h, w) / 2.0, 1.0)
    rn = r / rmax
    rings = [rn < 0.25, (rn >= 0.25) & (rn < 0.5),
             (rn >= 0.5) & (rn < 0.75), (rn >= 0.75) & (rn < 1.0)]
    core = rn < 0.3
    outer = (rn >= 0.7) & (rn < 1.0)
    ang = (((np.arctan2(yy - cy, xx - cx) + np.pi) * (8.0 / (2 * np.pi))).astype(int)) % 8
    sigma = max(rmax / 2.0, 1.0)
    tmpl = np.exp(-(r ** 2) / (2 * sigma ** 2)).astype(np.float32)
    tmpl = tmpl - tmpl.mean()
    g = {"rings": rings, "core": core, "outer": outer, "ang": ang,
         "tmpl": tmpl, "tmpl_norm": float(np.sqrt((tmpl ** 2).sum())) + 1e-6}
    _GEOM[shape] = g
    return g


def crop_features(crop: np.ndarray) -> np.ndarray:
    """Compute the FEATURE_NAMES vector for a single crop (any size)."""
    n = len(FEATURE_NAMES)
    if crop.size == 0:
        return np.zeros(n, dtype=float)
    eps = 1e-6
    c = crop.astype(np.float32)
    mean = float(c.mean())
    std = float(c.std()) + eps
    cmin, cmax = float(c.min()), float(c.max())
    contrast = (cmax - cmin) / (cmax + cmin + eps)
    gy, gx = np.gradient(c)
    gmag = np.sqrt(gx * gx + gy * gy)
    grad_mean, grad_std = float(gmag.mean()), float(gmag.std())

    hh, ww = c.shape
    qy, qx = max(hh // 4, 1), max(ww // 4, 1)
    center = c[qy: hh - qy, qx: ww - qx]
    center_mean = float(center.mean()) if center.size else mean
    border_n = max(c.size - center.size, 1)
    border_mean = (float(c.sum()) - float(center.sum())) / border_n
    central_ratio = (center_mean + eps) / (border_mean + eps)

    g = _geom(c.shape)
    cz = c - mean

    # radial profile (z-scored → illumination invariant)
    ring_means = [float(cz[m].mean()) if m.any() else 0.0 for m in g["rings"]]
    ring_z = [rm / std for rm in ring_means]
    core_v, outer_v = ring_means[0], ring_means[3]
    core_outer = abs(core_v - outer_v) / std
    rvec = np.asarray(ring_means, dtype=np.float32)
    radial_mono = (float(abs(np.corrcoef(np.arange(4, dtype=np.float32), rvec)[0, 1]))
                   if rvec.std() > eps else 0.0)

    core_m = g["core"]
    if core_m.any():
        cm, am = cz[core_m], g["ang"][core_m]
        sect = np.array([cm[am == s].mean() if (am == s).any() else 0.0 for s in range(8)])
        radial_sym = float(sect.std()) / std
    else:
        radial_sym = 0.0

    # matched filter: |NCC| with a centred Gaussian template
    cn = float(np.sqrt((cz ** 2).sum())) + eps
    gauss_ncc = abs(float((cz * g["tmpl"]).sum()) / (cn * g["tmpl_norm"]))

    # local SNR: core vs outer-ring background
    bg = cz[g["outer"]] if g["outer"].any() else cz
    center_snr = abs(core_v - float(bg.mean())) / (float(bg.std()) + eps)
    peak_prom = max(abs(cmax - mean), abs(cmin - mean)) / std

    # structure-tensor coherence (carbon edges = one dominant gradient direction)
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    tr = jxx + jyy
    disc = max(tr * tr / 4.0 - (jxx * jyy - jxy * jxy), 0.0) ** 0.5
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    grad_coherence = (l1 - l2) / (l1 + l2 + eps)
    grad_center = ((float(gmag[core_m].mean()) + eps) / (float(gmag[g["outer"]].mean()) + eps)
                   if (core_m.any() and g["outer"].any()) else 1.0)

    # distribution shape & sharpness
    z = cz / std
    skew = float((z ** 3).mean())
    kurtosis = float((z ** 4).mean()) - 3.0
    lap = (-4.0 * c + np.roll(c, 1, 0) + np.roll(c, -1, 0)
           + np.roll(c, 1, 1) + np.roll(c, -1, 1))
    lap_var = float(lap.var()) / (std * std)

    return np.array([
        mean, std, cmin, cmax, contrast, grad_mean, grad_std, central_ratio,
        ring_z[0], ring_z[1], ring_z[2], ring_z[3], core_outer, radial_mono, radial_sym,
        gauss_ncc, center_snr, peak_prom, grad_coherence, grad_center,
        skew, kurtosis, lap_var,
    ], dtype=float)


def _crop(image: np.ndarray, x: float, y: float, box: int) -> np.ndarray:
    """Box×box crop centred on (x, y); out-of-frame padding filled with the local
    median so the particle stays centred (radial features need it) and no fake edge
    is introduced."""
    h, w = image.shape[:2]
    half = box // 2
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x0 + box), min(h, y0 + box)
    if ix1 <= ix0 or iy1 <= iy0:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.full((box, box), np.nan, dtype=np.float32)
    xs0, ys0 = ix0 - x0, iy0 - y0
    out[ys0:ys0 + (iy1 - iy0), xs0:xs0 + (ix1 - ix0)] = image[iy0:iy1, ix0:ix1]
    if np.isnan(out).any():
        out[np.isnan(out)] = float(np.median(image[iy0:iy1, ix0:ix1]))
    return out


def extract_features(image: np.ndarray, coords: Sequence, box: int = 64) -> np.ndarray:
    """Return an (N, n_features) array for N candidate centres `coords` (x, y)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    feats = np.zeros((len(coords), len(FEATURE_NAMES)), dtype=float)
    for i, (x, y) in enumerate(coords):
        feats[i] = crop_features(_crop(image, x, y, box))
    return feats
