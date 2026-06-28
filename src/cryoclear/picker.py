"""Picking-engine wrapper (Bindu).

Abstracts over the picker so the rest of the app just calls `pick(image)` and
gets candidate (x, y) centres. Three backends behind one interface:

  * ``blob``       — dependency-light Laplacian-of-Gaussian placeholder so the UI
                     and pipeline run end-to-end before the real model is wired.
  * ``cryosegnet`` — our real engine. CryoSegNet runs offline on the GPU (slow,
                     needs torch + SAM) and writes one ``.star`` per micrograph;
                     this backend just *reads the cached ``.star``* so the live
                     demo never blocks on GPU inference. See scripts/run_cryosegnet.py.
  * ``topaz``      — optional baseline; still a stub.

Coordinate space: CryoSegNet picks from the full-resolution ``.mrc``, so cached
centres are in FULL-RES pixels. If you are overlaying them on a downsampled
display image (io_mrc.load_for_pipeline uses factor=4), pass ``scale=factor`` to
divide the centres into the displayed image's coordinate space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def pick(image: np.ndarray, backend: str = "blob", **kwargs) -> np.ndarray:
    """Return candidate particle centres as an (N, 2) array of (x, y)."""
    if backend == "blob":
        return _blob_pick(image, **kwargs)
    if backend == "cryosegnet":
        return _cryosegnet_pick(image, **kwargs)
    if backend == "topaz":
        return _topaz_pick(image, **kwargs)
    raise ValueError(f"unknown backend: {backend}")


def _blob_pick(image: np.ndarray, particle_px: float = 27.0, threshold_rel: float = 0.035,
               min_distance: float | None = None, max_peaks: int = 2600) -> np.ndarray:
    """Locally-normalised matched-filter (DoG band-pass) detector with NMS.

    Design note (why we over-pick on purpose): a *too-precise* picker leaves the junk
    classifier nothing distinguishable to remove, so after-triage F1 ≈ raw. We instead
    deliberately admit the **distinguishable** junk — carbon-film edges, hole rims, ice —
    as candidates, and let the classifier reject them (carbon is a coherent edge; the
    23-feature classifier has `grad_coherence`/`lap_var` for exactly this). Net effect:
    the kept (green) set stays clean *and* triage measurably beats raw picks, while a
    pure background blob that genuinely resembles a particle is the only thing neither
    stage can separate. Only the *border* is hard-excluded (no particle is half-off-frame).

    ``particle_px`` is the particle diameter in this image's pixels (~108/4 ≈ 27 for β-gal).
    """
    from scipy.ndimage import gaussian_filter, uniform_filter
    from skimage.feature import peak_local_max

    img = image.astype(np.float32)
    win = int(max(particle_px * 2.5, 8))
    # local contrast normalisation (per-window z-score) so the response is comparable
    # everywhere — low-contrast true particles clear threshold; bright ice can't dominate.
    mu = uniform_filter(img, win)
    var = uniform_filter(img * img, win) - mu * mu
    norm = (img - mu) / np.sqrt(np.maximum(var, 1e-6))
    sigma = max(particle_px / 4.0, 1.5)
    resp = np.abs(gaussian_filter(norm, sigma) - gaussian_filter(norm, sigma * 1.6))
    md = int(min_distance if min_distance is not None else max(particle_px * 0.55, 4))
    pk = peak_local_max(resp, min_distance=md, threshold_rel=threshold_rel, num_peaks=max_peaks)
    if pk.size == 0:
        return np.zeros((0, 2))
    b = int(particle_px)                            # drop only half-off-frame border peaks
    h, w = img.shape
    keep = (pk[:, 0] >= b) & (pk[:, 0] < h - b) & (pk[:, 1] >= b) & (pk[:, 1] < w - b)
    pk = pk[keep]
    if pk.size == 0:
        return np.zeros((0, 2))
    return np.stack([pk[:, 1], pk[:, 0]], axis=1).astype(float)  # (row,col) -> (x,y)


def _cryosegnet_pick(
    image: np.ndarray | None = None,
    *,
    star_path: str | Path | None = None,
    name: str | None = None,
    cache_dir: str | Path | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Read CryoSegNet's cached picks for one micrograph as (N, 2) (x, y) centres.

    CryoSegNet is run once on the GPU (scripts/run_cryosegnet.py) and writes a
    ``.star`` per micrograph; here we just load it — keeping the contract identical
    to ``_blob_pick`` so nothing downstream changes.

    Provide either ``star_path`` directly, or ``name`` (the micrograph filename,
    e.g. ``mic_0001.mrc``) + ``cache_dir`` (the folder of cached ``.star`` files).
    ``scale`` divides the returned centres (use the io_mrc downsample factor when
    overlaying on a downsampled display image).
    """
    from . import coords

    path = _resolve_star(star_path, name, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached CryoSegNet picks at {path}. Run scripts/run_cryosegnet.py "
            f"on the GPU first (writes one .star per micrograph)."
        )
    centres = coords.read_star_coords(path)
    if scale and scale != 1.0:
        centres = centres / float(scale)
    return centres


def _resolve_star(star_path, name, cache_dir) -> Path:
    if star_path is not None:
        return Path(star_path)
    if name is not None and cache_dir is not None:
        return Path(cache_dir) / f"{Path(name).stem}.star"
    raise ValueError(
        "cryosegnet backend needs either star_path, or name + cache_dir."
    )


def _topaz_pick(image: np.ndarray, model: str | None = None) -> np.ndarray:  # pragma: no cover
    """TODO: shell out to Topaz (`topaz extract`) or call its API, parse coords.

    See https://github.com/tbepler/topaz . Wire this on the GPU workstation and
    keep the same (N, 2) return contract so nothing downstream changes.
    """
    raise NotImplementedError("Wire Topaz on the GPU box; return (N,2) centres.")
