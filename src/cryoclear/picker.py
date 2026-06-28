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


def _blob_pick(image: np.ndarray, particle_px: float = 27.0, threshold_rel: float = 0.16,
               min_distance: float | None = None, max_peaks: int = 2200) -> np.ndarray:
    """Band-pass (Difference-of-Gaussians) blob detector with non-max suppression.

    A LoG with a permissive threshold fires on every texture variation → a dense,
    non-human over-pick field. This instead band-passes the image to the particle
    scale (suppressing large-scale carbon gradients and pixel noise), then takes
    spaced local maxima (min_distance NMS, particles don't overlap). The result is
    distinct, particle-like candidates — still deliberately over-picking (~2x GT) so
    the junk classifier has something to triage, but no longer a uniform blanket.

    ``particle_px`` is the particle diameter in *this image's* pixels (the display
    image is downsampled by io_mrc factor, so ~108/4 ≈ 27 px for β-gal).
    """
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

    img = image.astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)  # np.ptp: ndarray.ptp() removed in NumPy 2.0
    sigma = max(particle_px / 4.0, 1.5)
    dog = gaussian_filter(img, sigma) - gaussian_filter(img, sigma * 1.6)
    resp = np.abs(dog)                              # particle blobs (dark or bright) → strong |DoG|
    md = int(min_distance if min_distance is not None else max(particle_px * 0.55, 4))
    pk = peak_local_max(resp, min_distance=md, threshold_rel=threshold_rel, num_peaks=max_peaks)
    if pk.size == 0:
        return np.zeros((0, 2))
    # peak_local_max returns (row, col) -> (x, y) = (col, row)
    return np.stack([pk[:, 1], pk[:, 0]], axis=1).astype(float)


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
