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


def _blob_pick(image: np.ndarray, min_sigma: float = 4, max_sigma: float = 12,
               threshold: float = 0.02) -> np.ndarray:
    """Placeholder blob detector (Laplacian-of-Gaussian). Replace with a real picker.

    Good enough to produce candidate boxes so the UI/metrics/junk pipeline runs.
    """
    from skimage.feature import blob_log

    img = image.astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)  # np.ptp: ndarray.ptp() removed in NumPy 2.0
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    if blobs.size == 0:
        return np.zeros((0, 2))
    # blob_log returns (row, col, sigma) -> (x, y) = (col, row)
    return np.stack([blobs[:, 1], blobs[:, 0]], axis=1)


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
