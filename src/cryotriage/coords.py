"""Coordinate I/O: .star (RELION/cryoSPARC) and .box (crYOLO/EMAN).

`starfile` is imported lazily. Coordinates are returned as an (N, 2) array of
(x, y) centres in pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_star_coords(path: str | Path) -> np.ndarray:
    """Read particle (x, y) centres from a RELION-style .star file."""
    import starfile

    df = starfile.read(str(path))
    # starfile may return a dict of blocks; grab the one with coordinate columns
    if isinstance(df, dict):
        for block in df.values():
            if {"rlnCoordinateX", "rlnCoordinateY"}.issubset(block.columns):
                df = block
                break
    x = df["rlnCoordinateX"].to_numpy(dtype=float)
    y = df["rlnCoordinateY"].to_numpy(dtype=float)
    return np.stack([x, y], axis=1)


def read_box_coords(path: str | Path) -> np.ndarray:
    """Read EMAN/.box (x, y, w, h) and return box CENTRES as (N, 2)."""
    rows = np.loadtxt(path, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 2))
    x, y, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    return np.stack([x + w / 2.0, y + h / 2.0], axis=1)


def write_box_coords(path: str | Path, centres: np.ndarray, box: int = 128) -> None:
    """Write centres as an EMAN .box file (top-left origin)."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    half = box / 2.0
    with open(path, "w") as fh:
        for x, y in centres:
            fh.write(f"{x - half:.1f}\t{y - half:.1f}\t{box}\t{box}\n")
