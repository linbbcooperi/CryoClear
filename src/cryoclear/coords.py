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


def read_cryoppp_csv(path: str | Path) -> np.ndarray:
    """Read CryoPPP ground-truth coordinates (.csv) -> (N, 2) (x, y) centres.

    CryoPPP ships one CSV per micrograph with columns `X-Coordinate`,
    `Y-Coordinate` (+ `Diameter`). Robust to minor header casing/spacing.
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.strip().lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    xcol = cols.get("x_coordinate") or cols.get("xcoordinate") or cols.get("x")
    ycol = cols.get("y_coordinate") or cols.get("ycoordinate") or cols.get("y")
    if xcol is None or ycol is None:
        raise ValueError(f"{path}: no X/Y-Coordinate columns (found {list(df.columns)})")
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    return np.stack([x, y], axis=1)


def cryoppp_csv_diameter(path: str | Path) -> float | None:
    """Median particle Diameter (px) from a CryoPPP CSV, or None if absent.

    Handy for setting the picking-match radius per dataset.
    """
    import pandas as pd

    df = pd.read_csv(path)
    for c in df.columns:
        if c.strip().lower().startswith("diameter"):
            vals = df[c].to_numpy(dtype=float)
            return float(np.median(vals)) if len(vals) else None
    return None


def write_star_coords(path: str | Path, centres: np.ndarray) -> None:
    """Write (x, y) centres as a minimal RELION .star (rlnCoordinateX/Y).

    Round-trips with `read_star_coords`, so converted CryoPPP CSVs drop straight
    into the existing pipeline.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    with open(path, "w") as fh:
        fh.write("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n")
        for x, y in centres:
            fh.write(f"{x:.2f}\t{y:.2f}\n")


def csv_to_star(csv_path: str | Path, star_path: str | Path) -> int:
    """Convert one CryoPPP coordinate CSV to a RELION .star. Returns N particles."""
    centres = read_cryoppp_csv(csv_path)
    write_star_coords(star_path, centres)
    return len(centres)
