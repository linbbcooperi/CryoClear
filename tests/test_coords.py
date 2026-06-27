"""Tests for coordinate I/O, incl. the CryoPPP CSV -> .star conversion."""
import numpy as np
import pytest

from cryoclear import coords


def _write_csv(path, rows, header="X-Coordinate,Y-Coordinate,Diameter"):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


def test_read_cryoppp_csv(tmp_path):
    csv = tmp_path / "mic.csv"
    _write_csv(csv, [(100, 200, 176), (150.5, 250.5, 176)])
    xy = coords.read_cryoppp_csv(csv)
    assert xy.shape == (2, 2)
    assert np.allclose(xy, [[100, 200], [150.5, 250.5]])


def test_csv_to_star_roundtrip(tmp_path):
    csv = tmp_path / "mic.csv"
    pts = [(100, 200, 176), (150.5, 250.5, 176), (300, 400, 176)]
    _write_csv(csv, pts)
    star = tmp_path / "mic.star"
    n = coords.csv_to_star(csv, star)
    assert n == 3
    back = coords.read_star_coords(star)  # round-trips through the real starfile reader
    assert np.allclose(back, [[100, 200], [150.5, 250.5], [300, 400]])


def test_cryoppp_csv_diameter(tmp_path):
    csv = tmp_path / "mic.csv"
    _write_csv(csv, [(1, 1, 170), (2, 2, 180), (3, 3, 190)])
    assert coords.cryoppp_csv_diameter(csv) == 180.0


def test_read_cryoppp_csv_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    _write_csv(csv, [(1, 2)], header="foo,bar")
    with pytest.raises(ValueError):
        coords.read_cryoppp_csv(csv)
