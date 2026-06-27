"""Split CryoSegNet's combined .star into per-micrograph cache files.

CryoSegNet's generate_starfile_new_data_mrc.py writes ONE star with an
rlnMicrographName column; our `cryosegnet` picker backend reads one .star per
micrograph from data/processed/<empiar>/cryosegnet/<stem>.star. This bridges them.

  uv run python eval/split_cryosegnet_star.py --empiar 10017 \
       --star data/processed/10017/cryosegnet_raw/star_files/cryosegnet.star
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--star", required=True)
    args = ap.parse_args()

    import starfile

    cache = config.PROCESSED / args.empiar / "cryosegnet"
    cache.mkdir(parents=True, exist_ok=True)

    data = starfile.read(args.star)
    df = data
    if isinstance(data, dict):
        for block in data.values():
            if {"rlnCoordinateX", "rlnCoordinateY"}.issubset(getattr(block, "columns", [])):
                df = block
                break

    name_col = next((c for c in df.columns if "MicrographName" in c), None)
    if name_col is None:
        print("No rlnMicrographName column. Columns:", list(df.columns))
        return 1

    n = 0
    for name, grp in df.groupby(name_col):
        stem = Path(str(name)).stem
        xy = np.stack([grp["rlnCoordinateX"].to_numpy(dtype=float),
                       grp["rlnCoordinateY"].to_numpy(dtype=float)], axis=1)
        coords.write_star_coords(cache / f"{stem}.star", xy)
        n += 1
        print(f"  {stem}: {len(xy)} picks")
    print(f"split {n} micrographs -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
