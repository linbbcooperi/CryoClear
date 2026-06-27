"""Run CryoSegNet on the GPU once and cache its picks as one .star per micrograph.

CryoSegNet (https://github.com/jianlin-cheng/CryoSegNet) is heavy (torch + SAM)
and slow at inference, so we run it ONCE offline here and cache the results. The
live app/pipeline then read the cached `.star` files via the `cryosegnet` picker
backend and never block on the GPU.

Prereqs on the GPU box (see the build plan, Phase A):
  git clone https://github.com/jianlin-cheng/CryoSegNet
  conda env create -f environment.yml && conda activate cryosegnet
  curl https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz -o w.tar.gz
  tar -xvf w.tar.gz

Usage (from the cryosegnet conda env):
  python scripts/run_cryosegnet.py --cryosegnet-dir /workspace/CryoSegNet --empiar 10025
  python scripts/run_cryosegnet.py --cryosegnet-dir /workspace/CryoSegNet --dry-run   # just print commands

Output: data/processed/<empiar>/cryosegnet/<micrograph_stem>.star

NOTE: CryoSegNet's exact output filenames vary by version. This wrapper runs its
two entrypoints, then collects every produced `.star` into the cache layout we
expect. If the collection step finds nothing, inspect `--output_path` and adjust
`_collect_stars` (it's intentionally simple).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config  # noqa: E402


def _run(cmd: list[str], cwd: Path, dry: bool) -> None:
    print("  $", " ".join(str(c) for c in cmd), f"   (cwd={cwd})")
    if not dry:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def _split_combined_star(combined: Path, cache_dir: Path) -> int:
    """Split CryoSegNet's combined .star (rlnMicrographName + coords) into one
    cache .star per micrograph, matching the `cryosegnet` picker backend."""
    import numpy as np
    import starfile

    from cryoclear import coords

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not combined.exists():
        return 0
    data = starfile.read(str(combined))
    df = data
    if isinstance(data, dict):
        for block in data.values():
            if {"rlnCoordinateX", "rlnCoordinateY"}.issubset(getattr(block, "columns", [])):
                df = block
                break
    name_col = next((c for c in df.columns if "MicrographName" in c), None)
    if name_col is None:
        return 0
    n = 0
    for name, grp in df.groupby(name_col):
        stem = Path(str(name)).stem
        xy = np.stack([grp["rlnCoordinateX"].to_numpy(dtype=float),
                       grp["rlnCoordinateY"].to_numpy(dtype=float)], axis=1)
        coords.write_star_coords(cache_dir / f"{stem}.star", xy)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cryosegnet-dir", required=True,
                    help="path to the cloned CryoSegNet repo (with weights downloaded)")
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to run CryoSegNet (use its conda env's python, "
                         "e.g. /workspace/miniconda3/envs/cryosegnet/bin/python)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run")
    args = ap.parse_args()

    cs_dir = Path(args.cryosegnet_dir).resolve()
    mic_dir = (config.RAW / args.empiar / "micrographs").resolve()
    out_dir = (config.PROCESSED / args.empiar / "cryosegnet_raw").resolve()
    cache_dir = (config.PROCESSED / args.empiar / "cryosegnet").resolve()

    if not mic_dir.exists():
        print(f"No micrographs at {mic_dir} — download EMPIAR-{args.empiar} first.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CryoSegNet repo : {cs_dir}")
    print(f"Micrographs     : {mic_dir}")
    print(f"Raw output      : {out_dir}")
    print(f"Cache (.star)   : {cache_dir}\n")

    star_dir = out_dir / "star_files"
    star_dir.mkdir(parents=True, exist_ok=True)
    combined = star_dir / "cryosegnet.star"

    # generate_starfile_new_data_mrc.py runs SAM inference itself and writes ONE
    # combined .star (rlnMicrographName + coords). Pass it the dataset/output paths.
    _run([args.python, "generate_starfile_new_data_mrc.py",
          "--my_dataset_path", str(mic_dir),
          "--output_path", str(out_dir),
          "--file_name", "cryosegnet.star",
          "--device", args.device], cwd=cs_dir, dry=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] skipped execution + star split.")
        return 0

    n = _split_combined_star(combined, cache_dir)
    print(f"\nSplit {n} micrograph .star file(s) -> {cache_dir}")
    if n == 0:
        print(f"WARNING: no per-micrograph picks. Check {combined} and _split_combined_star().")
        return 1
    print("Done. Score with:  python scripts/run_baseline.py --backend cryosegnet "
          f"--empiar {args.empiar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
