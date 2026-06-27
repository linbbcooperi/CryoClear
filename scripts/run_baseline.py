"""Score a picker vs CryoPPP ground truth (the M1 number + the comparison bar).

Pipeline:
  1. load MRC micrographs for an EMPIAR id (cryoclear.io_mrc)
  2. pick particles (cryosegnet = cached .star picks, or blob placeholder)
  3. score vs CryoPPP ground truth (cryoclear.metrics)

Examples:
  python scripts/run_baseline.py --backend cryosegnet --empiar 10025
  python scripts/run_baseline.py --backend blob       --empiar 10025

Coordinate space: GT .star is in full-resolution pixels. CryoSegNet picks are
also full-res (read straight from cache). The blob picker runs on the downsampled
display image, so its centres are scaled back up by `--downsample` to compare
fairly against full-res GT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords, io_mrc, metrics, picker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--backend", default="cryosegnet",
                    choices=["blob", "cryosegnet", "topaz"])
    ap.add_argument("--radius", type=float, default=config.particle_radius_px())
    ap.add_argument("--downsample", type=int, default=4,
                    help="io_mrc factor; blob picks are scaled back up by this for scoring")
    ap.add_argument("--cache-dir", default=None,
                    help="cryosegnet cached .star dir (default data/processed/<empiar>/cryosegnet)")
    args = ap.parse_args()

    raw = config.RAW / args.empiar
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    gts = sorted((raw / "ground_truth").glob("*.star"))
    if not mics:
        print(f"No micrographs in {raw}/micrographs — see scripts/download_cryoppp.py")
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        config.PROCESSED / args.empiar / "cryosegnet")

    agg = []
    for mic, gt in zip(mics, gts):
        img = io_mrc.load_for_pipeline(mic, factor=args.downsample)
        if args.backend == "cryosegnet":
            # cached picks are full-res; GT is full-res → compare natively
            pred = picker.pick(img, backend="cryosegnet", name=mic.name, cache_dir=cache_dir)
        else:
            # blob runs on the downsampled image → scale centres back to full-res
            pred = picker.pick(img, backend=args.backend) * float(args.downsample)
        score = metrics.picking_metrics(pred, coords.read_star_coords(gt), radius=args.radius)
        agg.append(score)
        print(f"{mic.name}: {metrics.format_score(score)}")

    if agg:
        import numpy as np
        print("\nMean F1:", float(np.mean([s.f1 for s in agg])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
