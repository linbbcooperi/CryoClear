"""The headline M1 number: picking precision/recall/F1 BEFORE vs AFTER junk triage.

For each micrograph: pick candidates -> classify junk -> drop junk -> re-score vs
CryoPPP ground truth. Shows that the junk classifier turns an over-picking baseline
into a clean set (precision jumps, recall mostly preserved).

  uv run python eval/junk_triage_improvement.py --empiar 10017 --backend blob \
       --radius 54 --box 108

NOTE: if the model was trained on these same micrographs this is in-sample
(illustrative). The held-out junk-rejection F1 from train_junk_classifier.py is the
rigorous metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords, features, io_mrc, metrics, picker  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--backend", default="blob", choices=["blob", "cryosegnet"])
    ap.add_argument("--radius", type=float, default=config.particle_radius_px())
    ap.add_argument("--box", type=int, default=config.DEMO_PARTICLE_DIAMETER_PX)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    raw = config.RAW / args.empiar
    model_path = config.PROCESSED / args.empiar / "junk_classifier.joblib"
    cache_dir = config.PROCESSED / args.empiar / "cryosegnet"
    clf = JunkClassifier.load(model_path)
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    gts = sorted((raw / "ground_truth").glob("*.star"))

    before, after, n_before, n_after = [], [], 0, 0
    for mic, gt in zip(mics, gts):
        img = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
        if args.backend == "cryosegnet":
            pred = picker.pick(img, backend="cryosegnet", name=mic.name, cache_dir=cache_dir)
        else:
            pred = picker.pick(io_mrc.load_for_pipeline(mic, factor=args.downsample),
                               backend="blob") * float(args.downsample)
        gt_xy = coords.read_star_coords(gt)
        feats = features.extract_features(img, pred, box=args.box)
        keep = ~clf.predict_is_junk(feats, threshold=args.threshold)
        before.append(metrics.picking_metrics(pred, gt_xy, args.radius))
        after.append(metrics.picking_metrics(pred[keep], gt_xy, args.radius))
        n_before += len(pred)
        n_after += int(keep.sum())

    def m(rows, k):
        return float(np.mean([getattr(s, k) for s in rows]))

    print(f"micrographs: {len(before)}   candidates kept: {n_after}/{n_before} "
          f"({100*n_after/max(n_before,1):.0f}%)")
    print(f"BEFORE junk triage:  P={m(before,'precision'):.3f}  R={m(before,'recall'):.3f}  "
          f"F1={m(before,'f1'):.3f}")
    print(f"AFTER  junk triage:  P={m(after,'precision'):.3f}  R={m(after,'recall'):.3f}  "
          f"F1={m(after,'f1'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
