"""Honest, micrograph-level held-out evaluation.

Trains the junk classifier on a subset of micrographs and tests on DIFFERENT
micrographs (not the in-sample random-candidate split). Reports held-out
junk-rejection F1 and picking F1 before/after junk triage — the rigorous numbers.

  uv run python eval/heldout_eval.py --empiar 10017 --radius 54 --box 108
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords, features, io_mrc, metrics, picker  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def _mic_data(mic, gt, radius, box, factor):
    imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
    pf = picker.pick(io_mrc.load_for_pipeline(mic, factor=factor), backend="blob") * float(factor)
    gt_xy = coords.read_star_coords(gt)
    matches, _fp, _fn = metrics.match_particles(pf, gt_xy, radius)
    is_junk = np.ones(len(pf), dtype=bool)
    for pi, _gi, _d in matches:
        is_junk[pi] = False
    feats = features.extract_features(imgf, pf, box=box)
    return pf, gt_xy, feats, is_junk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--radius", type=float, default=config.particle_radius_px())
    ap.add_argument("--box", type=int, default=config.DEMO_PARTICLE_DIAMETER_PX)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.34)
    ap.add_argument("--max-mics", type=int, default=30, help="cap micrographs (0 = all)")
    args = ap.parse_args()

    raw = config.RAW / args.empiar
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    gts = sorted((raw / "ground_truth").glob("*.star"))
    if args.max_mics:
        mics, gts = mics[:args.max_mics], gts[:args.max_mics]
    n = len(mics)
    if n < 3:
        print(f"Need >=3 micrographs (have {n}).")
        return 1
    n_test = max(1, int(round(args.test_frac * n)))
    order = np.random.default_rng(0).permutation(n)
    test_idx = set(order[:n_test].tolist())

    data = []
    for i in range(n):
        data.append(_mic_data(mics[i], gts[i], args.radius, args.box, args.factor))
        print(f"  processed {i + 1}/{n}", flush=True)
    Xtr = np.vstack([data[i][2] for i in range(n) if i not in test_idx])
    ytr = np.concatenate([data[i][3] for i in range(n) if i not in test_idx]).astype(int)
    clf = JunkClassifier().fit(Xtr, ytr)

    jf1, before, after, kept_frac = [], [], [], []
    for i in test_idx:
        pf, gt_xy, feats, true_j = data[i]
        pred_j = np.asarray(clf.predict_is_junk(feats), dtype=bool)
        jf1.append(metrics.junk_rejection_metrics(pred_j, true_j)["junk_f1"])
        before.append(metrics.picking_metrics(pf, gt_xy, args.radius).f1)
        after.append(metrics.picking_metrics(pf[~pred_j], gt_xy, args.radius).f1)
        kept_frac.append(float((~pred_j).mean()))

    print(f"HELD-OUT EVAL  EMPIAR-{args.empiar}: train {n - n_test} micrographs, "
          f"test {n_test} micrographs")
    print(f"  junk-rejection F1 (held-out): {np.mean(jf1):.3f}")
    print(f"  picking F1: raw={np.mean(before):.3f}  ->  after junk triage={np.mean(after):.3f}")
    print(f"  candidates kept after triage: {100*np.mean(kept_frac):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
