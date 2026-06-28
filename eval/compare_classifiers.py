"""Honest comparison: RandomForest vs LightGBM junk classifiers, on blob vs CryoSegNet
picks, micrograph-level held-out. Tells us the best (picker x model) combination.

  uv run python eval/compare_classifiers.py --empiar 10017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords, features, io_mrc, metrics  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def blob_data(empiar):
    cache = config.PROCESSED / empiar / "webcache"
    idx = json.loads((cache / "index.json").read_text())
    out = {}
    for m in idx["micrographs"]:
        d = np.load(cache / "data" / f"{m['stem']}.npz")
        out[m["stem"]] = {"feats": d["feats"], "tj": d["true_junk"],
                          "pred": d["pred_full"], "n_gt": m["n_gt"]}
    return out


def cryosegnet_data(empiar, radius, box):
    raw = config.RAW / empiar
    cdir = config.PROCESSED / empiar / "cryosegnet"
    out = {}
    for star in sorted(cdir.glob("*.star")):
        stem = star.stem
        mic, gt = raw / "micrographs" / f"{stem}.mrc", raw / "ground_truth" / f"{stem}.star"
        if not mic.exists() or not gt.exists():
            continue
        pred, gx = coords.read_star_coords(star), coords.read_star_coords(gt)
        if not len(pred):
            continue
        matches, _fp, _fn = metrics.match_particles(pred, gx, radius)
        tj = np.ones(len(pred), np.int8)
        for pi, _gi, _d in matches:
            tj[pi] = 0
        imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
        out[stem] = {"feats": features.extract_features(imgf, pred, box=box),
                     "tj": tj, "pred": pred, "n_gt": len(gx)}
    return out


def _f1(real, mask, tj, n_gt):
    tp = int((real & mask).sum())
    fp = int((~real & mask & (tj != -1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evalset(data, model_type, test_frac=0.3, seed=0):
    """Micrograph-level held-out: train on one split, evaluate junk-F1 and picking-F1
    (both at the default 0.5 threshold and at the threshold that maximises held-out
    picking-F1 — the honest achievable ceiling)."""
    stems = list(data.keys())
    order = np.random.default_rng(seed).permutation(len(stems))
    test = {stems[i] for i in order[:max(1, int(test_frac * len(stems)))]}
    train = [s for s in stems if s not in test and len(data[s]["feats"])]
    Xtr = np.vstack([data[s]["feats"] for s in train])
    ytr = np.concatenate([data[s]["tj"] for s in train]).astype(int)
    k = ytr != -1
    clf = JunkClassifier(model_type=model_type).fit(Xtr[k], ytr[k])

    jf, bf, rows = [], [], []
    for s in test:
        ff, tj, pred, n_gt = (data[s][x] for x in ("feats", "tj", "pred", "n_gt"))
        if not len(ff):
            continue
        proba = clf.predict_junk_proba(ff)
        jf.append(metrics.junk_rejection_metrics(proba >= 0.5, tj == 1)["junk_f1"])
        real = tj == 0
        bf.append(_f1(real, np.ones(len(pred), bool), tj, n_gt))
        rows.append((real, proba, tj, n_gt))

    def mean_after(t):
        return float(np.mean([_f1(real, proba < t, tj, n_gt) for real, proba, tj, n_gt in rows]))

    af05 = mean_after(0.5)
    best_t, best_af = 0.5, af05
    for t in np.linspace(0.1, 0.95, 35):
        m = mean_after(float(t))
        if m > best_af:
            best_af, best_t = m, float(t)
    return float(np.mean(jf)), float(np.mean(bf)), af05, best_af, best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    args = ap.parse_args()
    R, B = config.particle_radius_px(), config.DEMO_PARTICLE_DIAMETER_PX
    datasets = [("blob", blob_data(args.empiar))]
    cs = cryosegnet_data(args.empiar, R, B)
    if cs:
        datasets.append(("cryosegnet", cs))
    for pname, data in datasets:
        for mt in ("rf", "lgbm", "sgd"):
            jf, bf, af05, afb, bt = evalset(data, mt)
            print(f"RESULT picker={pname:10s} model={mt:4s}: junkF1={jf:.3f}  "
                  f"pickF1 raw={bf:.3f} -> after@0.5={af05:.3f} -> best={afb:.3f}@thr{bt:.2f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
