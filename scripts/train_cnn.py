"""Train the CNN junk classifier (GPU) and evaluate it HONESTLY (held-out by micrograph).

Reads the web cache (backend/precompute.py), extracts 64x64 crops at every candidate,
trains a small CNN, reports held-out junk-F1 + picking-after-triage F1, saves the model,
and caches per-candidate CNN scores back into the cache for the backend.

  /workspace/cs_bw_venv/bin/python scripts/train_cnn.py --empiar 10017   # (needs torch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, io_mrc, metrics  # noqa: E402


def crops_for(mic: Path, xy: np.ndarray, box: int = 160, size: int = 64):
    from scipy.ndimage import zoom

    imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
    h, w = imgf.shape
    half = box // 2
    cs, valid = [], []
    for i, (x, y) in enumerate(xy):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        c = imgf[yi - half:yi + half, xi - half:xi + half].astype(np.float32)
        if c.shape[0] != box or c.shape[1] != box:
            continue
        c = zoom(c, size / box, order=1)
        c = (c - c.mean()) / (c.std() + 1e-6)
        cs.append(c)
        valid.append(i)
    return (np.stack(cs) if cs else np.zeros((0, size, size), np.float32)), np.array(valid, int)


def _picking_after(tjf, junk, n_gt):
    real = tjf == 0
    kept = ~junk
    tp = int((real & kept).sum())
    fp = int((~real & kept & (tjf != -1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_gt if n_gt else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--score-from", default=None,
                    help="EMPIAR id whose trained CNN (junk_cnn.pt) to apply (cross-protein, "
                         "no training — used to give every dataset cnn_scores)")
    ap.add_argument("--picker", default="blob", help="which webcache to score")
    args = ap.parse_args()

    from cryoclear.cnn_classifier import CNNJunkClassifier

    cache = config.PROCESSED / args.empiar / ("webcache" if args.picker == "blob"
                                              else f"webcache_{args.picker}")
    idx = json.loads((cache / "index.json").read_text())
    mics = idx["micrographs"]

    if args.score_from:                               # cross-apply an existing CNN, cache scores only
        model_path = config.PROCESSED / args.score_from / "junk_cnn.pt"
        clf = CNNJunkClassifier.load(model_path)
        n = 0
        for m in mics:
            stem = m["stem"]
            d = dict(np.load(cache / "data" / f"{stem}.npz"))
            mic = config.RAW / args.empiar / "micrographs" / f"{stem}.mrc"
            crops, valid = crops_for(mic, d["pred_full"])
            full = np.ones(len(d["pred_full"]), np.float32)
            if len(crops):
                full[valid] = clf.predict_junk_proba(crops).astype(np.float32)
            d["cnn_scores"] = full
            np.savez_compressed(cache / "data" / f"{stem}.npz", **d)
            n += 1
        print(f"cross-applied {args.score_from}'s CNN -> cached cnn_scores for {n} micrographs "
              f"of {args.empiar} ({args.picker})", flush=True)
        return 0

    data = {}
    for m in mics:
        stem = m["stem"]
        d = np.load(cache / "data" / f"{stem}.npz")
        mic = config.RAW / args.empiar / "micrographs" / f"{stem}.mrc"
        crops, valid = crops_for(mic, d["pred_full"])
        data[stem] = {"crops": crops, "valid": valid, "tjf": d["true_junk"],
                      "n_gt": m["n_gt"], "npick": len(d["pred_full"])}
    print(f"extracted crops for {len(data)} micrographs", flush=True)

    stems = list(data.keys())
    order = np.random.default_rng(0).permutation(len(stems))
    n_test = max(1, int(args.test_frac * len(stems)))
    test = {stems[i] for i in order[:n_test]}

    def stack(sset):
        Xs = [data[s]["crops"] for s in sset if len(data[s]["crops"])]
        ys = [data[s]["tjf"][data[s]["valid"]] for s in sset if len(data[s]["crops"])]
        X, y = np.concatenate(Xs), np.concatenate(ys)
        k = y != -1
        return X[k], y[k]

    Xtr, ytr = stack([s for s in stems if s not in test])
    Xv, yv = stack(list(test)[:2])
    print(f"train crops={len(Xtr)} (junk {ytr.mean():.2f})  val crops={len(Xv)}", flush=True)

    clf = CNNJunkClassifier()
    clf.fit(Xtr, ytr, epochs=args.epochs, val=(Xv, yv), log=lambda m: print(m, flush=True))

    jf, af = [], []
    for s in test:
        dd = data[s]
        sc = clf.predict_junk_proba(dd["crops"])
        full = np.ones(dd["npick"])
        full[dd["valid"]] = sc
        junk = full >= 0.5
        jf.append(metrics.junk_rejection_metrics(junk, dd["tjf"] == 1)["junk_f1"])
        af.append(_picking_after(dd["tjf"], junk, dd["n_gt"]))
    print(f"HELD-OUT CNN ({n_test} test mics): junk-F1={np.mean(jf):.3f}  "
          f"picking-after-F1={np.mean(af):.3f}", flush=True)

    # cache CNN scores for every micrograph (border crops default to junk=1.0)
    for s in stems:
        dd = data[s]
        full = np.ones(dd["npick"], np.float32)
        if len(dd["crops"]):
            full[dd["valid"]] = clf.predict_junk_proba(dd["crops"]).astype(np.float32)
        npz = dict(np.load(cache / "data" / f"{s}.npz"))
        npz["cnn_scores"] = full
        np.savez_compressed(cache / "data" / f"{s}.npz", **npz)

    out = config.PROCESSED / args.empiar / "junk_cnn.pt"
    clf.save(out)
    print(f"saved CNN -> {out}; cached cnn_scores into webcache", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
