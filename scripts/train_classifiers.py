"""Train RF + LightGBM junk classifiers on the web cache and cache their per-candidate
scores (rf_scores / lgbm_scores) so the backend can switch between them instantly.
The CNN scores (cnn_scores) are cached separately by scripts/train_cnn.py.

  uv run python scripts/train_classifiers.py --empiar 10017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--score-from", default=None,
                    help="EMPIAR id whose trained rf/lgbm models to apply (for a dataset "
                         "with no ground-truth labels — a cross-protein generalisation demo)")
    args = ap.parse_args()
    cache = config.PROCESSED / args.empiar / "webcache"
    idx = json.loads((cache / "index.json").read_text())

    Xs, ys = [], []
    for m in idx["micrographs"]:
        d = np.load(cache / "data" / f"{m['stem']}.npz")
        k = d["true_junk"] != -1
        Xs.append(d["feats"][k])
        ys.append(d["true_junk"][k])
    y = np.concatenate(ys).astype(int) if ys else np.zeros(0, int)
    labelled = len(y) and len(np.unique(y)) > 1

    if args.score_from:                                   # no training — reuse another dataset's models
        src = config.PROCESSED / args.score_from
        models = {mt: JunkClassifier.load(src / f"junk_{mt}.joblib") for mt in ("rf", "lgbm")}
        print(f"scoring {args.empiar} with {args.score_from}'s rf + lgbm models (no GT)", flush=True)
    elif labelled:
        X = np.vstack(Xs)
        print(f"training on {len(X)} labelled candidates ({y.mean():.2f} junk)", flush=True)
        models = {mt: JunkClassifier(model_type=mt).fit(X, y) for mt in ("rf", "lgbm")}
        for mt, clf in models.items():
            clf.save(config.PROCESSED / args.empiar / f"junk_{mt}.joblib")
        print("fitted rf + lgbm; caching scores…", flush=True)
    else:
        print(f"no labels in {args.empiar} and no --score-from given; nothing to do", flush=True)
        return 1

    for m in idx["micrographs"]:
        stem = m["stem"]
        d = dict(np.load(cache / "data" / f"{stem}.npz"))
        f = d["feats"]
        for mt, clf in models.items():
            d[f"{mt}_scores"] = (clf.predict_junk_proba(f) if len(f)
                                 else np.zeros(0)).astype(np.float32)
        np.savez_compressed(cache / "data" / f"{stem}.npz", **d)
    print("cached rf_scores + lgbm_scores into webcache", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
