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
    args = ap.parse_args()
    cache = config.PROCESSED / args.empiar / "webcache"
    idx = json.loads((cache / "index.json").read_text())

    Xs, ys = [], []
    for m in idx["micrographs"]:
        d = np.load(cache / "data" / f"{m['stem']}.npz")
        k = d["true_junk"] != -1
        Xs.append(d["feats"][k])
        ys.append(d["true_junk"][k])
    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(int)
    print(f"training on {len(X)} labelled candidates ({y.mean():.2f} junk)", flush=True)

    models = {mt: JunkClassifier(model_type=mt).fit(X, y) for mt in ("rf", "lgbm")}
    for mt, clf in models.items():
        clf.save(config.PROCESSED / args.empiar / f"junk_{mt}.joblib")
    print("fitted rf + lgbm; caching scores…", flush=True)

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
