"""Parallel precompute of the per-micrograph web cache (uses all CPU cores).

For each micrograph: downsampled display PNG + candidate picks (display & full-res),
junk probabilities, ground-truth-derived true labels. Cached so the FastAPI backend
serves everything instantly (this is the real-time speed unlock alongside the GPU).

  uv run python backend/precompute.py --empiar 10017
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config  # noqa: E402

_CLF = None  # per-worker model


def cache_dir(empiar: str) -> Path:
    return config.PROCESSED / empiar / "webcache"


def _init_worker(model_path: str):
    global _CLF
    from cryoclear.junk_classifier import JunkClassifier
    _CLF = JunkClassifier.load(model_path) if Path(model_path).exists() else None


def _process_one(task: dict) -> dict:
    import matplotlib.image as mpimg
    import numpy as np

    from cryoclear import coords, features, io_mrc, metrics, picker

    mic = Path(task["mic"])
    gt = Path(task["gt"]) if task["gt"] else None
    factor, box, radius = task["factor"], task["box"], task["radius"]
    out = Path(task["out"])
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
    img = io_mrc.load_for_pipeline(mic, factor=factor)              # downsampled 8-bit
    mpimg.imsave(out / "img" / f"{mic.stem}.png", img, cmap="gray", vmin=0, vmax=255)

    pred_disp = picker.pick(img, backend="blob")
    pred_full = pred_disp * float(factor)
    feats = features.extract_features(imgf, pred_full, box=box)
    try:
        scores = (_CLF.predict_junk_proba(feats) if (_CLF is not None and len(feats))
                  else np.zeros(len(pred_full)))
    except Exception:        # stale model (e.g. feature-dim change) → real scores come from train_classifiers
        scores = np.zeros(len(pred_full))

    true_junk = np.full(len(pred_full), -1, dtype=np.int8)          # -1 = unknown
    n_gt = 0
    if gt is not None and gt.exists():
        gt_xy = coords.read_star_coords(gt)
        n_gt = len(gt_xy)
        matches, _fp, _fn = metrics.match_particles(pred_full, gt_xy, radius)
        tj = np.ones(len(pred_full), dtype=np.int8)
        for pi, _gi, _d in matches:
            tj[pi] = 0
        true_junk = tj

    payload = dict(
        pred_disp=pred_disp.astype(np.float32),
        pred_full=pred_full.astype(np.float32),
        scores=scores.astype(np.float32),
        true_junk=true_junk,
        feats=feats.astype(np.float32),
    )
    # carry over CNN scores (computed on raw crops, independent of these features)
    # so re-running precompute after a feature change doesn't force a GPU retrain.
    npz_path = out / "data" / f"{mic.stem}.npz"
    if npz_path.exists():
        try:
            old = np.load(npz_path)
            if "cnn_scores" in old.files and len(old["cnn_scores"]) == len(pred_full):
                payload["cnn_scores"] = old["cnn_scores"]
        except Exception:
            pass
    np.savez_compressed(npz_path, **payload)
    return {"stem": mic.stem, "n_picks": int(len(pred_full)),
            "h": int(img.shape[0]), "w": int(img.shape[1]), "n_gt": int(n_gt)}


def precompute(empiar: str, factor: int = 4, box: int | None = None,
               radius: float | None = None, workers: int | None = None) -> dict:
    box = box or config.DEMO_PARTICLE_DIAMETER_PX
    radius = radius or config.particle_radius_px()
    raw = config.RAW / empiar
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    gts = {g.stem: g for g in (raw / "ground_truth").glob("*.star")}
    out = cache_dir(empiar)
    model = config.PROCESSED / empiar / "junk_classifier.joblib"

    tasks = [{"mic": str(m), "gt": str(gts.get(m.stem, "")) or "",
              "factor": factor, "box": box, "radius": radius, "out": str(out)}
             for m in mics]
    import os
    workers = workers or min(len(tasks), max(1, (os.cpu_count() or 8) - 2))
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(model),)) as ex:
        for r in ex.map(_process_one, tasks):
            results.append(r)

    index = {"empiar": empiar, "factor": factor, "box": box, "radius": radius,
             "micrographs": results}
    (out).mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--box", type=int, default=None, help="particle box in full-res px (per dataset)")
    ap.add_argument("--radius", type=float, default=None, help="GT match radius in full-res px")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    idx = precompute(args.empiar, factor=args.factor, box=args.box,
                     radius=args.radius, workers=args.workers)
    print(f"cached {len(idx['micrographs'])} micrographs -> {cache_dir(args.empiar)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
