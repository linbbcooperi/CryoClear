"""Train the junk classifier on CryoPPP keep-vs-junk labels.

Pipeline (real path):
  1. for each micrograph: load full-res image + CryoSegNet cached candidate coords
  2. label each candidate keep/junk by matching against CryoPPP ground truth
     (matches a GT particle within particle_radius -> keep; otherwise -> junk FP)
  3. extract features (cryoclear.features.extract_features)
  4. fit JunkClassifier on a train split; report junk-rejection metrics on a held-out
     split; save the model for the app to load.

  python scripts/train_junk_classifier.py --empiar 10025      # real CryoPPP path
  python scripts/train_junk_classifier.py --demo              # synthetic sanity loop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords, features, io_mrc, metrics, picker  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def _demo() -> int:
    rng = np.random.default_rng(0)
    # real particles: low contrast, blob-like; junk: high contrast / strong edges
    real = rng.normal([0.4, 0.1, 0, 0.6, 0.3, 0.2, 0.1, 1.4], 0.05, size=(400, 8))
    junk = rng.normal([0.7, 0.3, 0, 0.95, 0.8, 0.6, 0.4, 0.9], 0.05, size=(200, 8))
    X = np.vstack([real, junk])
    y = np.array([0] * 400 + [1] * 200)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int(0.8 * len(y))
    clf = JunkClassifier().fit(X[:cut], y[:cut])
    pred = clf.predict_is_junk(X[cut:])
    jr = metrics.junk_rejection_metrics(pred, y[cut:].astype(bool))
    print("Junk classifier (synthetic):",
          f"P={jr['junk_precision']:.3f} R={jr['junk_recall']:.3f} "
          f"F1={jr['junk_f1']:.3f} acc={jr['accuracy']:.3f}")
    return 0


def _label_candidates(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> np.ndarray:
    """is_junk[i] = True if candidate i does NOT match a GT particle within `radius`.

    Matched candidates are real particles (keep); unmatched ones are the picker's
    false positives — exactly the junk the classifier learns to reject.
    """
    matches, fp, _fn = metrics.match_particles(pred_xy, gt_xy, radius)
    is_junk = np.ones(len(np.asarray(pred_xy, dtype=float).reshape(-1, 2)), dtype=bool)
    for i, _j, _d in matches:
        is_junk[i] = False
    return is_junk


def _train_real(empiar: str, cache_dir: Path, radius: float, box: int,
                out_path: Path, backend: str = "cryosegnet", downsample: int = 4,
                test_frac: float = 0.2, seed: int = 0) -> int:
    raw = config.RAW / empiar
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    gts = sorted((raw / "ground_truth").glob("*.star"))
    if not mics:
        print(f"No micrographs in {raw}/micrographs — download EMPIAR-{empiar} first.")
        return 1

    feats_all, labels_all = [], []
    for mic, gt in zip(mics, gts):
        # full-res image so candidate coords (full-res) align with the crops
        img = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
        if backend == "cryosegnet":
            try:
                pred = picker.pick(img, backend="cryosegnet", name=mic.name, cache_dir=cache_dir)
            except FileNotFoundError as e:
                print(f"  skip {mic.name}: {e}")
                continue
        else:  # blob runs on the downsampled image -> scale centres back to full-res
            pred = picker.pick(io_mrc.load_for_pipeline(mic, factor=downsample),
                               backend="blob") * float(downsample)
        if len(pred) == 0:
            continue
        gt_xy = coords.read_star_coords(gt)
        is_junk = _label_candidates(pred, gt_xy, radius)
        feats_all.append(features.extract_features(img, pred, box=box))
        labels_all.append(is_junk)
        print(f"  {mic.name}: {len(pred)} candidates "
              f"({int(is_junk.sum())} junk / {int((~is_junk).sum())} keep)")

    if not feats_all:
        print("No labeled candidates — check the CryoSegNet cache (scripts/run_cryosegnet.py).")
        return 1

    X = np.vstack(feats_all)
    y = np.concatenate(labels_all).astype(int)
    if len(set(y)) < 2:
        print(f"Only one class present (junk_count={int(y.sum())}); need both keep and junk.")
        return 1

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int((1 - test_frac) * len(y))
    clf = JunkClassifier().fit(X[:cut], y[:cut])
    pred_junk = clf.predict_is_junk(X[cut:])
    jr = metrics.junk_rejection_metrics(pred_junk, y[cut:].astype(bool))
    print(f"\nJunk classifier (EMPIAR-{empiar}, n={len(y)}): "
          f"P={jr['junk_precision']:.3f} R={jr['junk_recall']:.3f} "
          f"F1={jr['junk_f1']:.3f} acc={jr['accuracy']:.3f}")

    # refit on ALL data and save for the app
    JunkClassifier().fit(X, y).save(out_path)
    # also save the labelled feature table so the app can seed the live active-learning
    # loop (M2) from real examples
    train_path = out_path.parent / "junk_train.npz"
    np.savez_compressed(train_path, X=X.astype(np.float32), y=y.astype(np.int8))
    print(f"Saved model -> {out_path}")
    print(f"Saved training table -> {train_path}  (X={X.shape}, for active-learning seed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic sanity loop (no data needed)")
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--backend", default="cryosegnet", choices=["blob", "cryosegnet"],
                    help="candidate source: cryosegnet (cached .star) or blob")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--radius", type=float, default=config.particle_radius_px())
    ap.add_argument("--box", type=int, default=config.DEMO_PARTICLE_DIAMETER_PX,
                    help="crop size (px) for features; ~particle diameter at full res")
    ap.add_argument("--cache-dir", default=None,
                    help="CryoSegNet cached .star dir (default data/processed/<empiar>/cryosegnet)")
    ap.add_argument("--out", default=None,
                    help="model output path (default data/processed/<empiar>/junk_classifier.joblib)")
    args = ap.parse_args()
    if args.demo:
        return _demo()

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        config.PROCESSED / args.empiar / "cryosegnet")
    out_path = Path(args.out) if args.out else (
        config.PROCESSED / args.empiar / "junk_classifier.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _train_real(args.empiar, cache_dir, args.radius, args.box, out_path,
                       backend=args.backend, downsample=args.downsample)


if __name__ == "__main__":
    raise SystemExit(main())
