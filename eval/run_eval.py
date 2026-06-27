"""Evaluation harness — the numbers we show the judges.

Usage:
  python eval/run_eval.py --demo
      Run on synthetic data (no downloads needed) to prove the harness works.

  python eval/run_eval.py --pred preds.box --gt gt.star --particle-radius 90
      Score real predictions vs ground truth.

Outputs picking precision/recall/F1 (+ junk-rejection metrics in --demo mode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryotriage import metrics  # noqa: E402


def _demo() -> int:
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 4000, size=(300, 2))
    # predictions: 270 true (jittered) + 40 false positives, 30 missed
    keep = gt[:270] + rng.normal(0, 15, size=(270, 2))
    fps = rng.uniform(0, 4000, size=(40, 2))
    pred = np.vstack([keep, fps])

    score = metrics.picking_metrics(pred, gt, radius=90)
    print("== Picking (synthetic) ==")
    print(metrics.format_score(score))

    # junk-rejection demo: 40 fps are junk, 270 keeps are real
    true_is_junk = np.array([False] * 270 + [True] * 40)
    pred_is_junk = true_is_junk.copy()
    pred_is_junk[:8] = True       # 8 false alarms
    pred_is_junk[-5:] = False     # 5 missed junk
    jr = metrics.junk_rejection_metrics(pred_is_junk, true_is_junk)
    print("\n== Junk rejection (synthetic) ==")
    print(f"P={jr['junk_precision']:.3f}  R={jr['junk_recall']:.3f}  "
          f"F1={jr['junk_f1']:.3f}  acc={jr['accuracy']:.3f}")
    return 0


def _score_files(pred_path: str, gt_path: str, radius: float) -> int:
    from cryotriage import coords

    pred = coords.read_box_coords(pred_path) if pred_path.endswith(".box") \
        else coords.read_star_coords(pred_path)
    gt = coords.read_box_coords(gt_path) if gt_path.endswith(".box") \
        else coords.read_star_coords(gt_path)
    score = metrics.picking_metrics(pred, gt, radius=radius)
    print(metrics.format_score(score))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CryoTriage evaluation harness")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--pred", help="predicted coords (.box or .star)")
    ap.add_argument("--gt", help="ground-truth coords (.box or .star)")
    ap.add_argument("--particle-radius", type=float, default=90.0,
                    help="match tolerance in pixels (≈ particle radius)")
    args = ap.parse_args()

    if args.demo or not (args.pred and args.gt):
        return _demo()
    return _score_files(args.pred, args.gt, args.particle_radius)


if __name__ == "__main__":
    raise SystemExit(main())
