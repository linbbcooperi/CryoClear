"""Evaluation metrics for particle picking and junk rejection.

These are the numbers we show the judges:
  * picking precision / recall / F1 vs expert ground truth (CryoPPP)
  * junk-rejection precision / recall (positive class = junk)

A predicted particle is a True Positive if it lies within `radius` pixels of an
unmatched ground-truth particle. Matching is greedy by ascending distance, so
each ground-truth particle is matched at most once.

Only depends on numpy + scipy → safe to import anywhere and unit-test offline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

try:  # fast neighbor search; fall back to brute force if scipy missing
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class PickingScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def _as_xy(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr.reshape(-1, 2)


def match_particles(pred_xy: Sequence, gt_xy: Sequence, radius: float):
    """Greedy nearest-neighbour matching within `radius` pixels.

    Returns (matches, fp_idx, fn_idx) where matches is a list of
    (pred_index, gt_index, distance).
    """
    pred = _as_xy(pred_xy)
    gt = _as_xy(gt_xy)
    if len(pred) == 0 or len(gt) == 0:
        return [], set(range(len(pred))), set(range(len(gt)))

    # collect candidate pairs within radius
    pairs = []
    if _HAVE_SCIPY:
        tree = cKDTree(gt)
        neighbours = tree.query_ball_point(pred, r=radius)
        for i, js in enumerate(neighbours):
            for j in js:
                pairs.append((float(np.linalg.norm(pred[i] - gt[j])), i, j))
    else:  # pragma: no cover - brute force fallback
        d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
        for i in range(len(pred)):
            for j in range(len(gt)):
                if d[i, j] <= radius:
                    pairs.append((float(d[i, j]), i, j))

    pairs.sort(key=lambda t: t[0])
    used_p, used_g, matches = set(), set(), []
    for dist, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, dist))

    fp = set(range(len(pred))) - used_p
    fn = set(range(len(gt))) - used_g
    return matches, fp, fn


def picking_metrics(pred_xy: Sequence, gt_xy: Sequence, radius: float) -> PickingScore:
    """Precision / recall / F1 for particle picking against ground truth."""
    matches, fp, fn = match_particles(pred_xy, gt_xy, radius)
    tp, n_fp, n_fn = len(matches), len(fp), len(fn)
    precision = tp / (tp + n_fp) if (tp + n_fp) else 0.0
    recall = tp / (tp + n_fn) if (tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PickingScore(precision, recall, f1, tp, n_fp, n_fn)


def junk_rejection_metrics(pred_is_junk: Sequence, true_is_junk: Sequence) -> dict:
    """Binary metrics for the junk classifier (positive class = junk)."""
    p = np.asarray(pred_is_junk).astype(bool)
    t = np.asarray(true_is_junk).astype(bool)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same length")
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(p), 1)
    return {
        "junk_precision": precision,
        "junk_recall": recall,
        "junk_f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def format_score(score: PickingScore) -> str:
    return (
        f"Picking  P={score.precision:.3f}  R={score.recall:.3f}  "
        f"F1={score.f1:.3f}  (TP={score.tp} FP={score.fp} FN={score.fn})"
    )
