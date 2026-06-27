import numpy as np

from cryoclear import metrics


def test_perfect_match():
    gt = np.array([[10, 10], [100, 100], [200, 50]], float)
    score = metrics.picking_metrics(gt.copy(), gt, radius=5)
    assert score.tp == 3 and score.fp == 0 and score.fn == 0
    assert score.precision == 1.0 and score.recall == 1.0 and score.f1 == 1.0


def test_fp_and_fn():
    gt = np.array([[10, 10], [100, 100]], float)
    pred = np.array([[11, 9], [500, 500]], float)  # 1 TP, 1 FP, 1 FN
    score = metrics.picking_metrics(pred, gt, radius=5)
    assert score.tp == 1 and score.fp == 1 and score.fn == 1
    assert abs(score.precision - 0.5) < 1e-9
    assert abs(score.recall - 0.5) < 1e-9


def test_each_gt_matched_once():
    gt = np.array([[50, 50]], float)
    pred = np.array([[50, 50], [51, 50]], float)  # two preds, one GT -> 1 TP, 1 FP
    score = metrics.picking_metrics(pred, gt, radius=5)
    assert score.tp == 1 and score.fp == 1 and score.fn == 0


def test_empty_inputs():
    assert metrics.picking_metrics([], [], radius=5).f1 == 0.0
    assert metrics.picking_metrics([[1, 1]], [], radius=5).fp == 1
    assert metrics.picking_metrics([], [[1, 1]], radius=5).fn == 1


def test_junk_rejection():
    true_junk = np.array([False, False, True, True])
    pred_junk = np.array([False, True, True, True])  # 2 TP, 1 FP, 0 FN
    jr = metrics.junk_rejection_metrics(pred_junk, true_junk)
    assert jr["tp"] == 2 and jr["fp"] == 1 and jr["fn"] == 0
    assert abs(jr["junk_recall"] - 1.0) < 1e-9
