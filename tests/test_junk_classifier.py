import numpy as np

import pytest

from cryotriage import metrics


def test_junk_classifier_separates_easy_classes():
    sklearn = pytest.importorskip("sklearn")  # skip if sklearn not installed
    from cryotriage.junk_classifier import JunkClassifier

    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 0.3, size=(200, 8))
    junk = rng.normal(3.0, 0.3, size=(200, 8))
    X = np.vstack([real, junk])
    y = np.array([0] * 200 + [1] * 200)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = 320

    clf = JunkClassifier(n_estimators=50).fit(X[:cut], y[:cut])
    pred = clf.predict_is_junk(X[cut:])
    jr = metrics.junk_rejection_metrics(pred, y[cut:].astype(bool))
    assert jr["junk_f1"] > 0.9
