"""Human-in-the-loop active learning (Bindu).

Maintains a running buffer of user-confirmed labels and refits the junk
classifier so accuracy visibly improves during the live demo when Eva rejects
junk clusters / confirms particles.
"""
from __future__ import annotations

import numpy as np

from .junk_classifier import JunkClassifier


class ActiveLearner:
    def __init__(self, classifier: JunkClassifier | None = None):
        self.clf = classifier or JunkClassifier()
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._batches: list[int] = []   # size of each add_corrections call (for undo)

    def seed(self, features: np.ndarray, is_junk) -> "ActiveLearner":
        """Initialize the buffer with the pre-trained set (e.g. CryoPPP labels)."""
        for f, j in zip(np.asarray(features, dtype=float), np.asarray(is_junk).astype(int)):
            self._X.append(f)
            self._y.append(int(j))
        self._refit()
        return self

    def add_corrections(self, features: np.ndarray, is_junk) -> dict:
        """Add user corrections (rejected junk / confirmed particles) and refit.

        Refit on the full accumulated buffer (which always holds both classes from the
        cold-start seed). With the linear SGD backend this is a few-millisecond fit, so the
        "watch it learn" loop stays lag-free without the single-class ``partial_fit`` pitfall
        (balanced class weights can't be computed from an all-junk correction batch).
        """
        n0 = len(self._y)
        for f, j in zip(np.asarray(features, dtype=float), np.asarray(is_junk).astype(int)):
            self._X.append(f)
            self._y.append(int(j))
        self._batches.append(len(self._y) - n0)
        self._refit()
        return {"n_labels": len(self._y), "n_junk": int(sum(self._y))}

    def undo_last(self):
        """Remove the most recent correction batch and refit. Returns the removed
        (features, labels) for redo, or None if there is nothing to undo (the seed
        is never popped, so both classes always remain)."""
        if not self._batches:
            return None
        n = self._batches.pop()
        X = np.array(self._X[-n:]) if n else np.zeros((0, 0))
        y = np.array(self._y[-n:]) if n else np.zeros(0, int)
        if n:
            del self._X[-n:]
            del self._y[-n:]
        self._refit()
        return X, y

    def _refit(self) -> None:
        if len(set(self._y)) < 2:
            return  # need both classes before fitting
        self.clf.fit(np.vstack(self._X), np.asarray(self._y))

    def predict_is_junk(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.clf.predict_is_junk(features, threshold=threshold)
