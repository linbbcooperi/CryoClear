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

    def seed(self, features: np.ndarray, is_junk) -> "ActiveLearner":
        """Initialize the buffer with the pre-trained set (e.g. CryoPPP labels)."""
        for f, j in zip(np.asarray(features, dtype=float), np.asarray(is_junk).astype(int)):
            self._X.append(f)
            self._y.append(int(j))
        self._refit()
        return self

    def add_corrections(self, features: np.ndarray, is_junk) -> dict:
        """Add user corrections (rejected junk / confirmed particles) and refit."""
        for f, j in zip(np.asarray(features, dtype=float), np.asarray(is_junk).astype(int)):
            self._X.append(f)
            self._y.append(int(j))
        self._refit()
        return {"n_labels": len(self._y), "n_junk": int(sum(self._y))}

    def _refit(self) -> None:
        if len(set(self._y)) < 2:
            return  # need both classes before fitting
        self.clf.fit(np.vstack(self._X), np.asarray(self._y))

    def predict_is_junk(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.clf.predict_is_junk(features, threshold=threshold)
