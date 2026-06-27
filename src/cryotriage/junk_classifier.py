"""The novel piece: a lightweight junk classifier.

Trained on CryoPPP keep-vs-junk labels via the features in features.py.
Start simple (RandomForest); it trains in seconds and updates instantly for the
human-in-the-loop active-learning demo. Swap in a small CNN later if time allows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class JunkClassifier:
    """Binary classifier: is a candidate JUNK (True) or a real particle (False)."""

    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, features: np.ndarray, is_junk: Sequence) -> "JunkClassifier":
        X = np.asarray(features, dtype=float)
        y = np.asarray(is_junk).astype(int)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_junk_proba(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        proba = self.model.predict_proba(X)
        # column for the "junk" (class 1)
        classes = list(self.model.classes_)
        col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, col]

    def predict_is_junk(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_junk_proba(features) >= threshold

    def update(self, features: np.ndarray, is_junk: Sequence) -> "JunkClassifier":
        """Active-learning refit incorporating new user-labeled corrections.

        For a hackathon the simplest robust approach is to append the new labels
        to the running training set and refit (RandomForest fits in seconds).
        active_learning.py owns the running buffer.
        """
        return self.fit(features, is_junk)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "JunkClassifier":
        import joblib

        obj = cls()
        obj.model = joblib.load(path)
        obj._fitted = True
        return obj
