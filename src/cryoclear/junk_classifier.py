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
    """Binary junk classifier (JUNK=1 / particle=0) over per-candidate features.

    Two backends: ``rf`` (scikit-learn RandomForest) and ``lgbm`` (LightGBM gradient-
    boosted trees — best tabular accuracy, captures the nonlinear ice/carbon/aggregate
    signatures, refits in well under a second). Same API for both.
    """

    def __init__(self, model_type: str = "rf", n_estimators: int = 400,
                 random_state: int = 0):
        self.model_type = model_type
        if model_type == "lgbm":
            from lightgbm import LGBMClassifier

            # more trees + slow LR + explicit regularisation (min_child_samples,
            # reg_lambda, feature/row subsampling) so it generalises across
            # micrographs rather than memorising them.
            self.model = LGBMClassifier(
                n_estimators=600, num_leaves=63, learning_rate=0.03,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
                n_jobs=-1, random_state=random_state, verbosity=-1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            # min_samples_leaf + sqrt features + balanced_subsample temper the
            # in-sample memorisation that makes a vanilla RF report a fake ~1.0.
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=random_state,
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
