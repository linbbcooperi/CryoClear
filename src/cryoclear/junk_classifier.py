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

    Three backends, same API:
      - ``rf``   scikit-learn RandomForest — zero-tuning, robust.
      - ``lgbm`` LightGBM gradient-boosted trees — best honest tabular accuracy (default);
                 captures the nonlinear ice/carbon/aggregate signatures.
      - ``sgd``  StandardScaler + SGDClassifier(log_loss) — purest active learning: true
                 incremental ``partial_fit`` updates (ms) + a linear, interpretable boundary.
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
        elif model_type == "sgd":
            from sklearn.linear_model import SGDClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            # log_loss → calibrated probabilities; scaling is essential for SGD since the
            # raw features span very different magnitudes. partial_fit (see update) gives
            # true online learning for the human-in-the-loop loop.
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("sgd", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                                      class_weight="balanced", max_iter=1000, tol=1e-3,
                                      random_state=random_state)),
            ])
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
        """Active-learning update from new user-labeled corrections.

        For ``sgd`` this is a true incremental ``partial_fit`` (no full retrain) once the
        model has been seeded — the smoothest "watch it learn" online update. For rf/lgbm
        we append to the running set and refit (active_learning.py owns the buffer); both
        fit in well under a second.
        """
        y = np.asarray(is_junk).astype(int)
        # partial_fit needs both classes in the batch when class_weight="balanced"
        # (it recomputes weights per call); fall back to a full fit otherwise.
        if self.model_type == "sgd" and self._fitted and len(np.unique(y)) >= 2:
            X = np.asarray(features, dtype=float)
            scaler = self.model.named_steps["scaler"]
            self.model.named_steps["sgd"].partial_fit(scaler.transform(X), y)
            return self
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
