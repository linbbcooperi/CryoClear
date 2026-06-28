"""A small CNN junk classifier on raw particle crops (the GPU-worthy upgrade).

The 8 hand-crafted features don't separate real particles from junk across
micrographs (they overfit in-sample). A tiny CNN on the 64x64 crops learns the
discriminative texture/shape and generalizes far better. torch is imported lazily
so the rest of cryoclear still runs without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _build(torch_nn):
    nn = torch_nn

    class JunkCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                   # 32
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                   # 16
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            return self.head(self.features(x).flatten(1)).squeeze(1)

    return JunkCNN


class CNNJunkClassifier:
    """Binary CNN: P(junk) for a stack of 64x64 crops. Mirrors JunkClassifier API."""

    def __init__(self, device: str = "cuda:0"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = _build(torch.nn)().to(self.device)

    def fit(self, crops: np.ndarray, is_junk, epochs: int = 8, batch: int = 512,
            lr: float = 1e-3, val=None, log=print) -> "CNNJunkClassifier":
        torch = self.torch
        X = torch.tensor(crops[:, None, :, :], dtype=torch.float32)
        y = torch.tensor(np.asarray(is_junk, dtype=np.float32))
        n = len(X)
        pos = float(y.mean())
        pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                xb = X[idx].to(self.device)
                yb = y[idx].to(self.device)
                opt.zero_grad()
                loss = lossf(self.model(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            msg = f"  epoch {ep + 1}/{epochs} loss={tot / n:.4f}"
            if val is not None:
                from cryoclear import metrics
                p = self.predict_is_junk(val[0])
                jr = metrics.junk_rejection_metrics(p, val[1].astype(bool))
                msg += f"  val junk-F1={jr['junk_f1']:.3f}"
            log(msg)
        return self

    def predict_junk_proba(self, crops: np.ndarray, batch: int = 1024) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        out = []
        X = torch.tensor(crops[:, None, :, :], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = X[i:i + batch].to(self.device)
                out.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_is_junk(self, crops: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_junk_proba(crops) >= threshold

    def save(self, path: str | Path) -> None:
        self.torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda:0") -> "CNNJunkClassifier":
        obj = cls(device=device)
        obj.model.load_state_dict(obj.torch.load(path, map_location=obj.device))
        obj.model.eval()
        return obj
