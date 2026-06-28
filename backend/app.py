"""CryoClear real-time backend (FastAPI).

Serves the precomputed per-micrograph cache + the React canvas frontend, and
handles human-in-the-loop corrections, live metrics, 2D classification, and a
WebSocket stream. Reuses all of `cryoclear`.

  uv run uvicorn backend.app:app --host 0.0.0.0 --port 8501
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from cryoclear import config, metrics  # noqa: E402
from cryoclear.active_learning import ActiveLearner  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402
from backend.precompute import cache_dir, precompute  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app = FastAPI(title="CryoClear")


class State:
    def __init__(self, empiar: str):
        self.empiar = empiar
        self.cache = cache_dir(empiar)
        idx_path = self.cache / "index.json"
        if not idx_path.exists():
            precompute(empiar)
        self.index = json.loads(idx_path.read_text())
        self.factor = self.index["factor"]
        self.radius = self.index["radius"]
        self.threshold = 0.5
        self._npz: dict[str, dict] = {}
        self.learner = ActiveLearner(JunkClassifier())
        self.coldstart = False
        self.corrections = 0
        self.f1_history: list[float] = []
        self.seen: list[str] = []         # streamed micrograph order (cumulative metrics)
        self._seed_learner(coldstart=False)

    # ---- model / learner ----
    def _train_table(self):
        p = config.PROCESSED / self.empiar / "junk_train.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return d["X"].astype(float), d["y"].astype(int)

    def _seed_learner(self, coldstart: bool):
        tbl = self._train_table()
        self.learner = ActiveLearner(JunkClassifier())
        self.coldstart = coldstart
        self.corrections = 0
        self.f1_history = []
        if tbl is None:
            return
        X, y = tbl
        rng = np.random.default_rng(0)
        if coldstart:                      # junk-poor seed → model misses junk, climbs as taught
            keep, junk = np.where(y == 0)[0], np.where(y == 1)[0]
            idx = np.concatenate([rng.choice(keep, min(28, len(keep)), replace=False),
                                  rng.choice(junk, min(3, len(junk)), replace=False)])
        else:                              # full-data seed → strong model
            idx = rng.permutation(len(y))[:min(40000, len(y))]
        self.learner.seed(X[idx], y[idx])

    def npz(self, stem: str) -> dict:
        if stem not in self._npz:
            self._npz[stem] = dict(np.load(self.cache / "data" / f"{stem}.npz"))
        return self._npz[stem]

    def scores(self, stem: str) -> np.ndarray:
        feats = self.npz(stem)["feats"]
        if len(feats) == 0:
            return np.zeros(0)
        return np.asarray(self.learner.clf.predict_junk_proba(feats))

    # ---- metrics from cached true labels ----
    def picking(self, stem: str, kept_mask: np.ndarray):
        d = self.npz(stem)
        tj = d["true_junk"]
        info = next(m for m in self.index["micrographs"] if m["stem"] == stem)
        n_gt = info["n_gt"]
        real = (tj == 0)
        tp = int((real & kept_mask).sum())
        fp = int((~real & kept_mask & (tj != -1)).sum())
        fn = max(n_gt - tp, 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


_STATES: dict[str, State] = {}


def get_state(empiar: str | None = None) -> State:
    empiar = empiar or config.DEMO_EMPIAR_ID
    if empiar not in _STATES:
        _STATES[empiar] = State(empiar)
    return _STATES[empiar]


# ---------------------------------------------------------------- API
@app.get("/api/state")
def api_state(empiar: str | None = None):
    s = get_state(empiar)
    return {"empiar": s.empiar, "factor": s.factor, "radius": s.radius,
            "threshold": s.threshold, "coldstart": s.coldstart,
            "corrections": s.corrections, "f1_history": s.f1_history,
            "micrographs": s.index["micrographs"]}


@app.get("/api/img/{empiar}/{stem}.png")
def api_img(empiar: str, stem: str):
    return FileResponse(get_state(empiar).cache / "img" / f"{stem}.png")


@app.get("/api/picks/{empiar}/{stem}")
def api_picks(empiar: str, stem: str):
    s = get_state(empiar)
    d = s.npz(stem)
    pd = d["pred_disp"]
    sc = s.scores(stem)
    return {"stem": stem, "x": pd[:, 0].tolist(), "y": pd[:, 1].tolist(),
            "score": sc.tolist(), "true_junk": d["true_junk"].tolist()}


@app.post("/api/correct")
async def api_correct(payload: dict):
    s = get_state(payload.get("empiar"))
    stem = payload["stem"]
    feats = s.npz(stem)["feats"]
    dump = [i for i in payload.get("dump_idx", []) if 0 <= i < len(feats)]
    keep = [i for i in payload.get("keep_idx", []) if 0 <= i < len(feats)]
    if dump:
        s.learner.add_corrections(feats[dump], np.ones(len(dump), int))
    if keep:
        s.learner.add_corrections(feats[keep], np.zeros(len(keep), int))
    s.corrections += len(dump) + len(keep)
    sc = s.scores(stem)
    kept = sc < s.threshold
    if (s.npz(stem)["true_junk"] != -1).any():
        jr = metrics.junk_rejection_metrics(sc >= s.threshold, s.npz(stem)["true_junk"] == 1)
        s.f1_history.append(jr["junk_f1"])
    return {"stem": stem, "score": sc.tolist(), "corrections": s.corrections,
            "f1_history": s.f1_history, "picking_after": s.picking(stem, kept)}


@app.post("/api/threshold")
def api_threshold(payload: dict):
    s = get_state(payload.get("empiar"))
    s.threshold = float(payload["threshold"])
    return {"threshold": s.threshold}


@app.post("/api/reset")
def api_reset(payload: dict):
    s = get_state(payload.get("empiar"))
    s._seed_learner(coldstart=bool(payload.get("coldstart", False)))
    return {"ok": True, "coldstart": s.coldstart}


@app.get("/api/metrics/{empiar}/{stem}")
def api_metrics(empiar: str, stem: str):
    s = get_state(empiar)
    sc = s.scores(stem)
    junk = sc >= s.threshold
    d = s.npz(stem)
    n = len(sc)
    out = {"stem": stem, "n_candidates": n, "n_kept": int((~junk).sum()),
           "junk_pct": float(100 * junk.mean()) if n else 0.0,
           "picking_before": s.picking(stem, np.ones(n, bool)),
           "picking_after": s.picking(stem, ~junk)}
    if (d["true_junk"] != -1).any():
        out["junk_rejection"] = metrics.junk_rejection_metrics(junk, d["true_junk"] == 1)
    return out


@app.post("/api/classify2d")
def api_classify2d(payload: dict):
    s = get_state(payload.get("empiar"))
    from cryoclear import class2d, io_mrc
    max_particles = int(payload.get("max_particles", 400))
    crops_all, n = [], 0
    for info in s.index["micrographs"]:
        stem = info["stem"]
        mic = config.RAW / s.empiar / "micrographs" / f"{stem}.mrc"
        d = s.npz(stem)
        kept = d["pred_full"][s.scores(stem) < s.threshold]
        imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
        crops_all.append(class2d.extract_particles(imgf, kept, box=160, out_size=64))
        n += len(crops_all[-1])
        if n >= max_particles:
            break
    stack = np.concatenate(crops_all)[:max_particles] if crops_all else np.zeros((0, 64, 64))
    avgs, _lab, counts = class2d.classify_2d(stack, n_classes=int(payload.get("n_classes", 6)))
    keep = [i for i in range(len(avgs)) if counts[i] >= 8]
    return {"n_particles": int(len(stack)),
            "classes": [{"png": _png_b64(avgs[i]), "count": int(counts[i])} for i in keep]}


def _png_b64(arr: np.ndarray) -> str:
    import matplotlib.image as mpimg
    a = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    buf = io.BytesIO()
    mpimg.imsave(buf, a, cmap="gray", format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    s = get_state()
    speed = 1.0
    try:
        msg = await ws.receive_json()
        speed = float(msg.get("speed", 1.0))
        mics = [m["stem"] for m in s.index["micrographs"]]
        i = 0
        while True:
            stem = mics[i % len(mics)]
            s.seen.append(stem)
            sc = s.scores(stem)
            junk = sc >= s.threshold
            await ws.send_json({"stem": stem, "i": i, "n_kept": int((~junk).sum()),
                                "n_candidates": int(len(sc)),
                                "junk_pct": float(100 * junk.mean()) if len(sc) else 0.0})
            i += 1
            await asyncio.sleep(speed)
    except WebSocketDisconnect:
        return


# ---------------------------------------------------------------- frontend
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if (FRONTEND).exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
