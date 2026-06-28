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

# Junk-classifier options. `heldout` = best achievable held-out picking F1 (raw blob
# picks are 0.227); `thr` = the per-model calibrated default threshold from that sweep
# (eval/compare_classifiers.py). RF must run at a high threshold or it over-rejects
# (F1 collapses to 0.0 at 0.5); LightGBM is robust and is the default.
CLF_OPTIONS = {
    "lgbm": {"label": "LightGBM — boosted trees (robust, default)", "heldout": 0.248, "thr": 0.60},
    "rf": {"label": "RandomForest — needs high threshold or over-rejects", "heldout": 0.248, "thr": 0.85},
    "cnn": {"label": "CNN — learned on raw 64px crops", "heldout": 0.248, "thr": 0.50},
}


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
        self.clf_model = "lgbm"           # "lgbm" | "rf" | "cnn" (model mode)
        self.threshold = CLF_OPTIONS[self.clf_model]["thr"]   # per-model calibrated default
        self.mode = "model"               # "model"=precomputed scores | "learn"=live cold-start AL
        self._npz: dict[str, dict] = {}
        self.overrides: dict[str, dict] = {}   # {stem: {idx: label}} manual HITL corrections
        self.learner = ActiveLearner(JunkClassifier())
        self.coldstart = True
        self.corrections = 0
        self.f1_history: list[float] = []
        self.seen: list[str] = []         # streamed micrograph order (cumulative metrics)
        self._seed_learner(coldstart=True)  # fast cold seed; only used in 'learn' mode

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
        d = self.npz(stem)
        if self.mode == "learn":
            feats = d["feats"]
            return (np.asarray(self.learner.clf.predict_junk_proba(feats))
                    if len(feats) else np.zeros(0))
        key = {"lgbm": "lgbm_scores", "rf": "rf_scores", "cnn": "cnn_scores"}.get(self.clf_model)
        if key and key in d:
            return d[key]
        return d["lgbm_scores"] if "lgbm_scores" in d else d["scores"]

    def junk_mask(self, stem: str) -> np.ndarray:
        junk = self.scores(stem) >= self.threshold
        for i, lab in self.overrides.get(stem, {}).items():
            if 0 <= int(i) < len(junk):
                junk[int(i)] = bool(lab)
        return junk

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
            "clf_model": s.clf_model, "mode": s.mode,
            "clf_options": CLF_OPTIONS,
            "micrographs": s.index["micrographs"]}


@app.get("/api/img/{empiar}/{stem}.png")
def api_img(empiar: str, stem: str):
    return FileResponse(get_state(empiar).cache / "img" / f"{stem}.png")


@app.get("/api/picks/{empiar}/{stem}")
def api_picks(empiar: str, stem: str):
    s = get_state(empiar)
    d = s.npz(stem)
    pd = d["pred_disp"]
    return {"stem": stem, "x": pd[:, 0].tolist(), "y": pd[:, 1].tolist(),
            "score": s.scores(stem).tolist(), "junk": s.junk_mask(stem).astype(int).tolist(),
            "true_junk": d["true_junk"].tolist()}


@app.post("/api/mode")
def api_mode(payload: dict):
    s = get_state(payload.get("empiar"))
    s.mode = payload.get("mode", "model")
    if s.mode == "learn":
        s._seed_learner(coldstart=True)
        s.overrides = {}
    return {"mode": s.mode}


@app.post("/api/clf_model")
def api_clf_model(payload: dict):
    s = get_state(payload.get("empiar"))
    s.clf_model = payload.get("clf_model", "lgbm")
    # snap the threshold to this model's calibrated default so it doesn't over/under-reject
    s.threshold = CLF_OPTIONS.get(s.clf_model, {}).get("thr", s.threshold)
    return {"clf_model": s.clf_model, "threshold": s.threshold}


@app.post("/api/correct")
async def api_correct(payload: dict):
    s = get_state(payload.get("empiar"))
    stem = payload["stem"]
    feats = s.npz(stem)["feats"]
    dump = [int(i) for i in payload.get("dump_idx", []) if 0 <= int(i) < len(feats)]
    keep = [int(i) for i in payload.get("keep_idx", []) if 0 <= int(i) < len(feats)]
    if s.mode == "learn":   # corrections retrain the live model (the climb demo)
        if dump:
            s.learner.add_corrections(feats[dump], np.ones(len(dump), int))
        if keep:
            s.learner.add_corrections(feats[keep], np.zeros(len(keep), int))
    else:                   # model mode: manual per-particle overrides (instant)
        ov = s.overrides.setdefault(stem, {})
        for i in dump:
            ov[i] = 1
        for i in keep:
            ov[i] = 0
    s.corrections += len(dump) + len(keep)
    junk = s.junk_mask(stem)
    d = s.npz(stem)
    if (d["true_junk"] != -1).any():
        jr = metrics.junk_rejection_metrics(junk, d["true_junk"] == 1)
        s.f1_history.append(jr["junk_f1"])
    return {"stem": stem, "score": s.scores(stem).tolist(),
            "junk": junk.astype(int).tolist(), "corrections": s.corrections,
            "f1_history": s.f1_history, "picking_after": s.picking(stem, ~junk)}


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
    junk = s.junk_mask(stem)
    d = s.npz(stem)
    n = len(junk)
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
        kept = d["pred_full"][~s.junk_mask(stem)]
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
            junk = s.junk_mask(stem)
            await ws.send_json({"stem": stem, "i": i, "n_kept": int((~junk).sum()),
                                "n_candidates": int(len(junk)),
                                "junk_pct": float(100 * junk.mean()) if len(junk) else 0.0})
            i += 1
            await asyncio.sleep(speed)
    except WebSocketDisconnect:
        return


# ---------------------------------------------------------------- frontend
@app.get("/api/export/coords/{empiar}/{stem}")
def api_export_coords(empiar: str, stem: str, fmt: str = "star"):
    """Kept-particle coordinates (full-res) as RELION .star or EMAN .box — the
    product output that feeds RELION/cryoSPARC."""
    s = get_state(empiar)
    kept = s.npz(stem)["pred_full"][~s.junk_mask(stem)]
    box = config.DEMO_PARTICLE_DIAMETER_PX
    if fmt == "box":
        body = "".join(f"{x - box/2:.1f}\t{y - box/2:.1f}\t{box}\t{box}\n" for x, y in kept)
        fname = f"{stem}_kept.box"
    else:
        body = ("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
                + "".join(f"{x:.2f}\t{y:.2f}\n" for x, y in kept))
        fname = f"{stem}_kept.star"
    return Response(body, media_type="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.get("/api/export/report/{empiar}")
def api_export_report(empiar: str):
    """One-page PDF scorecard: dataset, picker, classifier, and honest aggregate metrics."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    s = get_state(empiar)
    tot_c = tot_k = 0
    pb, pa = [], []
    for m in s.index["micrographs"]:
        stem = m["stem"]
        junk = s.junk_mask(stem)
        tot_c += len(junk)
        tot_k += int((~junk).sum())
        pb.append(s.picking(stem, np.ones(len(junk), bool))["f1"])
        pa.append(s.picking(stem, ~junk)["f1"])
    pb_m, pa_m = float(np.mean(pb)), float(np.mean(pa))

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.93, "CryoClear — junk-triage report", fontsize=20, weight="bold")
        fig.text(0.08, 0.90, f"EMPIAR-{empiar} · picker: blob · classifier: {s.clf_model} "
                 f"· threshold {s.threshold:.2f}", fontsize=11, color="#555")
        lines = [
            f"Micrographs: {len(s.index['micrographs'])}",
            f"Candidates picked: {tot_c:,}",
            f"Particles kept (after junk triage): {tot_k:,}  ({100*tot_k/max(tot_c,1):.1f}%)",
            f"Junk removed: {100*(1-tot_k/max(tot_c,1)):.1f}%",
            "",
            f"Picking F1 vs CryoPPP ground truth (mean over micrographs):",
            f"    raw picks:        {pb_m:.3f}",
            f"    after junk triage: {pa_m:.3f}",
            "",
            "Note: in-sample numbers are optimistic; held-out generalization is the honest",
            "metric (see eval/heldout_eval.py and eval/compare_classifiers.py).",
        ]
        fig.text(0.08, 0.82, "\n".join(lines), fontsize=12, va="top", family="monospace")
        ax = fig.add_axes([0.1, 0.32, 0.5, 0.28])
        ax.bar(["raw", "after triage"], [pb_m, pa_m], color=["#888", "#29c393"])
        ax.set_ylim(0, 1)
        ax.set_title("Picking F1", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=cryoclear_report.pdf"})


@app.on_event("startup")
def _prewarm():
    try:
        get_state()   # build the default-dataset State so the first request is instant
    except Exception:
        pass


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if (FRONTEND).exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
