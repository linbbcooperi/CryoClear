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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
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
    "sgd": {"label": "SGD log-loss — true online active learning (partial_fit)", "heldout": 0.243, "thr": 0.50},
    "cnn": {"label": "CNN — learned on raw 64px crops", "heldout": 0.248, "thr": 0.50},
}

# Preloaded datasets selectable in the UI. 10017 has CryoPPP ground truth (full metrics);
# 10005/10025 are different proteins scored by the 10017-trained classifier — a live
# cross-protein generalisation demo (no ground truth → counts only, no precision/recall).
DATASETS = {
    "10017": {"label": "β-galactosidase · EMPIAR-10017", "diameter": 108, "has_gt": True},
    "10005": {"label": "TRPV1 ion channel · EMPIAR-10005", "diameter": 180, "has_gt": False},
    "10025": {"label": "T20S proteasome · EMPIAR-10025", "diameter": 160, "has_gt": False},
    "10075": {"label": "EMPIAR-10075", "diameter": 108, "has_gt": True},
    "10345": {"label": "EMPIAR-10345", "diameter": 108, "has_gt": True},
    "10081": {"label": "EMPIAR-10081 · TMV", "diameter": 108, "has_gt": True},
    "10093": {"label": "EMPIAR-10093", "diameter": 108, "has_gt": True},
}


# The full picker menu. `status: ready` = wired + cacheable on this Blackwell pod;
# `status: planned` = registered behind the same interface but not installed (honest:
# crYOLO/MicrographCleaner are TensorFlow with no NVIDIA sm_120 wheels; CryoFSL/cryo-EMMAE
# are research code). A planned picker becomes selectable the moment its cache appears.
PICKERS = {
    "blob": {"label": "Blob LoG", "device": "CPU", "speed": "instant", "framework": "scikit-image",
             "why": "zero-deps over-picker — the guaranteed floor and the best junk-triage showcase",
             "status": "ready"},
    "topaz": {"label": "Topaz", "device": "GPU / CPU", "speed": "fast", "framework": "PyTorch",
              "why": "industry standard; pretrained resnet16_u64; the only strong picker that also runs CPU-only",
              "status": "ready"},
    "cryosegnet": {"label": "CryoSegNet", "device": "GPU", "speed": "slow", "framework": "PyTorch",
                   "why": "benchmark precision leader (SAM + attention-gated U-Net); cached offline",
                   "status": "ready"},
    "cryolo": {"label": "crYOLO", "device": "GPU", "speed": "fastest", "framework": "TensorFlow",
               "why": "highest FPS — but TensorFlow has no NVIDIA Blackwell sm_120 wheels (pluggable, not installed)",
               "status": "planned"},
    "cryofsl": {"label": "CryoFSL", "device": "GPU", "speed": "medium", "framework": "PyTorch + SAM2",
                "why": "few-shot: adapt from 5–20 of your own seed picks (research code; pluggable)",
                "status": "planned"},
    "cryoemmae": {"label": "cryo-EMMAE", "device": "GPU", "speed": "medium", "framework": "PyTorch",
                  "why": "self-supervised masked autoencoder — best generalization to unseen proteins (research code; pluggable)",
                  "status": "planned"},
}


def picker_menu(empiar: str) -> dict:
    """Full picker registry annotated with per-dataset readiness (has a cache here)."""
    out = {}
    for k, m in PICKERS.items():
        ready = m["status"] == "ready" and (k == "blob" or (cache_dir(empiar, k) / "index.json").exists())
        out[k] = {**m, "ready": ready}
    return out


def available_pickers(empiar: str) -> list[str]:
    """Pickers actually selectable for this dataset (ready + cached; blob always)."""
    return [k for k, m in picker_menu(empiar).items() if m["ready"]]


class State:
    def __init__(self, empiar: str):
        self.empiar = empiar
        self.picker = "blob"
        self.cache = cache_dir(empiar, self.picker)
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
        self.undo_stack: list[dict] = []       # each: {stem, prev:{idx: prior_label_or_None}}
        self.redo_stack: list[dict] = []
        self.learner = ActiveLearner(JunkClassifier(model_type="sgd"))  # true online partial_fit
        self.coldstart = True
        self.corrections = 0
        self.f1_history: list[float] = []
        self.seen: list[str] = []         # streamed micrograph order (cumulative metrics)
        self._seed_learner(coldstart=True)  # fast cold seed; only used in 'learn' mode

    def set_picker(self, picker: str) -> bool:
        """Switch the active picker by pointing at its precomputed cache; clears
        per-picker derived state (overrides/history/npz). Blob is always available."""
        cand = cache_dir(self.empiar, picker)
        if not (cand / "index.json").exists():
            return False
        self.picker = picker
        self.cache = cand
        self.index = json.loads((cand / "index.json").read_text())
        self.factor, self.radius = self.index["factor"], self.index["radius"]
        self._npz = {}
        self.overrides = {}
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.corrections = 0
        return True

    # ---- model / learner ----
    def _train_table(self):
        p = config.PROCESSED / self.empiar / "junk_train.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return d["X"].astype(float), d["y"].astype(int)

    def _seed_learner(self, coldstart: bool):
        tbl = self._train_table()
        self.learner = ActiveLearner(JunkClassifier(model_type="sgd"))  # true online partial_fit
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
        key = {"lgbm": "lgbm_scores", "rf": "rf_scores", "sgd": "sgd_scores",
               "cnn": "cnn_scores"}.get(self.clf_model)
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
            "picker": s.picker, "pickers": available_pickers(s.empiar), "picker_menu": picker_menu(s.empiar),
            "micrographs": s.index["micrographs"]}


@app.get("/api/datasets")
def api_datasets():
    """List preloaded datasets, whether each has a precomputed cache, and (from the
    cache) whether it carries CryoPPP ground truth (→ full precision/recall metrics)."""
    out = []
    for empiar, meta in DATASETS.items():
        idx_path = cache_dir(empiar) / "index.json"
        ready = idx_path.exists()
        has_gt = meta["has_gt"]
        if ready:
            try:
                mics = json.loads(idx_path.read_text())["micrographs"]
                has_gt = any(m.get("n_gt", 0) > 0 for m in mics)
            except Exception:
                pass
        out.append({"empiar": empiar, "ready": ready, **meta, "has_gt": has_gt})
    return {"datasets": out}


@app.get("/api/img/{empiar}/{stem}.png")
def api_img(empiar: str, stem: str):
    # micrograph PNG is immutable per (empiar, stem) → let the browser hard-cache it so
    # navigation between already-seen frames is instant (zero-lag).
    return FileResponse(get_state(empiar).cache / "img" / f"{stem}.png",
                        headers={"Cache-Control": "public, max-age=86400"})


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
        s.undo_stack.clear()
        s.redo_stack.clear()
    return {"mode": s.mode}


@app.post("/api/picker")
def api_picker(payload: dict):
    s = get_state(payload.get("empiar"))
    ok = s.set_picker(payload.get("picker", "blob"))
    return {"ok": ok, "picker": s.picker, "n_micrographs": len(s.index["micrographs"])}


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
    else:                   # model mode: manual per-particle overrides (instant, undoable)
        ov = s.overrides.setdefault(stem, {})
        prev = {i: ov.get(i) for i in (*dump, *keep)}   # prior label (None = unset) for undo
        for i in dump:
            ov[i] = 1
        for i in keep:
            ov[i] = 0
        if prev:
            s.undo_stack.append({"stem": stem, "prev": prev})
            s.redo_stack.clear()
    s.corrections += len(dump) + len(keep)
    junk = s.junk_mask(stem)
    d = s.npz(stem)
    if (d["true_junk"] != -1).any():
        jr = metrics.junk_rejection_metrics(junk, d["true_junk"] == 1)
        s.f1_history.append(jr["junk_f1"])
    return {"stem": stem, "score": s.scores(stem).tolist(),
            "junk": junk.astype(int).tolist(), "corrections": s.corrections,
            "f1_history": s.f1_history, "picking_after": s.picking(stem, ~junk),
            "can_undo": bool(s.undo_stack), "can_redo": bool(s.redo_stack)}


def _swap_overrides(s: "State", rec: dict) -> dict:
    """Apply rec['prev'] to the overrides for rec['stem']; return the inverse record."""
    stem = rec["stem"]
    ov = s.overrides.setdefault(stem, {})
    inv = {}
    for i, val in rec["prev"].items():
        inv[i] = ov.get(i)
        if val is None:
            ov.pop(i, None)
        else:
            ov[i] = val
    return {"stem": stem, "prev": inv}


def _undo_redo_result(s: "State", stem: str) -> dict:
    junk = s.junk_mask(stem)
    return {"stem": stem, "score": s.scores(stem).tolist(),
            "junk": junk.astype(int).tolist(), "picking_after": s.picking(stem, ~junk),
            "can_undo": bool(s.undo_stack), "can_redo": bool(s.redo_stack)}


@app.post("/api/undo")
def api_undo(payload: dict):
    s = get_state(payload.get("empiar"))
    if not s.undo_stack:
        return {"ok": False, "can_undo": False, "can_redo": bool(s.redo_stack)}
    rec = s.undo_stack.pop()
    s.redo_stack.append(_swap_overrides(s, rec))
    return {"ok": True, **_undo_redo_result(s, rec["stem"])}


@app.post("/api/redo")
def api_redo(payload: dict):
    s = get_state(payload.get("empiar"))
    if not s.redo_stack:
        return {"ok": False, "can_undo": bool(s.undo_stack), "can_redo": False}
    rec = s.redo_stack.pop()
    s.undo_stack.append(_swap_overrides(s, rec))
    return {"ok": True, **_undo_redo_result(s, rec["stem"])}


@app.post("/api/threshold")
def api_threshold(payload: dict):
    s = get_state(payload.get("empiar"))
    s.threshold = float(payload["threshold"])
    return {"threshold": s.threshold}


@app.post("/api/upload")
async def api_upload(request: Request, empiar: str = config.DEMO_EMPIAR_ID,
                     filename: str = "upload.mrc"):
    """Bring-your-own micrograph: accept a raw .mrc body, pick + score it with the
    current dataset's trained classifiers, and add it to the live micrograph list.
    No ground truth, so picking precision/recall read as N/A (counts still shown)."""
    import re

    import matplotlib.image as mpimg

    from cryoclear import features as feat, io_mrc, picker

    s = get_state(empiar)
    data = await request.body()
    if not data:
        return JSONResponse({"ok": False, "error": "empty upload"}, status_code=400)
    stem = "up_" + (re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:40] or "mrc")
    updir = config.RAW / empiar / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    mrc = updir / f"{stem}.mrc"
    mrc.write_bytes(data)
    try:
        imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mrc))
        img = io_mrc.load_for_pipeline(mrc, factor=s.factor)
    except Exception as e:                      # not a valid MRC
        return JSONResponse({"ok": False, "error": f"unreadable MRC: {e}"}, status_code=400)

    (s.cache / "img").mkdir(parents=True, exist_ok=True)
    (s.cache / "data").mkdir(parents=True, exist_ok=True)
    mpimg.imsave(s.cache / "img" / f"{stem}.png", img, cmap="gray", vmin=0, vmax=255)
    box = s.index.get("box", config.DEMO_PARTICLE_DIAMETER_PX)
    pred_disp = picker.pick(img, backend="blob")
    pred_full = pred_disp * float(s.factor)
    feats = feat.extract_features(imgf, pred_full, box=box)
    payload = dict(
        pred_disp=pred_disp.astype(np.float32), pred_full=pred_full.astype(np.float32),
        scores=np.zeros(len(pred_full), np.float32),
        true_junk=np.full(len(pred_full), -1, np.int8), feats=feats.astype(np.float32))
    for mt in ("rf", "lgbm"):
        mp = config.PROCESSED / empiar / f"junk_{mt}.joblib"
        if mp.exists() and len(feats):
            try:
                payload[f"{mt}_scores"] = JunkClassifier.load(mp).predict_junk_proba(feats).astype(np.float32)
            except Exception:
                pass
    np.savez_compressed(s.cache / "data" / f"{stem}.npz", **payload)

    s.index["micrographs"] = [m for m in s.index["micrographs"] if m["stem"] != stem]
    s.index["micrographs"].append({"stem": stem, "n_picks": int(len(pred_full)),
                                   "h": int(img.shape[0]), "w": int(img.shape[1]),
                                   "n_gt": 0, "uploaded": True})
    (s.cache / "index.json").write_text(json.dumps(s.index, indent=2))
    s._npz.pop(stem, None)
    return {"ok": True, "stem": stem, "n_picks": int(len(pred_full))}


@app.post("/api/reset")
def api_reset(payload: dict):
    s = get_state(payload.get("empiar"))
    s.overrides = {}
    s.undo_stack.clear()
    s.redo_stack.clear()
    s.corrections = 0
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
    """Detailed 2-page PDF scorecard: dataset + KPIs, precision/recall/F1 vs ground
    truth, held-out classifier comparison, a sample overlay, and the score histogram."""
    import io
    from datetime import datetime, timezone

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    s = get_state(empiar)
    KEEP, JUNK, MUT = "#29c393", "#e5484d", "#8a93a3"
    mics = s.index["micrographs"]

    tot_c = tot_k = 0
    pf_b, pf_a, rc_b, rc_a, pr_b, pr_a = [], [], [], [], [], []
    sj, sk = [], []                       # scores split by held-out true label
    best_stem, best_gt = None, -1
    for m in mics:
        stem = m["stem"]
        junk = s.junk_mask(stem)
        d = s.npz(stem)
        tot_c += len(junk)
        tot_k += int((~junk).sum())
        raw, aft = s.picking(stem, np.ones(len(junk), bool)), s.picking(stem, ~junk)
        pf_b.append(raw["f1"]); pf_a.append(aft["f1"])
        rc_b.append(raw["recall"]); rc_a.append(aft["recall"])
        pr_b.append(raw["precision"]); pr_a.append(aft["precision"])
        tj = d["true_junk"]
        if (tj != -1).any():
            sc = s.scores(stem)
            sj.extend(sc[tj == 1].tolist()); sk.extend(sc[tj == 0].tolist())
        if int(m.get("n_gt", 0)) > best_gt:
            best_gt, best_stem = int(m.get("n_gt", 0)), stem

    avg = lambda a: float(np.mean(a)) if a else 0.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    junk_pct = 100 * (1 - tot_k / max(tot_c, 1))

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---------------- page 1: summary ----------------
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        fig.text(0.07, 0.955, "CryoClear", fontsize=27, weight="bold", color="#111")
        fig.text(0.07, 0.930, "Particle-picking + junk-triage report", fontsize=13, color="#444")
        fig.text(0.93, 0.957, ts, fontsize=9, color="#999", ha="right")
        fig.text(0.07, 0.905, f"EMPIAR-{empiar}    ·    picker: blob LoG    ·    classifier: "
                 f"{s.clf_model.upper()}    ·    threshold {s.threshold:.2f}", fontsize=10.5, color="#222")

        for i, (k, v) in enumerate([("Micrographs", f"{len(mics)}"), ("Candidates", f"{tot_c:,}"),
                                    ("Kept", f"{tot_k:,}"), ("Junk removed", f"{junk_pct:.1f}%")]):
            ax = fig.add_axes([0.07 + i * 0.225, 0.80, 0.205, 0.075]); ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f3f5f8", edgecolor="#dde3ea", lw=1))
            ax.text(0.5, 0.60, v, ha="center", va="center", fontsize=16, weight="bold", color="#111")
            ax.text(0.5, 0.22, k, ha="center", va="center", fontsize=9, color="#666")

        ax1 = fig.add_axes([0.10, 0.50, 0.37, 0.21])
        xs = np.arange(3)
        ax1.bar(xs - 0.2, [avg(pr_b), avg(rc_b), avg(pf_b)], 0.4, label="raw picks", color=MUT)
        ax1.bar(xs + 0.2, [avg(pr_a), avg(rc_a), avg(pf_a)], 0.4, label="after triage", color=KEEP)
        ax1.set_xticks(xs); ax1.set_xticklabels(["precision", "recall", "F1"], fontsize=9)
        ax1.set_ylim(0, 1); ax1.legend(fontsize=8, loc="upper left")
        ax1.set_title("Picking vs CryoPPP ground truth (in-sample mean)", fontsize=10)
        ax1.tick_params(labelsize=8)

        axT = fig.add_axes([0.55, 0.50, 0.40, 0.21]); axT.axis("off")
        axT.set_title("Held-out picking F1   (raw blob = 0.227)", fontsize=10, loc="left")
        rows = [["classifier", "@ 0.5", "@ calib."]]
        for k in ("rf", "lgbm", "cnn"):
            o = CLF_OPTIONS[k]
            at05 = {"rf": "0.000", "lgbm": "0.228", "cnn": "—"}[k]
            rows.append([o["label"].split(" —")[0], at05, f"{o['heldout']:.3f}@{o['thr']:.2f}"])
        tb = axT.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.46, 0.24, 0.30])
        tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.45)
        for j in range(3):
            tb[0, j].set_facecolor("#2f6feb"); tb[0, j].set_text_props(color="white", weight="bold")

        note = (
            f"Method.  The blob LoG picker over-picks (~{junk_pct:.0f}% of candidates flagged junk). A "
            f"{s.clf_model.upper()} classifier scores each candidate on 23 intensity-normalised features "
            "(radial profile, matched-filter NCC, structure-tensor edge coherence, distribution shape, "
            "sharpness); the scientist corrects it on an interactive canvas and the model can refit live. "
            "Kept coordinates export to RELION/cryoSPARC as .star/.box.\n\n"
            "Honesty.  The bars above are in-sample (optimistic); the table is micrograph-level held-out "
            "(the generalising metric). Per-model threshold calibration is essential — a vanilla "
            "RandomForest at 0.5 over-rejects and collapses to F1 0.000. The blob picker over-picks "
            "background that resembles the small, low-contrast particles, so even the best classifier only "
            "lifts picking F1 from 0.227 to 0.248; a stronger picker or a distinct-junk protein is the real "
            "lever. Reproduce with eval/compare_classifiers.py.")
        fig.text(0.07, 0.43, note, fontsize=9.5, va="top", linespacing=1.5,
                 wrap=True, color="#222")
        fig.text(0.07, 0.05, "CryoClear · open, real-time cryo-EM junk-triage copilot · MIT",
                 fontsize=8, color="#aaa")
        pdf.savefig(fig); plt.close(fig)

        # ---------------- page 2: visuals ----------------
        fig2 = plt.figure(figsize=(8.5, 11)); fig2.patch.set_facecolor("white")
        fig2.text(0.07, 0.955, "Sample micrograph & score separation", fontsize=15, weight="bold")
        if best_stem is not None:
            d = s.npz(best_stem); junk = s.junk_mask(best_stem); pdsp = d["pred_disp"]
            png = s.cache / "img" / f"{best_stem}.png"
            axA = fig2.add_axes([0.07, 0.46, 0.86, 0.44]); axA.axis("off")
            if png.exists():
                axA.imshow(mpimg.imread(png), cmap="gray")
            kxy, jxy = pdsp[~junk], pdsp[junk]
            if len(kxy):
                axA.scatter(kxy[:, 0], kxy[:, 1], s=7, facecolors="none", edgecolors=KEEP, linewidths=0.5)
            if len(jxy):
                axA.scatter(jxy[:, 0], jxy[:, 1], s=7, facecolors="none", edgecolors=JUNK, linewidths=0.5)
            axA.set_title(f"{best_stem}   —   green keep ({int((~junk).sum())})  ·  "
                          f"red junk ({int(junk.sum())})", fontsize=9.5)

        axB = fig2.add_axes([0.11, 0.09, 0.80, 0.28])
        if sj or sk:
            bins = np.linspace(0, 1, 30)
            axB.hist(sk, bins=bins, alpha=0.65, color=KEEP, label=f"true particle (n={len(sk):,})")
            axB.hist(sj, bins=bins, alpha=0.65, color=JUNK, label=f"true junk (n={len(sj):,})")
            axB.axvline(s.threshold, color="#111", ls="--", lw=1.2, label=f"threshold {s.threshold:.2f}")
            axB.set_xlabel("junk probability", fontsize=9); axB.set_ylabel("candidates", fontsize=9)
            axB.legend(fontsize=8); axB.tick_params(labelsize=8)
            axB.set_title("Classifier score distribution (CryoPPP-labelled candidates)", fontsize=10)
        else:
            axB.axis("off"); axB.text(0.5, 0.5, "no ground-truth labels for this dataset",
                                      ha="center", color="#888")
        pdf.savefig(fig2); plt.close(fig2)

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
