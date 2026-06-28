"""CryoClear — interactive picking + junk-triage copilot (Bindu).

Run:  streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

Wires the real pipeline behind a clean UI:
  * load a real EMPIAR micrograph (io_mrc) or a synthetic one (offline fallback)
  * pick particles (cryosegnet cached .star, or the blob placeholder)
  * color each candidate green=keep / red=junk via the trained JunkClassifier
  * live scoreboard (candidates / kept / junk% / micrographs-min) + picking P/R/F1
  * click a box to correct it -> ActiveLearner refits -> colors + metric update (M2)

Coordinate spaces: the display image is downsampled by `factor`; CryoSegNet picks
are full-res and divided by `factor` for display, scaled back up for scoring vs GT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from cryoclear import (  # noqa: E402
    config, coords, features, io_mrc, metrics, picker,
)
from cryoclear.active_learning import ActiveLearner  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402

# optional: pixel-accurate click-to-reject (pip install streamlit-image-coordinates)
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as click_image
    _HAVE_CLICK = True
except Exception:
    _HAVE_CLICK = False

st.set_page_config(page_title="CryoClear", page_icon="🧊", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      h1 {font-weight: 800; letter-spacing: -0.02em;}
      .subtitle {color: #8aa0b2; margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 0.95rem;}
      div[data-testid="stMetric"] {
        background: #11161c; border: 1px solid #1e2730; border-radius: 14px;
        padding: 14px 16px;}
      div[data-testid="stMetricValue"] {font-size: 1.7rem;}
      .legend span {font-size: 0.85rem; margin-right: 14px;}
      .dot {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("# 🧊 CryoClear")
st.markdown(
    '<div class="subtitle">Real-time particle-picking + junk-triage copilot — '
    "picks, flags junk (ice / carbon / aggregates), and learns from your corrections.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    empiar = st.text_input("EMPIAR id", value=config.DEMO_EMPIAR_ID)
    backend_label = st.selectbox(
        "Picker backend",
        ["blob — LoG (over-picks → best junk-triage demo)", "cryosegnet (cached)"])
    backend = "cryosegnet" if backend_label.startswith("cryosegnet") else "blob"
    factor = st.slider("Downsample factor", 1, 8, 4)
    box = st.slider("Feature box (px, display)", 24, 128, 64, 8)
    threshold = st.slider("Junk threshold", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    al_on = st.checkbox("🎓 Active-learning demo (M2)", value=False,
                        help="Seed a tiny model from a few labels, then teach it corrections "
                             "and watch the junk-rejection F1 climb live.")
    al_seed = st.slider("Seed examples", 10, 200, 30, 10, disabled=not al_on)
    st.caption("M1→M4 build order lives in CLAUDE.md")

RAW = config.RAW / empiar
CACHE = config.PROCESSED / empiar / "cryosegnet"
MODEL_PATH = config.PROCESSED / empiar / "junk_classifier.joblib"
RADIUS = config.particle_radius_px()


# ---------------------------------------------------------------- data + model helpers
def _demo_micrograph(h: int = 512, w: int = 512, n: int = 60, seed: int = 0) -> np.ndarray:
    """Synthetic micrograph with blobs so the UI runs before real data exists."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.5, 0.1, size=(h, w))
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        img += 0.6 * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 6.0 ** 2)))
    return (np.clip(img, 0, None) / img.max() * 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def _list_micrographs(empiar_id: str) -> list[str]:
    return [p.name for p in sorted((config.RAW / empiar_id / "micrographs").glob("*.mrc"))]


@st.cache_data(show_spinner=False)
def _load_full_res(path_str: str) -> np.ndarray:
    """Full-res 8-bit micrograph — features for the junk model must match training."""
    return io_mrc.normalize_8bit(io_mrc.load_mrc(path_str))


@st.cache_data(show_spinner=False)
def _load_train_table(empiar_id: str):
    """Labelled feature table saved by train_junk_classifier.py (for the M2 seed)."""
    p = config.PROCESSED / empiar_id / "junk_train.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["X"].astype(float), d["y"].astype(int)


def _load_model() -> JunkClassifier | None:
    if MODEL_PATH.exists():
        try:
            return JunkClassifier.load(MODEL_PATH)
        except Exception as e:  # pragma: no cover
            st.sidebar.warning(f"Could not load model: {e}")
    return None


def _picks(img_disp: np.ndarray, mic_name: str | None):
    """Return (pred_disp, pred_full) in display- and full-resolution pixel spaces."""
    if backend == "cryosegnet" and mic_name is not None:
        try:
            pred_full = picker.pick(img_disp, backend="cryosegnet",
                                    name=mic_name, cache_dir=CACHE)
            return pred_full / float(factor), pred_full
        except FileNotFoundError:
            st.warning("No cached CryoSegNet picks — run scripts/run_cryosegnet.py. "
                       "Falling back to the blob picker.")
    pred_disp = picker.pick(img_disp, backend="blob")
    return pred_disp, pred_disp * float(factor)


def _record_correction(i: int, feats: np.ndarray, label: bool = True) -> None:
    """Feed candidate `i`'s correction to the live learner; flag it red if it's junk."""
    if st.session_state.al_learner is not None and len(feats):
        st.session_state.al_learner.add_corrections(feats[i:i + 1], [int(label)])
    if label:
        st.session_state.flagged.setdefault(_key, set()).add(int(i))
    st.session_state.corrections += 1


# ---------------------------------------------------------------- session state
if "learner" not in st.session_state:
    st.session_state.learner = None      # ActiveLearner, seeded lazily
if "corrections" not in st.session_state:
    st.session_state.corrections = 0
if "flagged" not in st.session_state:
    st.session_state.flagged = {}        # mic_name -> set of user-flagged junk indices
if "stream_i" not in st.session_state:
    st.session_state.stream_i = 0
if "t0" not in st.session_state:
    st.session_state.t0 = time.time()
if "al_learner" not in st.session_state:
    st.session_state.al_learner = None     # ActiveLearner for the live M2 loop
if "al_seed" not in st.session_state:
    st.session_state.al_seed = None
if "f1_history" not in st.session_state:
    st.session_state.f1_history = []       # junk-F1 after each teach round
if "taught" not in st.session_state:
    st.session_state.taught = 0

# ---------------------------------------------------------------- pick a micrograph
mic_names = _list_micrographs(empiar)
have_real = bool(mic_names)

with st.sidebar:
    if have_real:
        if st.button("⏭ Stream next micrograph"):
            st.session_state.stream_i += 1
            st.session_state.t0 = st.session_state.get("t0", time.time())
        sel = st.selectbox("Micrograph", mic_names,
                           index=st.session_state.stream_i % len(mic_names))
    else:
        st.info("No real micrographs found — showing a synthetic demo.\n\n"
                f"Put .mrc files in {RAW}/micrographs/")
        sel = None

# ---------------------------------------------------------------- load + run pipeline
if have_real and sel is not None:
    img = io_mrc.load_for_pipeline(RAW / "micrographs" / sel, factor=factor)
    gt_path = RAW / "ground_truth" / (Path(sel).stem + ".star")
else:
    img = _demo_micrograph()
    gt_path = None

pred_disp, pred_full = _picks(img, sel)

clf = _load_model()
# Extract features the SAME way the model was trained: full-res image at full-res
# coords with the dataset box. (Synthetic/no-model path uses the display image.)
if have_real and sel is not None and len(pred_full):
    img_full = _load_full_res(str(RAW / "micrographs" / sel))
    feats = features.extract_features(img_full, pred_full, box=config.DEMO_PARTICLE_DIAMETER_PX)
else:
    feats = features.extract_features(img, pred_disp, box=box)

# Ground-truth-derived "true junk" labels for THIS micrograph — validates the live
# active-learning loop against expert labels (a candidate not matching any GT = junk).
true_is_junk = None
if gt_path is not None and gt_path.exists() and len(pred_full):
    _gt_xy = coords.read_star_coords(gt_path)
    _matches, _fp, _fn = metrics.match_particles(pred_full, _gt_xy, RADIUS)
    true_is_junk = np.ones(len(pred_full), dtype=bool)
    for _pi, _gi, _d in _matches:
        true_is_junk[_pi] = False

# M2: with the active-learning demo on, predict using a learner seeded from a few
# labels (so corrections visibly improve it); otherwise use the trained model.
train_tbl = _load_train_table(empiar) if al_on else None
if al_on and train_tbl is not None and len(feats):
    if st.session_state.al_learner is None or st.session_state.al_seed != al_seed:
        _Xt, _yt = train_tbl
        _rng = np.random.default_rng(0)
        # Cold start: a junk-POOR seed, so the model initially MISSES junk (low junk-F1);
        # the scientist's corrections then teach it — an honest, visible learning climb.
        _keep = np.where(_yt == 0)[0]
        _junk = np.where(_yt == 1)[0]
        _nj = max(2, al_seed // 10)
        _nk = max(2, al_seed - _nj)
        _idx = np.concatenate([
            _rng.choice(_keep, size=min(_nk, len(_keep)), replace=False),
            _rng.choice(_junk, size=min(_nj, len(_junk)), replace=False)])
        st.session_state.al_learner = ActiveLearner().seed(_Xt[_idx], _yt[_idx])
        st.session_state.al_seed = al_seed
        st.session_state.f1_history = []
        st.session_state.taught = 0
    is_junk = np.asarray(
        st.session_state.al_learner.predict_is_junk(feats, threshold=threshold), dtype=bool)
elif clf is not None and len(feats):
    is_junk = np.asarray(clf.predict_is_junk(feats, threshold=threshold), dtype=bool)
else:
    is_junk = np.zeros(len(pred_disp), dtype=bool)

# user corrections for this micrograph override the prediction (visible HITL)
_key = sel or "__synthetic__"
for i in st.session_state.flagged.get(_key, set()):
    if 0 <= i < len(is_junk):
        is_junk[i] = True

n_total = len(pred_disp)
n_junk = int(is_junk.sum())
n_keep = n_total - n_junk

# ---------------------------------------------------------------- layout
left, right = st.columns([3, 1], gap="large")

with left:
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(img, cmap="gray")
    if n_total:
        keep_xy, junk_xy = pred_disp[~is_junk], pred_disp[is_junk]
        if len(keep_xy):
            ax.scatter(keep_xy[:, 0], keep_xy[:, 1], s=90, facecolors="none",
                       edgecolors="#2ecc71", linewidths=1.3, label="keep")
        if len(junk_xy):
            ax.scatter(junk_xy[:, 0], junk_xy[:, 1], s=90, facecolors="none",
                       edgecolors="#e74c3c", linewidths=1.3, label="junk")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    st.pyplot(fig, width="stretch")
    st.markdown(
        '<div class="legend"><span><span class="dot" style="background:#2ecc71"></span>'
        'keep</span><span><span class="dot" style="background:#e74c3c"></span>junk</span></div>',
        unsafe_allow_html=True,
    )

with right:
    elapsed_min = max((time.time() - st.session_state.t0) / 60.0, 1e-6)
    mpm = (st.session_state.stream_i + 1) / elapsed_min if have_real else 0.0
    st.metric("Candidates", n_total)
    st.metric("Kept", n_keep)
    st.metric("Junk %", f"{(100.0 * n_junk / n_total) if n_total else 0:.1f}%")
    st.metric("Micrographs/min", f"{mpm:.1f}")

    if gt_path is not None and gt_path.exists():
        gt_xy = coords.read_star_coords(gt_path)
        before = metrics.picking_metrics(pred_full, gt_xy, radius=RADIUS)
        after = metrics.picking_metrics(pred_full[~is_junk], gt_xy, radius=RADIUS)
        st.divider()
        st.caption("Picking F1 vs CryoPPP ground truth")
        st.metric("F1 after junk triage", f"{after.f1:.3f}",
                  delta=f"{after.f1 - before.f1:+.3f} vs raw picks")
        st.caption(f"raw F1={before.f1:.3f} (P={before.precision:.2f} R={before.recall:.2f}) → "
                   f"kept F1={after.f1:.3f} (P={after.precision:.2f} R={after.recall:.2f})")

    if al_on and true_is_junk is not None and len(is_junk):
        jr = metrics.junk_rejection_metrics(is_junk, true_is_junk)
        st.divider()
        st.caption(f"🎓 Active learning — taught {st.session_state.taught} corrections")
        _delta = None
        if st.session_state.f1_history:
            _delta = f"{jr['junk_f1'] - st.session_state.f1_history[0]:+.3f} since seed"
        st.metric("Junk-rejection F1 (live)", f"{jr['junk_f1']:.3f}", delta=_delta)

    st.divider()
    st.caption(f"Corrections fed back: {st.session_state.corrections}")
    if clf is None:
        st.warning("No junk model yet — run scripts/train_junk_classifier.py "
                   f"--empiar {empiar}. Showing all candidates as keep.")

# ---------------------------------------------------------------- active learning (M2)
st.divider()
st.subheader("🎓 Human-in-the-loop active learning (M2)")

if al_on and st.session_state.al_learner is not None and true_is_junk is not None and len(feats):
    learner = st.session_state.al_learner
    if not st.session_state.f1_history:
        st.session_state.f1_history = [
            metrics.junk_rejection_metrics(is_junk, true_is_junk)["junk_f1"]]

    c1, c2, c3 = st.columns([1.3, 1, 1])
    teach_n = c1.select_slider("Corrections / round", options=[5, 10, 20, 50], value=10)
    if c2.button("👩‍🔬 Teach", type="primary"):
        wrong = np.where(is_junk != true_is_junk)[0]
        if len(wrong):
            take = wrong[:teach_n]
            learner.add_corrections(feats[take], true_is_junk[take])
            st.session_state.taught += len(take)
            new_pred = np.asarray(learner.predict_is_junk(feats, threshold=threshold), dtype=bool)
            st.session_state.f1_history.append(
                metrics.junk_rejection_metrics(new_pred, true_is_junk)["junk_f1"])
        st.rerun()
    if c3.button("↺ Reset"):
        st.session_state.al_learner = None
        st.session_state.al_seed = None
        st.rerun()

    if len(st.session_state.f1_history) > 1:
        import pandas as pd
        st.caption("Junk-rejection F1 climbs as you teach it (seed → correction rounds):")
        st.line_chart(pd.DataFrame({"junk-F1": st.session_state.f1_history}), height=180)
    st.caption(f"seed F1 {st.session_state.f1_history[0]:.3f} → now "
               f"{st.session_state.f1_history[-1]:.3f}  ·  {st.session_state.taught} corrections taught")
elif al_on:
    st.info("Active-learning demo needs the saved training table "
            f"(data/processed/{empiar}/junk_train.npz) and ground truth. "
            "Run scripts/train_junk_classifier.py first.")
else:
    st.caption("Turn on **Active-learning demo (M2)** in the sidebar to teach the model live, "
               "or click a candidate below to correct it.")

# manual click-to-correct (feeds the true label when known)
if _HAVE_CLICK:
    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    clicked = click_image(rgb, key="clickimg")
    if clicked and n_total:
        d = np.linalg.norm(pred_disp - np.array([clicked["x"], clicked["y"]]), axis=1)
        i = int(np.argmin(d))
        if d[i] <= box:
            lbl = bool(true_is_junk[i]) if true_is_junk is not None else True
            _record_correction(i, feats, lbl)
            st.rerun()

st.caption("Pickers: blob (LoG) + CryoSegNet (SAM, cached) · junk classifier: RandomForest · "
           "live active learning · open source (MIT). Build ladder M1→M4 in CLAUDE.md.")
