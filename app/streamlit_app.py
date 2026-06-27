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
    backend_label = st.selectbox("Picker backend", ["cryosegnet (cached)", "blob (placeholder)"])
    backend = "cryosegnet" if backend_label.startswith("cryosegnet") else "blob"
    factor = st.slider("Downsample factor", 1, 8, 4)
    box = st.slider("Feature box (px, display)", 24, 128, 64, 8)
    threshold = st.slider("Junk threshold", 0.0, 1.0, 0.5, 0.05)
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


def _record_correction(i: int, feats: np.ndarray) -> None:
    """Flag candidate `i` as junk for this micrograph and feed the ActiveLearner."""
    learner = st.session_state.learner or ActiveLearner(_load_model() or JunkClassifier())
    learner.add_corrections(feats[i:i + 1], [True])
    st.session_state.learner = learner
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
feats = features.extract_features(img, pred_disp, box=box)

clf = _load_model()
if clf is not None and len(feats):
    is_junk = np.asarray(clf.predict_is_junk(feats, threshold=threshold), dtype=bool)
else:
    is_junk = np.zeros(len(pred_disp), dtype=bool)

# user corrections for this micrograph override the model (visible HITL — M2)
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
        score = metrics.picking_metrics(pred_full, coords.read_star_coords(gt_path),
                                        radius=RADIUS)
        st.divider()
        st.caption("Picking vs CryoPPP ground truth")
        st.metric("F1", f"{score.f1:.3f}")
        st.caption(f"P={score.precision:.3f} · R={score.recall:.3f} · "
                   f"TP={score.tp} FP={score.fp} FN={score.fn}")

    st.divider()
    st.caption(f"Corrections fed back: {st.session_state.corrections}")
    if clf is None:
        st.warning("No junk model yet — run scripts/train_junk_classifier.py "
                   f"--empiar {empiar}. Showing all candidates as keep.")

# ---------------------------------------------------------------- active learning (M2)
st.divider()
st.subheader("Human-in-the-loop correction")

if not _HAVE_CLICK:
    st.caption("Install `streamlit-image-coordinates` for click-to-reject on the image. "
               "Until then, use the buttons below.")
    c1, c2, _ = st.columns([1, 1, 4])
    flip_to_junk = c1.button("Flag brightest candidate as junk")
    if flip_to_junk and n_total:
        # heuristic stand-in for a click: pick the highest-contrast not-yet-flagged candidate
        order = np.argsort(-feats[:, 4]) if len(feats) else np.array([0])
        already = st.session_state.flagged.get(_key, set())
        i = next((int(k) for k in order if int(k) not in already), int(order[0]))
        _record_correction(i, feats)
        st.rerun()
else:
    st.caption("Click a candidate on the image below to mark it **junk** "
               "and feed the correction to the model.")
    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    clicked = click_image(rgb, key="clickimg")
    if clicked and n_total:
        cx, cy = clicked["x"], clicked["y"]
        d = np.linalg.norm(pred_disp - np.array([cx, cy]), axis=1)
        i = int(np.argmin(d))
        if d[i] <= box:  # only if the click is near a candidate
            _record_correction(i, feats)
            st.rerun()

st.caption("Engine: CryoSegNet (cached) · junk classifier: RandomForest on per-candidate features · "
           "open source (MIT). Build ladder M1→M4 in CLAUDE.md.")
