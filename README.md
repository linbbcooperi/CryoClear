# CryoClear 🧊🔬

An open, real-time cryo-EM **particle-picking + junk-removal copilot** 
It picks protein particles from cryo-EM micrographs, **flags and removes junk** (ice, carbon edges, aggregates) as the data streams in, and **gets smarter every time the scientist corrects it** — and it reports real metrics against expert ground truth.

> Built on top of state-of-the-art pickers (Topaz, CryoSegNet, CryoFSL). We don't claim a new picking model — our contribution is the **interactive, real-time junk-triage product** and a measurable junk classifier.

## Table of Contents

- [Overview](#overview)
- [What CryoClear Does](#what-cryoclear-does)
- [Quickstart (uv)](#quickstart-uv)
- [Features (M1–M4)](#features-the-milestone-ladder-all-live)
- [Architecture](#architecture)
- [Junk Classifier](#junk-classifier-the-novel-piece)
- [Evaluation & Metrics](#evaluation--metrics-honest)
- [Status](#status)
- [License](#license)

---

## Overview

Cryo-EM datasets can contain thousands of micrographs, and a significant fraction of picked "particles" are junk — ice contamination, carbon edge artifacts, aggregates, and damaged regions. Today, scientists catch this junk manually: they scroll through picks after the run, discard bad ones by eye, re-run the picker with adjusted settings, and repeat. This loop is slow, error-prone, and happens *after* hours of data collection, not during it.

Existing autopickers (Topaz, crYOLO, CryoSegNet) excel at finding particles but ship no junk-triage layer. The feedback from a scientist's corrections never flows back into the session automatically. And there is no standard, real-time dashboard showing how clean the particle stack actually is.

## What CryoClear Does

CryoClear wraps state-of-the-art pickers with an interactive, real-time junk-removal copilot:

- **Streams micrographs as they arrive** — no waiting for the full dataset
- **Classifies picks on the fly** — flags ice patches, carbon edges, and aggregates before they contaminate your stack
- **Learns from corrections** — every accept/reject from the scientist updates the junk classifier in-session
- **Reports honest metrics** — precision, recall, and F1 against expert ground truth, so you know exactly how clean your stack is

---

## Quickstart (uv)

We use [uv](https://docs.astral.sh/uv/) for env + packaging.

```bash
# 0. install uv once:  curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. create the env + install everything (project, deps, dev tools) and lock
uv sync

# 2. sanity check (tests should pass out of the box)
uv run pytest -q          # or: make test

# 3. run the UI
uv run streamlit run app/streamlit_app.py     # or: make app

# 4. eval harness (synthetic demo; works with no data)
uv run python eval/run_eval.py --demo         # or: make eval
```

## On the GPU box (RunPod)

```bash
uv sync                                  # our package + CPU deps
# install torch (CUDA build) from pytorch.org, then CryoSegNet (its own conda env)
uv pip install -e ".[gpu]"               # ASPIRE for the 2D-class wow (optional)
```

Full GPU + data + milestone plan: [`docs/07_runpod_build_plan.md`](docs/07_runpod_build_plan.md).

---

## Features (the milestone ladder, all live)

| | Feature | What it does |
|---|---|---|
| **M1** | Picking + junk flags + metrics | Pick particles (blob LoG or CryoSegNet), classify each candidate keep/junk, score precision/recall/F1 vs CryoPPP ground truth. |
| **M2** | Live active learning | Cold-start a tiny junk model, teach it corrections, watch junk-rejection F1 climb live (the "it learns from me" loop). |
| **M3** | Real-time streaming | Auto-feed micrographs on a timer with a live throughput + %junk dashboard. |
| **M4** | 2D class averages | Reference-free 2D classification of the kept particles → class-average montage (the "these picks are real" proof). |

Run each: `make app` then toggle the sidebar (Active-learning / Auto-stream) and the
"2D class averages" expander. CLI: `scripts/run_baseline.py`, `scripts/train_junk_classifier.py`,
`eval/class_averages.py`, `eval/heldout_eval.py`.

## Architecture

```
MRC micrograph → preprocess (io_mrc) → picker (blob LoG | CryoSegNet/SAM, cached .star)
   → per-candidate features (features.py) → JUNK CLASSIFIER (RandomForest, junk_classifier.py)
   → active-learning update from corrections (active_learning.py)
   → UI overlay green=keep / red=junk + live metrics (metrics.py) + stream dashboard (stream.py)
   → 2D class averages of kept particles (class2d.py)  → montage
```

## Junk Classifier (the novel piece)

A lightweight scikit-learn RandomForest on 8 interpretable per-candidate features
(mean/std/contrast/edge-density/blobiness…). Labels come from CryoPPP: a candidate
matching a ground-truth particle = keep, otherwise = junk. Trains in seconds, updates
instantly for the human-in-the-loop loop.

## Evaluation & Metrics (honest)

On EMPIAR-10017 β-galactosidase (real CryoPPP ground truth):

- **The problem:** the blob picker over-picks — recall ≈ 0.997 but precision ≈ 0.13
  (~4,300 junk false-positives per micrograph), picking **F1 ≈ 0.23**.
- **Junk classifier (held-out, micrograph-level):** junk-rejection **F1 ≈ 0.93**.
- **After junk triage (in-sample):** picking **F1 ≈ 0.998** — *optimistic / in-sample*.
- **After junk triage (held-out):** modest improvement at a calibrated threshold
  (picking F1 ≈ 0.22 → ≈ 0.36); the default threshold over-rejects on unseen micrographs.
  More training micrographs + per-dataset threshold calibration would improve this.
- **CryoSegNet** runs on the GPU (incl. NVIDIA Blackwell via a torch-cu128 env) but
  under-picks β-gal with default settings (recall ≈ 0.11); blob + junk-triage is the
  stronger demo.

Reproduce: `uv run python eval/heldout_eval.py --empiar 10017 --radius 54 --box 108`.

## Running on the GPU box

See [`docs/07_runpod_build_plan.md`](docs/07_runpod_build_plan.md) and
[`docs/08_runpod.md`](docs/08_runpod.md). CryoSegNet on a Blackwell GPU:
[`scripts/setup_cryosegnet_blackwell.sh`](scripts/setup_cryosegnet_blackwell.sh).

---

## Status

**M1–M4 are all built and running on real data** (EMPIAR-10017). The app is an interactive
Streamlit copilot; `metrics.py`, `features.py`, `coords.py`, `class2d.py` are tested. Honest
about limits: in-sample picking numbers are optimistic; held-out generalization of the
*picking* improvement needs threshold calibration (the *junk-rejection* generalizes well).

## License

MIT (required by the hackathon — see `LICENSE`).