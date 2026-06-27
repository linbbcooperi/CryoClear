# CryoClear 🧊🔬

An open, real-time cryo-EM **particle-picking + junk-removal copilot** 
It picks protein particles from cryo-EM micrographs, **flags and removes junk** (ice, carbon edges, aggregates) as the data streams in, and **gets smarter every time the scientist corrects it** — and it reports real metrics against expert ground truth.

> Built on top of state-of-the-art pickers (Topaz, CryoSegNet, CryoFSL). We don't claim a new picking model — our contribution is the **interactive, real-time junk-triage product** and a measurable junk classifier.

## Table of Contents

- [Overview](#Overview)
- [What CryoClear Does](#what-cryoclear-does)
- [Quickstart (uv)](#quickstart-uv)
- [On the GPU box (RunPod)](#on-the-gpu-box-runpod)
- [Architecture](#architecture)
- [Junk Classifier](#junk-classifier)
- [Evaluation & Metrics](#evaluation--metrics)
- [API Endpoints (RunPod)](#api-endpoints-runpod)
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

## Architecture

_TODO_

## Junk Classifier

_TODO_

## Evaluation & Metrics

_TODO_

## API Endpoints (RunPod)

_TODO_

---

## Status

This is a **starter scaffold**. `metrics.py` and its tests are real and pass. Everything else is a clearly-marked stub with `TODO`s — build it following the milestone ladder in `CLAUDE.md`.

## License

MIT (required by the hackathon — see `LICENSE`).