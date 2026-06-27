# CLAUDE.md — CryoTriage Live

> Read this first. It is the single source of truth for this project. Detailed background lives in `docs/`.

## What we're building
**CryoTriage Live** — an open, real-time cryo-EM **particle-picking + junk-removal copilot**. It picks protein particles from cryo-EM micrographs, **flags/removes junk** (ice, carbon edges, aggregates) as data streams in, and **learns from the scientist's corrections live**.

One-liner for judges: *"CryoFSL made few-shot picking accurate; we make picking + junk-cleaning interactive, real-time, and measurable — on a single GPU."*

## Why it exists (the gap)
Auto-pickers (Topaz) over-pick junk; cleaning it is slow, manual, and post-hoc. Research models (Topaz, CryoSegNet, CryoFSL) are batch/CLI. Real-time platforms (cryoSPARC Live, Warp) use simple pickers and are heavy/closed. **Nobody ships an open, interactive, real-time junk-triage copilot with a human-in-the-loop learning loop.** That's our novelty — the product + the junk intelligence + the live UX, NOT a new picking model.

## Event constraints (do not violate)
- QBI Hackathon 2026, **June 27–28**, UCSF Mission Bay. 48 hours. Submit by Sun 4:30 PM.
- **License: MIT** (required — all hackathon output must be open source).
- Hardware on site: 4× RTX 2080 Ti (11 GB each) + some cloud compute. **Design for a single GPU; keep models light.**

## Team & ownership
- **Bindu (CS):** picking-engine integration, the **junk classifier** (novel ML), active-learning loop, UI, metrics.
- **Tony (hardware/systems):** GPU/runtime, MRC I/O, the **real-time stream simulator**, throughput dashboard, Docker.
- **Eva (chemistry):** chooses demo protein, is the human-in-the-loop in the demo, validates 2D class averages, owns the scientific story/slides.

## Architecture (keep it modular)
```
MRC micrograph → preprocess → picker (Topaz/CryoSegNet) → candidate boxes
   → JUNK CLASSIFIER (keep / ice / carbon / aggregate)   ← the novel piece
   → active-learning update from user corrections
   → UI overlay (green=keep, red=junk) + live metrics
   → 2D classification of kept particles (ASPIRE)  → class-average montage (the "wow")
```

## Repo layout
- `src/cryotriage/` — library code (import as `cryotriage`)
  - `io_mrc.py` read/normalize MRC · `coords.py` .star/.box IO · `picker.py` picker wrapper
  - `features.py` per-candidate features · `junk_classifier.py` the novel classifier
  - `active_learning.py` HITL update · `metrics.py` precision/recall/F1 (DONE, tested)
  - `stream.py` real-time simulator · `config.py` paths/constants
- `app/streamlit_app.py` — interactive UI skeleton
- `scripts/` — `download_cryoppp.py`, `run_baseline.py`, `train_junk_classifier.py`
- `eval/run_eval.py` + `eval/EVALS.md` — evaluation harness & rubric
- `tests/` — pytest (start green: `test_metrics.py`)
- `data/raw`, `data/processed` — gitignored; see `data/README.md`
- `docs/` — full context (see index below)

## Build order — milestone "fallback ladder" (always keep the lower rung working)
1. **M1 (must-have):** static picking + junk flags on ONE protein + **precision/recall vs CryoPPP ground truth**. This alone is presentable.
2. **M2:** interactive correction that visibly improves the metric (active learning).
3. **M3:** real-time streaming dashboard (Tony's stream simulator).
4. **M4 (wow):** 2D class averages of kept particles via ASPIRE.
Build in this exact order. Never break a working rung to chase the next.

## Evals (what we measure — see `eval/EVALS.md`)
- **Technical:** picking **precision / recall / F1** vs CryoPPP ground truth; **junk-rejection precision/recall** (using CryoPPP's labeled ice/carbon false positives); throughput (micrographs/min). Beat a **Topaz baseline**.
- **Hackathon rubric (judges):** Impact · Execution (usable product) · Professional-background variety · Novelty. Keep every decision pointed at these.
- Run: `python eval/run_eval.py --pred <coords> --gt <coords> --particle-radius <px>` or `make eval`.

## Datasets (see `docs/03_datasets.md` — download a SMALL subset, not 2.6 TB)
- **CryoPPP** (labeled ground truth incl. junk) — github.com/BioinfoMachineLearning/cryoppp
- **EMPIAR** raw micrographs (hero demo: T20S proteasome 10025 or β-gal 10017) — ebi.ac.uk/empiar
- Pretrained pickers: **Topaz** (baseline), CryoSegNet, CryoFSL. SAM2 optional.

## Coding conventions
- Python 3.10, `cryotriage` package, type hints + docstrings, `ruff`/`black` style.
- **Junk classifier: start SIMPLE** (scikit-learn RandomForest on `features.py`), upgrade to a small CNN only if M1–M3 are solid.
- Keep per-micrograph inference to a few seconds (cache picks/embeddings).
- Pure functions in `src/`; keep Streamlit glue in `app/`. Write a test when you add a metric.
- Don't add heavy deps (no full RELION/cryoSPARC). ASPIRE only for the 2D-class wow.

## Commands (uv toolchain)
- Setup: `make setup` (`uv sync` — makes .venv, installs project + dev group, writes uv.lock)
- App: `make app` (`uv run streamlit run app/streamlit_app.py`)
- Tests: `make test` (`uv run pytest -q`)
- Eval: `make eval` · Baseline: `make baseline` · Train junk clf: `make train`
- GPU box: `uv sync`, install torch (CUDA) + CryoSegNet out-of-band, `uv pip install -e ".[gpu]"`

## Docs index
- `docs/00_hackathon_context.md` — dates, venue, scope, rules, agenda, why A2 fits the category
- `docs/01_idea_cryotriage.md` — the idea + novelty + honest prior art
- `docs/02_build_plan.md` — full 48h plan (imported)
- `docs/07_runpod_build_plan.md` — RunPod + CryoSegNet + M1 Day-1 execution plan (current)
- `docs/03_datasets.md` — every dataset/library with links + download steps
- `docs/04_evals.md` / `eval/EVALS.md` — judging rubric + technical metrics
- `docs/05_demo_script.md` — 5-min demo + slide outline
- `docs/06_roles.md` — detailed task split
- `docs/90–92_*.md` — supporting research (idea comparisons, UCSF context, 2025 recap)
