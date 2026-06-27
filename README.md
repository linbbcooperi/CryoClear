# CryoTriage Live 🧊🔬

An open, real-time cryo-EM **particle-picking + junk-removal copilot** for the QBI Hackathon 2026 (UCSF, June 27–28).

It picks protein particles from cryo-EM micrographs, **flags and removes junk** (ice, carbon edges, aggregates) as the data streams in, and **gets smarter every time the scientist corrects it** — and it reports real metrics against expert ground truth.

> Built on top of state-of-the-art pickers (Topaz, CryoSegNet, CryoFSL). We don't claim a new picking model — our contribution is the **interactive, real-time junk-triage product** and a measurable junk classifier.

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

### On the GPU box (RunPod)
```bash
uv sync                                  # our package + CPU deps
# install torch (CUDA build) from pytorch.org, then CryoSegNet (its own conda env)
uv pip install -e ".[gpu]"               # ASPIRE for the 2D-class wow (optional)
```
Full GPU + data + milestone plan: [`docs/07_runpod_build_plan.md`](docs/07_runpod_build_plan.md).

## What's here
- `src/cryotriage/` — library (MRC IO, picker wrapper, **junk classifier**, **metrics**, stream simulator)
- `app/streamlit_app.py` — interactive picking + junk-triage UI
- `eval/` — evaluation harness + rubric (`EVALS.md`)
- `scripts/` — data download + baseline + training helpers
- `docs/` — full context: hackathon rules, the idea, datasets, 48h plan, demo script
- `CLAUDE.md` — start here if you're using Claude Code

## Status
This is a **starter scaffold**. `metrics.py` and its tests are real and pass. Everything else is a clearly-marked stub with `TODO`s — build it following the milestone ladder in `CLAUDE.md`.

## License
MIT (required by the hackathon — see `LICENSE`).
