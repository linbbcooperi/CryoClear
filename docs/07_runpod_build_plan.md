# CryoTriage Live — Day-1 Build Plan (RunPod + CryoSegNet + M1 + UI)

> Toolchain note: this repo uses **uv** for env + packaging. `uv sync` makes the venv,
> installs the project (editable) + dev group, and writes `uv.lock`. `uv run …` auto-syncs.

## Context
Day 1 of the QBI Hackathon (48h, submit Sun 4:30 PM). The repo is a strong scaffold: the **entire
core library is done and tested** (`metrics`, `features`, `junk_classifier`, `active_learning`,
`io_mrc`, `coords`, `stream`, `config`). What's missing is the real-data + GPU wiring and a
presentable UI. This plan gets us to **M1 (guaranteed presentable result): CryoSegNet picking +
junk flags on EMPIAR-10025 with precision/recall vs CryoPPP ground truth**, running on a **RunPod
GPU**, with a **polished Streamlit demo**. Decisions locked: engine = **CryoSegNet**, dataset =
**EMPIAR-10025 (T20S proteasome)**, priority = **full M1 + RunPod env + nice UI**.

Key architectural decision: **CryoSegNet runs offline on the GPU and writes `.star` files; we cache
those picks**. The pipeline and UI then read cached coords via `coords.read_star_coords()`. This keeps
the live demo fast and robust (CryoSegNet inference is slow) and means the UI never blocks on the GPU.
The always-working **blob** backend stays as the M0 fallback rung.

## What you need
- **RunPod** account + credit. GPU: **RTX 4090 (24 GB)** or A5000/A40. (Venue 2080 Ti, 11 GB, also works.)
- **RunPod Network Volume** (~50–100 GB) mounted at `/workspace` so data + weights + repos survive pod
  restarts. Do this first.
- **CryoPPP / EMPIAR**: no login for EMPIAR; CryoPPP metadata sheet gives per-protein paths
  (`docs/03_datasets.md`). Download **only EMPIAR-10025** (tens of GB), not the 2.6 TB full set.
- Local machine: a browser + SSH client; Streamlit reached via RunPod's HTTP proxy (port 8501).

## Phase A — RunPod environment (Tony; ~1–2h)
1. **Deploy pod**: RunPod official PyTorch 2.x / CUDA 12.x template. Attach the Network Volume at
   `/workspace`. Expose ports **8501** (Streamlit) and **22** (SSH).
2. **Clone our repo** into `/workspace/CryoClear`, then:
   - `curl -LsSf https://astral.sh/uv/install.sh | sh`  (install uv)
   - `uv sync`  (creates .venv, installs project + dev group, writes uv.lock)
   - `make test` → all current tests must stay green (box is healthy).
3. **Install CryoSegNet** in its own conda env under `/workspace/CryoSegNet`:
   - `git clone https://github.com/jianlin-cheng/CryoSegNet` → `conda env create -f environment.yml` →
     `conda activate cryosegnet`.
   - Weights: `curl https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz -o w.tar.gz &&
     tar -xvf w.tar.gz`.
   - Smoke test: `python predict_new_data_mrc.py --my_dataset_path <one_mrc_dir> --output_path output`
     then `python generate_starfile_new_data_mrc.py` → confirm a `.star` appears.
4. **Streamlit reachability**: `make app` (binds 0.0.0.0:8501), open via the RunPod proxy URL.

## Phase B — Data + cached picks (Tony + Eva; ~1–2h, parallel with A)
1. Download **EMPIAR-10025** into the layout `data/README.md` specifies:
   `data/raw/10025/micrographs/*.mrc` and `data/raw/10025/ground_truth/*.star`. Start with ~20–40
   micrographs.
2. Verify I/O: `io_mrc.load_for_pipeline()` reads an `.mrc`; `coords.read_star_coords()` reads a GT
   `.star`. Overlay GT on one micrograph to confirm coordinate convention/scale.
3. **Run CryoSegNet once** over the micrographs (from the cryosegnet conda env):
   `uv run python scripts/run_cryosegnet.py --cryosegnet-dir /workspace/CryoSegNet --empiar 10025`
   → caches one `.star` per micrograph under `data/processed/10025/cryosegnet/`.

## Phase C — Wire CryoSegNet into our pipeline (Bindu; DONE in code)
- `src/cryotriage/picker.py` — `_cryosegnet_pick` reads cached `.star` via `coords.read_star_coords()`,
  same `(N,2)` contract as `_blob_pick`; `backend="cryosegnet"` in `pick()`. Blob stays as fallback.
- `scripts/run_baseline.py` — `uv run python scripts/run_baseline.py --backend cryosegnet --empiar 10025`
  prints **precision/recall/F1 vs CryoPPP** (the M1 number; handles full-res vs downsampled coords).

## Phase D — Junk classifier on real CryoPPP labels (Bindu + Eva; DONE in code)
- `scripts/train_junk_classifier.py` — for each micrograph: load image + cached CryoSegNet candidates,
  label keep/junk by matching against GT within `particle_radius` (matched = keep, unmatched FP = junk),
  `features.extract_features()`, `JunkClassifier.fit()`, report junk-rejection metrics, save model to
  `data/processed/10025/junk_classifier.joblib`.  `--demo` stays green offline.
- Run: `uv run python scripts/train_junk_classifier.py --empiar 10025`  (or `make train`).

## Phase E — Polished UI demo (Bindu; DONE in code) → finishes M1, sets up M2/M3
- `.streamlit/config.toml` — clean dark theme.
- `streamlit-image-coordinates` (in deps) — **click-to-reject** on the micrograph.
- `app/streamlit_app.py` wired to the lib: real MRC load, cryosegnet picks, **green=keep / red=junk**
  overlay, live scoreboard (Candidates / Kept / Junk% / micrographs-min) + **picking F1 vs CryoPPP**,
  stream-next, and click-to-reject → `ActiveLearner` + visible re-color (M2). Synthetic fallback so it
  always runs.

## Milestone ladder (never break a lower rung)
- **M0 (green):** `make test`, `make eval`, blob picker in UI.
- **M1 (today):** CryoSegNet cached picks + junk flags on 10025 + P/R/F1 vs CryoPPP, clean UI.
- **M2 (Sat eve):** click-to-reject → refit → junk metric improves.
- **M3 (Sun AM):** stream mode + live micrographs/min.
- **M4 (Sun, wow):** ASPIRE 2D class averages of kept particles (`uv pip install -e ".[gpu]"`).

## Verification (end-to-end)
1. `make test` green on the pod; CryoSegNet smoke test writes a `.star`.
2. `uv run python scripts/run_baseline.py --backend cryosegnet --empiar 10025` → mean P/R/F1 above blob.
3. `uv run python scripts/train_junk_classifier.py --empiar 10025` → junk-rejection P/R/F1; `--demo` green.
4. `make app` → real micrograph with green/red boxes + scoreboard; click a junk box → metric updates.
5. Log seed, particle_radius, backend, EMPIAR id, counts (per `eval/EVALS.md`).

## Risks & fallbacks
- **CryoSegNet flaky** → cached `.star` (Phase B3) means the demo never depends on live GPU inference;
  worst case the blob backend (M0) still gives a full demo + metrics.
- **RunPod data lost on restart** → everything heavy lives on the `/workspace` Network Volume.
- **Streamlit interactivity** → `streamlit-image-coordinates` covers click-to-reject; FastAPI+canvas is a
  stretch, not a dependency.
- **Time** → M1 is the floor (Phases A–E). M2–M4 only start once M1 is demoable.
