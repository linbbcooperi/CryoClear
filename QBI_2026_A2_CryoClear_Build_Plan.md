# A2 → "CryoClear" — Full Build Plan for QBI Hackathon 2026

*A winnable, novel cryo-EM project for a 3-person team, engineered for visible results by Sunday 4:30 PM.*

**Team:** Tony (hardware/systems) · Eva (chemistry researcher) · Bindu (computer science)
**Event:** June 27–28, 2026, UCSF Mission Bay · Hardware on site: 4× RTX 2080 Ti + cloud compute · MIT open-source required.

---

## 1. The refined idea (and why it's novel)

### The one-liner
> **CryoClear** — an open, real-time cryo-EM "picking copilot" that doesn't just *find* protein particles, it **throws out the junk** (ice, carbon edges, aggregates) as the data streams in, and **gets smarter every time the scientist corrects it.**

### Why this, specifically
In cryo-EM you photograph thousands of frozen protein molecules, then must locate every good "particle" before a 3D structure can be built. Two painful truths:
1. **Junk is the real bottleneck, not finding particles.** The best auto-pickers (Topaz) *over-pick* — they grab ice crystals, carbon film edges, and protein clumps. Cleaning that out is slow, manual, and happens *after* the fact.
2. **It's not interactive or real-time for normal users.** Research models (Topaz, crYOLO, CryoSegNet, and the brand-new SAM2-based CryoFSL) are batch, command-line, expert tools. The big platforms that *do* run live (cryoSPARC Live, Warp) use simple blob/template pickers and are heavyweight licensed software.

### What already exists (honest novelty baseline — read this)
- **Topaz / crYOLO** — standard deep-learning pickers (batch, CLI).
- **CryoSegNet (2024)** — SAM + U-Net, very accurate.
- **CryoFSL (Sept 2025)** — *already* a SAM2 few-shot picker (as few as 5 labeled micrographs). → So "few-shot SAM2 picking" is taken. **We do not claim a new picking model.**
- **cryoSPARC Live / Warp** — real-time preprocessing, but blob/template pickers, closed/heavy.

### Where the genuine white space is (your novelty)
Nobody packages these into an **open, interactive, real-time picking + junk-triage copilot with a human-in-the-loop learning loop.** Our novelty is the **product + the junk-removal intelligence + the live UX**, not the picker math:
1. **A dedicated "junk classifier."** Train a lightweight model to recognize the *false positives* (ice contamination, carbon edges, aggregates) — using the fact that **CryoPPP literally labels these contaminants**. No mainstream picker ships an interactive junk-rejector. *(This is your concrete, novel, measurable ML contribution.)*
2. **Human-in-the-loop active learning.** The scientist rejects a few bad picks / one junk cluster, and the model updates **live** — accuracy visibly climbs in front of the judges.
3. **Real-time streaming operation on one GPU.** Micrographs stream in like a live microscope session; the tool picks + triages each in seconds and shows a live quality dashboard. This is Tony's hardware/systems showcase.

> **Defensible pitch:** *"CryoFSL made few-shot picking accurate. We make picking + junk-cleaning interactive and real-time — and we measure it: more true particles, far less junk, live, on a single GPU."*

---

## 2. What "visible results" you'll show the judges (design backward from this)

Three concrete, judge-friendly results — build the project so you always have at least #1:

1. **Hard numbers vs expert ground truth.** On held-out CryoPPP micrographs of 2–3 proteins: **precision / recall / F1** for particle picking, and a separate **junk-rejection precision/recall** (using CryoPPP's labeled ice/carbon false positives). Show a bar chart vs a **Topaz baseline**. *Numbers + a target you beat = credibility.*
2. **Live interactive demo.** Stream micrographs → picks + junk flags appear → Eva rejects a junk cluster with one click → the on-screen metrics update and improve. *The "it learns from me" moment.*
3. **The "these picks are real" proof (the wow).** Run a quick **2D classification** on your picked particles and show clean **class averages** (recognizable protein views). This is the sanity check every cryo-EM scientist trusts. Even a simple class-average montage is very persuasive.

---

## 3. All datasets & resources (with links)

### Primary dataset — labeled ground truth (this is the project's backbone)
- **CryoPPP** — large expert-labeled cryo-EM dataset: 34 EMPIAR protein sets, ~9,893 micrographs with ground-truth particle coordinates **and labeled false positives (ice, carbon edges)**. Full set is 2.6 TB — **you only download 2–3 proteins (~tens of GB).**
  - GitHub (scripts + structure): https://github.com/BioinfoMachineLearning/cryoppp
  - Paper (open): https://pmc.ncbi.nlm.nih.gov/articles/PMC10287764/
  - Metadata spreadsheet (per-EMPIAR download paths): http://calla.rnet.missouri.edu/cryoppp/EMPIAR_metadata_335.xlsx

### Raw micrograph source
- **EMPIAR** (Electron Microscopy Public Image Archive): https://www.ebi.ac.uk/empiar/
  - Good "clear particle" demo candidates (confirm exact IDs against the CryoPPP list): **EMPIAR-10025** (T20S proteasome), **EMPIAR-10017** (β-galactosidase), **EMPIAR-10081** (HCN1 channel), **EMPIAR-10028** (Plasmodium 80S ribosome), **EMPIAR-10005** (TRPV1).
  - **Pick one large, high-contrast protein for the demo** (proteasome or β-gal are forgiving); keep a second/third for "it generalizes."

### Pretrained pickers (use as engine + baseline — don't train from scratch)
- **Topaz** (your baseline to beat): https://github.com/tbepler/topaz
- **CryoSegNet** (SAM + U-Net, same lab as CryoPPP): search GitHub `BioinfoMachineLearning/cryosegnet` for code + weights.
- **CryoFSL** (SAM2 few-shot; cite + optionally build on): https://www.biorxiv.org/content/10.1101/2025.09.19.677446 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12458156/
- **crYOLO** (alternative picker): https://cryolo.readthedocs.io
- **SAM2** (if you go the prompt route): https://github.com/facebookresearch/sam2

### Supporting libraries
- **mrcfile** — read/write MRC micrographs in Python: https://github.com/ccpem/mrcfile
- **starfile** — read/write `.star` coordinate files: https://github.com/teamtomo/starfile
- **ASPIRE-Python** — lightweight cryo-EM toolkit for the **2D classification "wow"** (avoids heavy RELION/cryoSPARC): https://github.com/ComputationalCryoEM/ASPIRE-Python
- Core stack: Python 3.10+, PyTorch + CUDA, NumPy, scikit-image, scikit-learn, OpenCV, pandas, matplotlib.
- UI: **Streamlit** (fastest to build) or FastAPI/Flask + a JS image canvas (more control over real-time). Recommend Streamlit unless Bindu wants custom streaming.

---

## 4. Tech architecture (simple, modular)

```
                 ┌────────────────────────────────────────────┐
  MRC micrographs│  Tony: Stream Simulator + MRC/IO + GPU runtime│
  (EMPIAR/CryoPPP)│  feeds 1 micrograph at a time, on a timer     │
                 └───────────────┬───────────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────┐
        │ Bindu: Picking engine (pretrained Topaz/CryoSegNet)│  → candidate boxes
        ├──────────────────────────────────────────────┤
        │ Bindu: JUNK CLASSIFIER (trained on CryoPPP       │  → keep / ice / carbon / aggregate
        │        false-positive labels)  ← the novel part  │
        ├──────────────────────────────────────────────┤
        │ Active-learning loop: Eva's corrections update   │
        │        the junk classifier live                  │
        └───────────────┬──────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────┐
        │ UI: live micrograph view + boxes (green=keep,     │
        │ red=junk) + dashboard (count, %junk, P/R, /min)   │
        │ + "reject this cluster" buttons                   │
        └───────────────┬──────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────┐
        │ Eva: 2D classification of kept particles (ASPIRE) │ → class-average montage (the wow)
        └──────────────────────────────────────────────┘
```

The junk classifier can start dead-simple (features per candidate box: local variance, mean intensity, gradient/edge density, size, "blobiness" → a scikit-learn random forest / logistic regression on CryoPPP keep-vs-junk labels). That alone gives you measurable junk rejection in hours. Upgrade to a small CNN on the cropped patches if time allows.

---

## 5. Who does what (plays to each person)

**Bindu (CS) — the brain + the app.**
- Picking-engine integration (run Topaz/CryoSegNet, parse boxes).
- The **junk classifier** (the novel ML piece) + active-learning update logic.
- The interactive UI (image canvas, boxes, click-to-reject, live metrics).
- Metrics code (precision/recall/F1 vs CryoPPP ground truth).

**Tony (hardware/systems) — make it real-time and reliable.**
- Environment + GPU setup on the venue workstation; CUDA/PyTorch; Docker image.
- **MRC I/O + preprocessing** (normalize, downsample to 8-bit images, tiling).
- The **stream simulator** (feed micrographs on a timer to mimic a live session) + throughput dashboard (micrographs/min, GPU use).
- Performance: batching, caching, keeping per-micrograph inference to a few seconds.

**Eva (chemistry) — the science, the user, the story.**
- Choose the demo protein(s); know what a *real* particle vs ice/carbon/aggregate looks like.
- Be the **human-in-the-loop** in the demo (does the corrections that make the model improve).
- Run/interpret the **2D class averages**; confirm they look like real protein views.
- Lead the scientific framing + slides + the "why this matters for structure determination" narrative.

> This split also nails your **"professional background variety"** criterion — three genuinely different disciplines, each owning a layer.

---

## 6. Pre-hackathon checklist (do BEFORE June 27 — this is what wins it)

The mixers (Apr 27 UCB / May 14 UCSC / Jun 11 UCSF) and the weeks between exist for exactly this. Walking in cold = losing a day to setup.

- [ ] **Download data early (Tony):** 2–3 CryoPPP proteins via the metadata sheet (NOT the full 2.6 TB). Verify you can read MRC with `mrcfile` and overlay ground-truth coordinates.
- [ ] **Stand up the environment (Tony + Bindu):** conda env with PyTorch+CUDA, Topaz installed and running inference on **one** micrograph end-to-end.
- [ ] **Baseline numbers (Bindu):** run Topaz on a held-out set, compute precision/recall vs CryoPPP labels → this is the number you'll beat.
- [ ] **Junk labels ready (Bindu + Eva):** extract CryoPPP's keep-vs-false-positive labels into a simple training table.
- [ ] **UI skeleton (Bindu):** a page that loads a micrograph + draws boxes from a coordinate file. (Everything else plugs into this.)
- [ ] **Pick the hero protein (Eva):** decide the demo dataset; pre-make a "known good" 2D class average so you know the target picks are real.
- [ ] **Pre-compute the heavy stuff:** cache embeddings / baseline picks for the demo set so day-of is about the *novel* interactive + junk + real-time layers.

---

## 7. Hour-by-hour 48-hour plan

**Day 1 — Saturday (build the spine, in priority order so you always have a demo)**
- **9:30–10:30** Opening; finalize scope; confirm pre-loaded data + env works on the venue GPU.
- **10:30–13:00** Bindu: wire pretrained picker → boxes into the UI on the hero micrograph. Tony: MRC pipeline + stream simulator skeleton. Eva: finalize ground-truth overlay + define junk categories.
- **13:00–16:00** Bindu: **junk classifier v1** (random forest on CryoPPP keep/junk features) + show keep=green / junk=red overlay. Tony: per-micrograph inference under a few seconds; throughput counter. Eva: assemble the held-out test set.
- **16:00–18:00** **Milestone 1 (guaranteed result):** static picking + junk flags on one protein, with **precision/recall vs ground truth**. *If everything else fails, you can already present this.*
- **18:00–21:00** Add the **active-learning loop** (Eva rejects → classifier retrains live → metrics update). Tony: dockerize; back up everything.

**Day 2 — Sunday (novelty + polish + the wow)**
- **9:30–12:00** **Real-time mode:** stream micrographs through the full pipeline live; live dashboard (count, %junk, P/R, micrographs/min). **Milestone 2.**
- **12:00–14:00** **2D classification (ASPIRE)** on kept particles → class-average montage. **Milestone 3 (the wow).** Eva validates they look like real protein.
- **14:00–16:00** Generalization test on a 2nd protein; finalize the comparison chart (You vs Topaz: more true particles, less junk). Freeze features.
- **16:00–16:30** **Submit project (open-source repo, MIT).**
- **16:30–17:00** Dry-run the live demo twice (have a recorded backup video!). Present.

---

## 8. Risks & fallbacks (how you guarantee a result)

| Risk | Mitigation |
|---|---|
| Data/format plumbing eats the weekend | **Do all of it before the event** (checklist §6). Tony owns MRC/IO ahead of time. |
| A picker won't install / is slow | Use **Topaz** (well-supported); pre-cache its picks for the demo set. |
| Live retraining is unstable on stage | Keep the junk classifier **simple** (random forest) so updates are instant and predictable. |
| Real-time mode flaky | It's optional polish — Milestone 1 (static + metrics) is the floor. **Record a backup demo video.** |
| 2D classification too heavy | Use **ASPIRE** (light) or a custom rotational-align + k-means; it's the wow, not the core. |

**Fallback ladder (always have the next-lower rung working):**
1. Static picking + junk flags + precision/recall on one protein. ← minimum viable, must-have
2. + interactive correction that improves the metric.
3. + real-time streaming dashboard.
4. + 2D class-average "wow."

---

## 9. Slide outline (5 min)
1. **The pain:** auto-pickers drown you in junk; cleaning it is manual and slow. (Eva)
2. **Idea:** real-time picking *copilot* that removes junk and learns from you. (Bindu)
3. **What's new:** not a new picker — an interactive junk-triage + live-learning product on top of SOTA models. Honest nod to Topaz/CryoFSL. (Bindu)
4. **Live demo:** stream → pick → red/green junk flags → reject a cluster → metric jumps. (Eva drives, Tony narrates throughput)
5. **Results:** P/R/F1 vs Topaz; junk-rejection rate; 2D class averages prove picks are real. (Bindu)
6. **Why it matters + next steps:** faster, cleaner datasets → better structures; works on multiple proteins; open-source. (Eva)

---

## 10. Scorecard (your rubric)
| Impact | Execution | Team variety | Novelty |
|:---:|:---:|:---:|:---:|
| 5 — junk removal is a real, universal cryo-EM bottleneck | 4 — built on pretrained models + a simple, robust junk classifier; staged milestones guarantee a demo | 5 — hardware + chemistry + CS each own a layer | 4 — picking models exist, but an interactive real-time junk-triage + live-learning copilot does not |

---

## Sources
- CryoPPP dataset — [paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10287764/) · [GitHub](https://github.com/BioinfoMachineLearning/cryoppp) · [metadata xlsx](http://calla.rnet.missouri.edu/cryoppp/EMPIAR_metadata_335.xlsx)
- CryoFSL (SAM2 few-shot picking, 2025) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.19.677446) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458156/)
- CryoSegNet (SAM + U-Net) — [Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/4/bbae282/7690949)
- Topaz picker — [GitHub](https://github.com/tbepler/topaz) · SAM-for-cryoEM prompt study — [arXiv 2311.16140](https://arxiv.org/pdf/2311.16140)
- EMPIAR archive — [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) · ASPIRE-Python — [GitHub](https://github.com/ComputationalCryoEM/ASPIRE-Python) · mrcfile — [GitHub](https://github.com/ccpem/mrcfile)
