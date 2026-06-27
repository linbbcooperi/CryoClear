# QBI Hackathon — Last Year's Recap + Ideas for 2026

*Prepared for Bhargav · Research compiled June 27, 2026*

---

## 1. The event at a glance

**This year (2026):**
- **Dates:** June 27–28, 2026 (48 hours, Sat–Sun)
- **Location:** UCSF Mission Hall, Room 1400 (Mission Bay), 9:00 AM – 9:00 PM
- **Mixers (team formation — strongly recommended):**
  - April 27, 2026 — UC Berkeley
  - May 14, 2026 — UC Santa Cruz
  - June 11, 2026 — UCSF
- **Register:** Eventbrite + join the QBI Hackathon Slack workspace
- **Hosts:** UCSF/QBI with UC Berkeley and UC Santa Cruz (plus QB3)

**What they want you to build (unchanged theme for years):**
Take real biomedical data — **light microscopy, electron microscopy (cryoEM), MRI, X-ray, and mass-spec / proteomics** — and apply **computer vision, ML/DL** to it. Recurring explicit asks from the organizers:
- Automatically **segment** cells / organs / particles
- **Denoise** and **cluster** data
- Wrap clunky **command-line tools in a friendly GUI** so non-coders can use them

**Rules / logistics that shape strategy:**
- All submissions must be **open source under the MIT license**.
- **Judges** are a mix of scientists, developers, and **VCs** → a working, demo-able product matters more than a perfect model.
- Hardware provided: workstations with **4× RTX 2080 Ti GPUs** + some cloud compute. Plan for modest GPU memory, not a giant training run.
- Winning teams get **prizes + long-term QBI support to push the work toward a peer-reviewed publication** → pick something that can keep going.
- Ideal team ≈ 3 people: one scientist with data, one ML/CV developer, one generalist for UI/glue.

---

## 2. Last year (2025) recap

- **When:** March 8–9, 2025 · **Teams:** 10 (a record at the time)
- **Who came:** developers + scientists from UCSF, UC Berkeley, UC Santa Cruz, and USF
- **Placements:** Cys-TEAM (1st), MaxBind-AI & DECODE (tied 2nd), Ideonella (3rd)

### The 10 projects

| # | Team | Placement | One-line summary | Core theme fit |
|---|------|-----------|------------------|----------------|
| 1 | **Cys-TEAM** | 🥇 1st | Web tool for pan-cancer proteomics: pick proteins of interest → get a minimal cell-line panel; plus a web-accessible cysteine enrichment (CSEA) tool | Proteomics ✅ (strong) |
| 2 | **MaxBind-AI** | 🥈 2nd (split) | Predict the optimal ligand's structural features + binding affinity for a given binding site | Drug/structure |
| 3 | **DECODE** | 🥈 2nd (split) | Drug repurposing: ESM + MolecularTransformer embeddings → predict drug–protein inhibition (IC50) from BindingDB | Proteomics/ML |
| 4 | **Ideonella** | 🥉 3rd | Optimize PETase (plastic-eating enzyme): ML to predict Km of mutants; ESM-2 to rank best variants | Protein ML |
| 5 | **Mind Over Data** | — | Real-time "mind reading" from a Muse 2 EEG; identify a number you visualize | EEG (off-core) |
| 6 | **E.L.O.R.A** | — | Muse EEG + AI to flag brainwave anomalies / emotional state in veterans (PTSD) | EEG (off-core) |
| 7 | **RM Fold** | — | Use sFold to find antisense oligos (ASOs) that bind a target gene; dockerized a Linux-only tool behind a web app | Genomics tooling |
| 8 | **EvoBeevos** | — | Variant-effect predictor built around the Evo 2 model; benchmark vs ClinVar/Ensembl; Streamlit + chatbot | Genomics ✅ |
| 9 | **HYPERA** | — | Multi-agent reinforcement learning to auto-tune hyperparameters for **2D microscopy image segmentation** | Microscopy ✅ (strong) |
| 10 | **SNP-automate** | — | Automate in-silico screening of SNPs in ABC transporters using 3D structure (H-bonds, clashes) for precision medicine | Structure ✅ |

### Per-project detail (from the GitHub repos)

**1. Cys-TEAM — 🥇 winner.** A real, polished product: React frontend + Firebase backend + Python cloud functions + an R Shiny app. It re-implements **CSEA (Cysteine Set Enrichment Analysis)** — originally from the Bar-Peled Lab's *DrugMap* paper — as an easy web tool, adapted by the **Zaro Lab (UCSF)**. Users upload a cysteine list, pick a cancer-tissue background, run permutation-based enrichment (Benjamini-Hochberg FDR), and get plots + CSV results. A second module lets users pick proteins of interest and generate a *minimal panel of cell lines* from a 59-cancer-cell-line proteomics dataset. *Repo: Zaro-Lab/CysTeam.*
→ **Why it won:** took a powerful but hard-to-use research tool + a real lab's dataset and made it genuinely usable, with a finished UI. This is the template.

**2. MaxBind-AI — 🥈 (split).** Goal: for a given binding site, predict the ideal ligand's structural features and expected binding affinity. Used PLINDER example data (myoglobin + cofactor) and ligand data queried from ChEMBL at various preprocessing stages. Repo is thin (mostly data prep) — strong idea, partial execution. *Repo: MatthewLaw1/MaxBind-AI.*

**3. DECODE — 🥈 (split).** Drug repurposing via embeddings: combine **ESM** (protein) + **MolecularTransformer** (drug) features to predict inhibitory efficacy, trained on **BindingDB**. A Random-Forest baseline + a GPU fully-connected net; reported **>80% accuracy** training on ~15% of BindingDB. Flask demo app. *Repo: benhuang3/QBI_Hackathon_2025.*

**4. Ideonella — 🥉 3rd.** Enzyme engineering to fight plastic pollution: predict the **Km** of mutated **PETase** sequences with XGBoost + Linear Regression, and use **ESM-2** to identify the best evolutionary candidates — cutting wet-lab cost. *(No public GitHub link provided.)*

**5. Mind Over Data.** A Muse 2 EEG "mind reading" demo — a recursive deep-learning approach that tries to read the number a user is visualizing in real time. Full stack: muselsl stream → Python backend → React frontend. Fun and demo-friendly but off the core imaging/proteomics theme. *Repo: matthewlaw1/qbi.*

**6. E.L.O.R.A.** Muse EEG + **CNN+RNN fusion** to detect brainwave anomalies and predict emotional state, aimed at veteran PTSD/mental-health diagnostics; saves results to Firebase. Includes an exploratory table mapping tumor types to EEG patterns. Ambitious clinical framing, hard to validate in 48h. *Repo: savir2010/Aurna.*

**7. RM Fold.** Practical engineering win: **sFold** (predicts antisense-oligo binding to a target gene) only runs on x86 Linux, so they deployed it to a cloud VM, wrapped it in a Node.js server, **dockerized** it for portability, and built a cleaner web app over the output. *Repo: michaelwaves/aso-backend (README sparse).*

**8. EvoBeevos.** A **Variant Effect Predictor** designed around the **Evo 2** genomics model, benchmarked against **ClinVar** and **Ensembl**, with a Streamlit UI + an AI chatbot. Honest caveat in their README: **they couldn't actually install Evo 2** (dependency hell), so they shipped everything *around* it. *Repo: tntly/qbi-evobeevos.*

**9. HYPERA.** The most directly on-theme microscopy project: a **multi-agent reinforcement-learning** system where each hyperparameter gets its own Q-learning agent that adapts *during* training, for **2D microscopy image segmentation** (MONAI + PyTorch). Very ambitious for a weekend; likely more framework than validated result. *Repo: anya-decarlo/HYPERA (code under /HYPERA1).*

**10. SNP-automate.** From a UCSF lab (Bajaj): automate **in-silico screening of SNPs** in ABC transporters by analyzing structural interactions (H-bonds, steric clashes, non-polar contacts) before vs after a mutation across conformations — a precision-medicine triage tool. *Repo: ishaan-awasthi/qbi-hackathon-2025.*

---

## 3. What actually wins here (pattern analysis)

Reading the results against the brief, three things separated winners from the rest:

1. **Finish a usable product, not a research idea.** Cys-TEAM and RM Fold both shipped working web tools. The flashier ML projects (HYPERA, EvoBeevos, MaxBind) had great ideas but ran out of runway. With VCs on the judging panel, a clean 3-minute demo beats a notebook.
2. **Anchor to a real UCSF lab + real dataset.** The winner was literally a lab's tool made usable. If you walk in with a scientist who has data and a pain point, you're already ahead. This is exactly what the mixers are for.
3. **Hit the stated core themes** — microscopy / cryoEM / imaging and proteomics/mass-spec. Notably, **nobody did cryoEM** in 2025, and only two teams (Cys-TEAM, HYPERA) hit the core squarely. Several drifted into consumer EEG, which is fun but off-brief. There's open space in the themes the organizers keep advertising.

---

## 4. Idea recommendations for 2026

### Option A — Build a *new* project in the winning lane (recommended)

**A1. "No-code microscopy lab" — segment + denoise + measure, in a browser.**
A clean web GUI that lets a biologist drag in a microscopy image stack and run modern foundation models — **Cellpose / Stardist** for cell & nucleus segmentation, **micro-SAM / Segment Anything** for interactive object picking, a denoiser (Noise2Void-style) — then export counts, areas, and measurements as CSV/plots. This *is* the organizers' headline ask ("automatically segment cells/organs/particles or denoise… generate a user-friendly interface"). Foundation models already exist, so a weekend is enough to wire them into a usable tool. High demo value, directly publishable as a lab utility.
*Fits 2080 Ti memory; pick a real lab's images at the mixer.*

**A2. "cryoEM helper" — particle picking / denoising GUI (white space).**
Nobody tackled cryoEM in 2025 even though it's explicitly invited, and UCSF is a cryoEM powerhouse. A friendly front end over an existing picker/denoiser (Topaz-style ML picking, or a SAM-based picker) for a real QBI cryoEM dataset would stand out simply because it's on-theme and uncontested.

**A3. "Proteomics results explorer" — instant dashboard for mass-spec output.**
A drop-in dashboard that ingests DIA-NN / FragPipe / MaxQuant output and gives interactive QC, volcano plots, and differential-abundance tables with a clean UI. This is the Cys-TEAM lane (proteomics + usability) without copying it, and proteomics scientists always need this.

### Option B — Improve a 2025 project (your "use last year's and improve it" option)

**B1. Finish EvoBeevos (highest "obvious win" potential).** They shipped everything except the actual model because **Evo 2 wouldn't install**. A 2026 team that delivers a *working* variant-effect predictor — Evo 2 properly containerized (Docker), **or** a swap to a reliably installable model like **ESM-1v / AlphaMissense / ESM-2** — plus the existing ClinVar/Ensembl benchmarking and clean report, has a clear, credible story: "we made the thing actually run." Low conceptual risk, strong narrative. *(MIT-license and model-license check needed.)*

**B2. Ground HYPERA into a usable segmentation tool.** The multi-agent RL idea is cool but heavy. Re-frame it as a practical **auto-tuning wrapper around Cellpose/MONAI segmentation** with a simple UI and a clear before/after metric on a real microscopy dataset. Keep the smart part, drop the parts that can't be validated in a weekend, add the usability that judges reward.

**B3. Extend Cys-TEAM / SNP-automate** only if you can partner with the original lab (Zaro Lab / Bajaj). These are real lab tools; the best version of "improving" them is joining forces, not forking solo.

### My pick if you want the best odds
**A1 (no-code microscopy segmentation/denoising tool)** as a fresh build, or **B1 (finish EvoBeevos)** if you'd rather improve an existing project. Both follow the winning pattern: on-theme, demo-able, built on existing models so you spend the 48h on usability and a real dataset rather than training from scratch.

---

## 5. Practical next steps

1. **Go to a mixer** (Apr 27 UCB / May 14 UCSC / Jun 11 UCSF) and recruit *one UCSF/UCB/UCSC scientist who has real data and a pain point* — this single move correlates most with winning.
2. **Lock the dataset before the weekend.** The org explicitly says to use mixer time to gather data. A real, ready dataset is the difference between a demo and a notebook.
3. **Scope to a finished demo.** Decide the one thing a judge will click and see work. Build backward from that.
4. **Use existing models** (Cellpose, SAM, ESM/AlphaMissense, Topaz) so the weekend goes to integration + UI, not training.
5. **Keep it MIT-licensed** and check that any model you depend on is redistributable.
6. **Aim for "publishable utility"** — winners get long-term QBI support toward a paper, so pick something a lab will actually keep using.

---

## Sources
- [QBI Hackathon 2025 results & project list](https://qbi.ucsf.edu/hackathon2025)
- [QBI Hackathon 2025 event detail](https://qbi.ucsf.edu/qbi-hackathon-2025)
- [QBI Hackathon (series + 2026 note)](https://qbi.ucsf.edu/hackathon)
- [QBI Hackathon 2026 details](https://qbi.ucsf.edu/hackathon-details-2026) · [2026 event page](https://qbi.ucsf.edu/events/hackathon-2026)
- [QBI Hackathon FAQs](https://qbi.ucsf.edu/hackathon-faq)
- Project repos: [Cys-TEAM](https://github.com/Zaro-Lab/CysTeam) · [MaxBind-AI](https://github.com/MatthewLaw1/MaxBind-AI) · [DECODE](https://github.com/benhuang3/QBI_Hackathon_2025) · [Mind Over Data](https://github.com/matthewlaw1/qbi) · [E.L.O.R.A](https://github.com/savir2010/Aurna) · [RM Fold](https://github.com/michaelwaves/aso-backend) · [EvoBeevos](https://github.com/tntly/qbi-evobeevos) · [HYPERA](https://github.com/anya-decarlo/HYPERA/tree/main/HYPERA1) · [SNP-automate](https://github.com/ishaan-awasthi/qbi-hackathon-2025)
