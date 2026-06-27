# C1 vs A2 — Deep Dive for QBI Hackathon 2026

*Two strong options compared on your rubric: Impact · Execution · Team variety · Novelty*

A quick honesty note up front: I checked what already exists. **Both ideas have real prior art.** That's not a reason to drop them — it's a reason to aim at the *specific gap* the existing tools leave open. The novelty points come from the angle, not the category. Below, each idea is broken into: the plain idea → what already exists → the creative angle that's actually new → the 48-hour build → team → scorecard.

---

## C1 — The "talk to your microscope images" analysis assistant

### The plain idea (easy version)
A biologist has a folder of microscopy images and a question — "how many cells are dividing in the treated dish vs the control?" Today they either learn to code, or spend hours clicking in ImageJ. With C1, they **type the question in plain English**, and an AI agent picks the right tools, runs them, and hands back the answer: a number, a chart, and a table — plus the code it used, so it's reproducible.

Think of it as a lab assistant who knows every image-analysis tool and does the boring parts for you.

### What already exists (be honest — this is your novelty baseline)
This category is **already published**, and notably by a *Bay Area lab next door*:
- **Omega / napari-chatGPT** — Loic Royer's group at CZ Biohub SF, published in **Nature Methods (2024)**. You type "segment the nuclei" and it writes + runs the code inside napari. Some judges may personally know this work.
- **BIA Bob (bia-bob)** — Robert Haase's LLM assistant for bioimage analysis in Jupyter notebooks.

So "an LLM that turns plain English into image-analysis code" is **not new**. If you just rebuild that, you score low on novelty (and judges who know Omega will notice).

### The creative angle that IS still open
Omega and BIA Bob are **single-image, code-generation copilots** — they help one expert write a script faster. They do *not* do the things a scientist actually needs to publish a result. That gap is your novelty:

1. **Experiment-level reasoning, not one image at a time.** Your agent reasons over a *whole experiment* — many images, multiple conditions, replicates. It runs the pipeline in batch, then does the **statistics** (treated vs control, n per group, p-value, effect size) and draws the comparison plot. Existing tools stop at "here's the segmentation of this image." Yours answers "is the difference real?"
2. **A trust loop (human-in-the-loop confidence).** Before trusting the count, the agent shows you the **3–5 borderline cells** it was unsure about and asks you to confirm/reject. It then adjusts. This turns a black box into something a scientist will actually believe — and no existing tool does this cleanly.
3. **Reproducibility as the output.** The deliverable isn't just a number — it's a **ready-to-share package**: the exact code, the parameter log, and a **drafted "Methods" paragraph** for their paper. "Plain English in → publishable, reproducible analysis out." That's a genuinely fresh framing.
4. **Specialize to one real UCSF assay.** Instead of "works on any image" (which is Omega's pitch), make it *great* at one concrete, common task a partner lab actually does — colony counting, organoid sizing, neurite outgrowth, wound-healing assays. Narrow + excellent beats broad + shaky in a 48-hour demo.

> **One-line pitch that wins:** *"Omega helps an expert write code for one image. Ours takes a non-coder from a folder of images to a statistically-tested, reproducible, paper-ready result — and shows its work so you can trust it."*

### What you'd actually build in 48 hours
- **Under the hood (don't reinvent):** Cellpose 3 / Stardist (segmentation), Segment Anything (interactive objects), scikit-image (measurements), pandas + SciPy (stats). These already work — you're orchestrating, not training.
- **The agent layer:** an LLM with a *fixed, small toolbox* of those functions (constrained tool-calling = reliable, not a free-for-all). It plans → runs → reports.
- **The UI:** a simple web page or a napari panel: drop images, type request, see result + borderline-cell confirmations + downloadable code/methods.
- **The demo (3 min):** drag in a treated/control image set → type "compare cell counts" → confirm 3 borderline cells → out comes a bar chart with a p-value + a downloadable notebook and Methods paragraph.

### Team you'd want
A microscopist with a real assay + a folder of images · one ML/LLM engineer (tool-calling, glue) · one UI/full-stack generalist. Easy to staff — almost every lab has light microscopy, so you'll find a scientist at any mixer.

### Risks
- Agent reliability — mitigate by **constraining the toolbox** (don't let it free-write arbitrary code in the demo).
- Looking like an Omega clone — mitigate by leaning hard into the **batch + stats + trust-loop + reproducibility** angle and *citing Omega* (shows you did your homework).

### Scorecard (refined, honest)
| Impact | Execution | Team variety | Novelty |
|:---:|:---:|:---:|:---:|
| 5 — every bio lab has images | 4 — builds on working tools | 4 — broad, easy to staff | 3 — category exists; your angle (experiment-level + trust + reproducibility) is the new part |

---

## A2 — The friendly cryo-EM particle-picking tool

### The plain idea (easy version)
In cryo-EM, scientists photograph thousands of frozen protein molecules, then must find and circle every molecule ("particle") in noisy, grainy images before a 3D structure can be built. Doing this well is slow and fiddly. A2 is a **friendly tool that finds the particles for you** — and lets you guide it by example instead of fighting command-line software.

### What already exists (your novelty baseline)
Particle picking is a **mature ML field**:
- **Topaz** (positive-unlabeled learning) and **crYOLO** (YOLO object detection) are the standard deep-learning pickers.
- **CryoSegNet (2024)** already combines **Segment Anything (SAM) + a U-Net** and beats Topaz/crYOLO on resolution.
- **cryoSAM / CRISP (2025)** push SAM-based and segmentation approaches further.

So "use a deep model (even SAM) to pick particles" is **already done**. Rebuilding a picker model won't score on novelty. The known pain points the literature itself flags: **crYOLO misses real particles; Topaz over-picks junk and duplicates**, which clogs storage and later steps.

### The creative angle that IS still open
The *models* are solved; the **interaction and workflow** are not. That's your gap:

1. **Few-shot "click one, get all."** The user clicks **2–3 example particles** on a single image; the tool uses SAM-style similarity to **propagate the picks across all micrographs** — no training run, no parameter wrangling. Fast, visual, few-shot, human-guided picking is a UX almost no mainstream tool offers well.
2. **Live junk triage.** Immediately group the picked particles into rough visual classes and let the user **delete whole "junk" classes in one click** — directly attacking Topaz's over-picking problem the moment it happens, not three steps later.
3. **Active-learning loop.** Model picks → human corrects a handful → model updates live → precision climbs in front of the judges. A visible "it learns from me" moment is a great demo.
4. **Aim at the harder, whiter space: cryo-ET (tomography).** Single-particle picking is crowded; **subtomogram picking in cryo-electron *tomography*** is far less solved and a real UCSF strength. Same UX idea, much more novelty and impact.
5. **Picking + an agent copilot (bridges into C1).** Wrap it with an assistant that reports "you picked ~42,000 particles, ~15% look like junk, estimated quality X, here's the next step." This fuses A2's depth with C1's friendliness.

> **One-line pitch that wins:** *"Topaz and CryoSegNet made picking accurate. We make it interactive — click two particles, watch it pick the rest, throw out the junk in one click, and it gets smarter as you correct it."*

### What you'd actually build in 48 hours
- **Under the hood:** a pretrained picker/segmenter (SAM or a published cryo-EM model) for inference only — a 2080 Ti handles inference fine. Read MRC micrographs, output particle coordinates (.star/.box).
- **The UI:** an image viewer where you click examples, see picks appear, brush away false positives, and view the junk-class triage panel.
- **The demo (3 min):** load real micrographs → click 2 particles → watch hundreds get picked → one-click remove a junk class → correct a few → accuracy visibly improves → export coordinates ready for cryoSPARC/RELION.

### Team you'd want
A **cryo-EM scientist** with a real micrograph dataset (scarcer skill — but UCSF/QBI is a world cryo-EM hub, so very findable) · an ML engineer comfortable with SAM/PyTorch + the MRC/.star file formats · one UI person. More specialized than C1, which means fewer people *can* help — but also means judges who know how hard cryo-EM is will be impressed.

### Risks
- **Data dependency is the big one** — you must line up real micrographs (and ideally a known "answer") *before* the weekend. Without data, this idea stalls.
- File-format plumbing (MRC, .star) eats time — assign it to someone early.
- If you stay in single-particle picking, novelty is only "medium" (UX layer); pushing toward cryo-ET raises novelty but also difficulty.

### Scorecard (refined, honest)
| Impact | Execution | Team variety | Novelty |
|:---:|:---:|:---:|:---:|
| 5 — acute pain, high-value, UCSF-core | 3–4 — depends on having real data ready | 4 — specialized but UCSF-rich; impressive mix | 3–4 — models exist; interactive few-shot UX (and cryo-ET) is the new part |

---

## How to choose between them

| If you... | Lean toward |
|---|---|
| Can recruit a **microscopy** scientist (easy) and want the **safest path to a finished, broadly-useful demo** | **C1** |
| Can recruit a **cryo-EM** scientist **with a dataset ready** and want the **higher-prestige, more technically impressive** project judges will respect | **A2** |
| Want the **highest novelty ceiling** | **A2 via cryo-ET**, or **C1 with a hard lean into experiment-level stats + trust loop** |
| Want the **broadest impact / easiest to staff** | **C1** |
| Want to **wow a panel that includes cryo-EM-savvy UCSF scientists** | **A2** |

**My honest read:** 
- **C1 is the lower-risk, higher-reach choice.** More people can join, you'll definitely have a working demo, and the experiment-level + reproducibility + trust-loop angle is a real, defensible difference from Omega. Its ceiling is slightly capped by that prior art.
- **A2 is the higher-prestige, higher-variance choice.** cryo-EM is a UCSF crown jewel, the pain is real, and the interactive few-shot UX feels fresh — *but it lives or dies on having a real micrograph dataset and a cryo-EM teammate locked in before the event.*

The decision really hinges on **one question: which scientist can you actually get on your team at the mixers — a microscopist (→ C1) or a cryo-EM person with data (→ A2)?** The idea should follow the teammate, because "real scientist + real data" is the single biggest predictor of winning here.

---

## Sources
- Omega / napari-chatGPT — [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02310-w) · [GitHub](https://github.com/royerlab/napari-chatgpt) · [napari hub](https://napari-hub.org/plugins/napari-chatgpt.html)
- BIA Bob — [GitHub](https://github.com/haesleinhuepf/bia-bob)
- AI/LLMs in microscopy overview — [Janelia BioImaging Guide](https://bioimagingai.janelia.org/3-llms.html)
- CryoSegNet (SAM + U-Net particle picking) — [Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/4/bbae282/7690949) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11165428/)
- SAM prompt-based picking — [arXiv 2311.16140](https://arxiv.org/pdf/2311.16140) · CRISP (2025) — [arXiv 2502.08287](https://arxiv.org/pdf/2502.08287)
