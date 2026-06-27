# A UCSF-Biochem-Anchored Idea for QBI Hackathon 2026

*Goal: closer to UCSF's core biochemistry/biophysics research, and more impactful, new, and innovative.*

## Why this direction fits UCSF (and the hosts)
UCSF biochemistry/biophysics is, above all, **structure-based**. A few things were literally invented there:
- **DOCK** — the original computer-aided drug-design (molecular docking) program, today led by **Brian Shoichet**.
- **ZINC** — the free ultra-large library of buyable molecules (ZINC-22 reached ~6.4 billion compounds by July 2024), and **ultra-large library docking** (Shoichet & Irwin, UCSF).
- **AMBER** — a foundational molecular-dynamics engine.

Add to that **Nevan Krogan / QBI** (the hackathon's host) mapping disease protein-interaction networks by mass spec, **James Fraser** showing proteins behave as *conformational ensembles* (not one fixed shape), and a world-class **cryo-EM/ET** program.

Last year's projects only grazed this core (Cys-TEAM's chemoproteomics, SNP-automate's structural screening). There's a wide-open lane to do something **central to UCSF biochemistry**.

---

# ⭐ Lead idea — "Hidden Door": finding cryptic druggable pockets in 'undruggable' proteins

**Anchored to:** James Fraser (protein ensembles) + Brian Shoichet (DOCK / ZINC druggability).

### The problem in one breath
AlphaFold gives you **one** picture of a protein. But proteins **breathe** — they wiggle between shapes. Roughly half of important disease proteins (think KRAS, transcription factors, protein–protein interfaces) look **"undruggable"** because their drug-binding pocket is **hidden in the static structure and only opens up when the protein moves**. These are called **cryptic pockets**. Finding them is one of the hottest frontiers in drug discovery — and it's exactly the kind of "proteins are ensembles" thinking UCSF is famous for.

### The idea (plain version)
A tool where you type in a protein, and it:
1. **Makes the protein "move"** — generates a small set of realistic alternate shapes (a conformational ensemble).
2. **Looks for pockets in every shape** — including ones that don't exist in the frozen AlphaFold/crystal structure.
3. **Flags the cryptic ones** — pockets that appear only when the protein flexes ("hidden doors").
4. **Checks if they're druggable** and **docks a few real, buyable molecules** (from ZINC) into the best hidden pocket.
5. **Hands back a clean visual report**: "This 'undruggable' target has a hidden pocket here — and here's a purchasable molecule that fits it."

> **The winning sentence:** *"AlphaFold said this target was undruggable. We showed it has a hidden door — and handed you a molecule you can buy this week to open it."*

### What already exists (your honest novelty baseline)
- **PocketMiner** — a neural network that predicts *where* cryptic pockets are likely to open (from a single structure).
- **fpocket / P2Rank** — standard pocket-finders and druggability scorers (P2Rank is ML-based).
- Papers on **AlphaFold MSA-subsampling** to generate alternate conformers, then running fpocket/P2Rank.

So the *pieces* exist in the literature. **What does NOT exist is a friendly, end-to-end tool** that takes a non-expert from "protein name" → ensemble → cryptic-pocket map → druggability → a docked, *purchasable* hit → shareable report, in one click. **Your novelty = the integration, the usability, and the docking-into-the-cryptic-pocket payoff** — none of the existing tools close that loop for a biologist.

### The creative / innovative angles (pick 1–2 to stand out)
- **Hybrid speed-up:** use PocketMiner as a fast "where to look" first pass, then *confirm* with an actual ensemble + docking. Fast *and* grounded.
- **Druggability + buyability together:** don't just say "there's a pocket" — return a **ZINC purchasable fragment** that fits it. That's the UCSF Shoichet signature (real, orderable molecules).
- **An LLM "explainer":** the agent narrates *why* the pocket is cryptic and *why* a hit binds, and drafts the methods text — turns a black box into something a biochemist trusts.
- **Allosteric angle:** many cryptic pockets are **allosteric** (regulatory) sites — pitch it as finding *new ways to control* a protein, not just block it.

### What you'd actually build in 48 hours
- **Make it move (light options that fit the hardware):** normal-mode analysis (ProDy, fast on CPU) and/or AlphaFold MSA-subsampling (heavier — **pre-run 1–2 demo targets before the event**, and the venue also provides cloud compute for heavier runs). A short OpenMM run is optional.
- **Find + score pockets:** fpocket + P2Rank (fast, CPU).
- **Flag cryptic:** simple logic — pockets strong in the ensemble but weak/absent in the static structure.
- **Dock:** smina / AutoDock Vina with a small set of ZINC molecules (a 2080 Ti is plenty for this).
- **Show it:** a web viewer (NGL/Mol*) highlighting the hidden pocket + the docked molecule, plus a downloadable report.
- **The 3-minute demo:** type a known "tough" target → AlphaFold pocket score looks bad → reveal the cryptic pocket in a flexed conformation → dock a buyable fragment → download the report.

### Hardware fit
Great. The heavy step (ensemble generation) can be **pre-computed for the demo targets** or run on the provided **cloud compute**; everything else (pocket finding, docking, the UI) runs comfortably on the 4×RTX 2080 Ti workstations.

### Team you'd want (hits your "background variety" criterion)
A **structural biologist / biochemist** (knows real targets and what a good pocket looks like) · a **computational chemist or ML person** (ensembles, docking, scoring) · a **UI/full-stack generalist** (viewer + report). Three genuinely different skill sets — easy to recruit at the UCSF/Berkeley mixers, and very on-brand for QBI.

### Scorecard (your rubric)
| Impact | Execution | Team variety | Novelty |
|:---:|:---:|:---:|:---:|
| 5 — opens "undruggable" targets; core drug-discovery value | 4 — built from working pieces; pre-compute the heavy step | 5 — biologist + comp-chem + dev | 4 — components exist, but the end-to-end usable "find a hidden pocket → buyable hit" tool does not |

---

# Two strong alternatives (same UCSF-core spirit)

## Alt A — "Interactome in 3D": turn a Krogan/QBI protein-interaction map into druggable structures
**Anchored to:** Nevan Krogan / QBI (the host) + modern complex prediction.

QBI maps which proteins touch which (by mass spec) — e.g., the famous SARS-CoV-2 and cancer interactomes. But those maps are flat node-and-edge diagrams with **no 3D structure**. The idea: take a list of interacting protein pairs and **predict each pair's 3D complex** (with open tools like **Boltz-1** or **Chai-1**), score how confident the interface is, and **flag interfaces that look druggable** or that carry known disease mutations. You turn a wiring diagram into a **structural, druggable map**.
- **New?** Structural annotation of interactomes is emerging research — doing it as a *usable tool on a real Krogan dataset* is fresh, and it's the most "home-crowd" idea possible (Krogan directs QBI).
- **Risk:** complex prediction is GPU-hungry; on a 2080 Ti you're limited to small complexes and it's slow. **Mitigate** by pre-computing a curated subnetwork and/or using the venue's cloud compute.
- **Scorecard:** Impact 5 · Execution 3 (compute risk) · Team variety 5 · Novelty 4.

## Alt B — "DOCK Copilot": friendly, interpretable ultra-large docking on ZINC
**Anchored to:** Shoichet's DOCK / ZINC-22 / RAD (all UCSF).

A web tool + agent: pick a target pocket → dock a **smart subset of buyable ZINC-22 molecules** (RAD-style, so you search billions without docking billions) → ML-rescore → an **LLM explains each top hit and whether you can buy it** → shortlist to order. It makes UCSF's world-leading-but-expert-only docking pipeline usable by a regular biologist.
- **New?** The *models* aren't new (docking is mature, and docking GUIs exist), so novelty rides on the **agentic, interpretable, "buy-this-molecule" triage** layer.
- **Risk:** low — very buildable with smina + RDKit + a ZINC subset. This is the **safest-to-finish** option.
- **Scorecard:** Impact 4 · Execution 5 · Team variety 4 · Novelty 3.

---

# Recommendation
- **For the best blend of "UCSF-core + impactful + new":** go with **"Hidden Door" (cryptic pockets).** It sits exactly where Fraser (ensembles) and Shoichet (druggability) meet, tackles the field's biggest prize (undruggable targets), and no friendly end-to-end tool exists yet.
- **If you can recruit someone from the Krogan/QBI orbit with a real interactome dataset:** **Alt A** is the highest "home-crowd" play (just plan for compute).
- **If you want the safest path to a polished demo:** **Alt B (DOCK Copilot)** — lower novelty, but it leans on UCSF's most famous software and will definitely work.

As before, the deciding factor is the teammate: a **structural/comp-chem scientist → "Hidden Door"**, a **Krogan-network scientist with data → Alt A**, a **med-chem/docking person → Alt B.** Lock the scientist + a demo target/dataset at a mixer, and the idea picks itself.

---

## Sources
- UCSF Biochemistry & Biophysics — [department site](https://biochemistry.ucsf.edu/) · [Biophysics research areas](https://biophysics.ucsf.edu/degree-program/research-areas)
- DOCK / ultra-large docking / ZINC (Shoichet, UCSF) — [Ultra-large library docking, Nature 2019](https://www.nature.com/articles/s41586-019-0917-9) · [ZINC-22, JCIM 2023](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01253) · [Retrieval-Augmented Docking (RAD), 2024](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00683) · [Ultra-large-scale docking overview](https://en.wikipedia.org/wiki/Ultra-large-scale_docking)
- Cryptic pockets — [PocketMiner (GNN)](https://www.researchgate.net/publication/368909120_Predicting_locations_of_cryptic_pockets_from_single_protein_structures_using_the_PocketMiner_graph_neural_network) · [Accelerating cryptic-pocket discovery with AlphaFold, JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01189) · [P2Rank](https://github.com/rdk/p2rank) · [Cryptic binding-site review, 2025](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504)
- Complex/binder tools — [BindCraft, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6) · Boltz-1 / Chai-1 (open complex predictors)
