# The Idea — CryoTriage Live

## One-liner
An open, real-time cryo-EM **picking copilot** that doesn't just *find* particles — it **throws out the junk** (ice, carbon edges, aggregates) as data streams in, and **gets smarter every time the scientist corrects it.**

## The problem
1. **Junk is the real bottleneck.** Best auto-pickers (Topaz) over-pick: ice crystals, carbon film edges, protein aggregates. Cleaning is slow, manual, post-hoc.
2. **Not interactive / not real-time for normal users.** Research models (Topaz, crYOLO, CryoSegNet, CryoFSL) are batch/CLI. Real-time platforms (cryoSPARC Live, Warp) use blob/template pickers and are heavy/closed.

## Honest prior art (our novelty baseline)
- **Topaz / crYOLO** — standard DL pickers (batch, CLI).
- **CryoSegNet (2024)** — SAM + U-Net, very accurate.
- **CryoFSL (Sept 2025)** — *already* a SAM2 few-shot picker (≥5 labeled micrographs). → "few-shot SAM2 picking" is taken; **we do not claim a new picker model.**
- **cryoSPARC Live / Warp** — real-time preprocessing, simple pickers, closed/heavy.

## Our novelty (the white space)
An **open, interactive, real-time picking + junk-triage copilot with a human-in-the-loop learning loop.** Three concrete pieces:
1. **A dedicated junk classifier** trained on the *false positives* CryoPPP labels (ice, carbon, aggregates). No mainstream picker ships an interactive junk-rejector. ← our measurable ML contribution.
2. **Human-in-the-loop active learning** — reject a few picks / one junk cluster → model updates live → accuracy visibly climbs.
3. **Real-time streaming on one GPU** — micrographs stream like a live session; pick + triage each in seconds; live quality dashboard.

## Defensible pitch
*"CryoFSL made few-shot picking accurate. We make picking + junk-cleaning interactive and real-time — and we measure it: more true particles, far less junk, live, on a single GPU."*

## Scorecard (team rubric)
| Impact | Execution | Team variety | Novelty |
|:---:|:---:|:---:|:---:|
| 5 | 4 | 5 | 4 |
