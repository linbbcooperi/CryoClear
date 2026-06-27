# Demo Script & Slide Outline (5 minutes)

## Live demo flow (rehearse twice; record a backup video!)
1. **Load the hero micrograph** (e.g., T20S proteasome). Show raw, noisy image.
2. **Run picking** → boxes appear. Toggle the junk classifier → boxes split **green (keep)** vs **red (junk: ice/carbon/aggregate)**.
3. **Eva rejects a junk cluster** with one click → on-screen **precision/recall updates and improves** (active learning).
4. **Switch to real-time mode** → micrographs stream in; live dashboard ticks: running particle count, % junk, micrographs/min (Tony narrates throughput on the single GPU).
5. **The "wow":** show **2D class averages** of the kept particles — clean, recognizable protein views = proof the picks are real.
6. **Results slide:** bar chart — Us vs Topaz: more true particles, far less junk.

## Slides
1. **The pain** — auto-pickers drown you in junk; cleaning it is manual/slow. (Eva)
2. **Idea** — real-time picking *copilot* that removes junk and learns from you. (Bindu)
3. **What's new** — not a new picker; an interactive junk-triage + live-learning product on top of SOTA. Honest nod to Topaz/CryoFSL. (Bindu)
4. **Live demo** — the flow above. (Eva drives, Tony narrates)
5. **Results** — P/R/F1 vs Topaz; junk-rejection rate; 2D class averages. (Bindu)
6. **Why it matters + next steps** — cleaner datasets → better structures; multi-protein; open-source MIT. (Eva)

## Talking points for judges
- Lead with the scientist's pain (category fit + impact).
- Say the honest novelty line: "We didn't reinvent picking; we made it interactive, real-time, and junk-aware."
- End on publishable/reusable (QBI offers long-term support toward a paper).
