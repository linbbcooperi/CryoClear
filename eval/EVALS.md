# EVALS — what we measure & how we win

Two layers: (A) the **technical metrics** that produce our "visible results," and
(B) the **hackathon judging rubric** every decision should point at.

## A. Technical metrics (run `python eval/run_eval.py`)
1. **Picking precision / recall / F1** vs CryoPPP expert ground truth.
   - A predicted particle is a TP if within `particle_radius` px of an unmatched GT particle (greedy match — see `src/cryoclear/metrics.py`).
   - **Target: beat the Topaz baseline** (run `scripts/run_baseline.py` to get it).
2. **Junk-rejection precision / recall / F1** (positive class = junk).
   - Uses CryoPPP's labeled false positives (ice, carbon edges, aggregates).
   - This is our novel contribution → show it explicitly.
3. **Throughput** — micrographs/min in real-time mode (from `stream.StreamStats`), to prove single-GPU viability.
4. **(Wow) 2D class averages** of kept particles (ASPIRE) — qualitative proof the picks are real.

Report as: a **bar chart Us vs Topaz** (more true particles, less junk) + a **class-average montage**.

### Acceptance gates (don't ship below these)
- M1: picking F1 reported on ≥1 protein with a Topaz baseline to compare against.
- M2: junk-rejection F1 improves after ≥1 round of user corrections (show before/after).
- M3: real-time loop sustains a non-trivial micrographs/min on one GPU.
- M4: ≥1 clean, recognizable 2D class average.

## B. Hackathon judging rubric (keep every decision aimed here)
| Criterion | Question | How we win it |
|---|---|---|
| **Impact** | Is it useful? | Junk removal is a universal cryo-EM bottleneck; cleaner data → better structures. |
| **Execution** | Is the product usable? | Working interactive GUI + live demo + real numbers, not a notebook. |
| **Professional-background variety** | Diverse team? | Hardware (Tony) + Chemistry (Eva) + CS (Bindu), each owns a layer. |
| **Novelty** | Has it been done? | Not a new picker (honest about Topaz/CryoFSL) — an open, interactive, real-time junk-triage + live-learning copilot, which doesn't exist. |

## Reproducibility
- Fix seeds; log `particle_radius`, picker backend, dataset IDs, and counts with every score.
- Keep `eval/run_eval.py --demo` green so the harness is always demonstrably working.
