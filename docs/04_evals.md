# Evals

The full evaluation spec (technical metrics + hackathon judging rubric + acceptance
gates) lives next to the harness: **[`../eval/EVALS.md`](../eval/EVALS.md)**.

Quick reference:
- Technical: picking **P/R/F1** vs CryoPPP ground truth · **junk-rejection P/R/F1** · throughput (micrographs/min) · 2D class-average wow. **Beat the Topaz baseline.**
- Judges' rubric: **Impact · Execution · Professional-background variety · Novelty.**
- Run: `python eval/run_eval.py --demo` (synthetic) or `--pred ... --gt ... --particle-radius ...`.
