# data/

**Nothing here is committed** (see `.gitignore`). Download a small CryoPPP/EMPIAR subset.

## Get it (one command)
```bash
# labeled set for metrics (has CryoPPP ground truth) — default 10017 β-gal:
uv run python scripts/download_cryoppp.py --source cryoppp --empiar 10017 --n-micrographs 15
# unlabeled stream/visual demo (no ground truth) — 10025 T20S:
uv run python scripts/download_cryoppp.py --source empiar-averaged --empiar 10025 --n-micrographs 3
```
The script converts CryoPPP's per-micrograph coordinate **`.csv`** → RELION **`.star`** automatically
(`coords.csv_to_star`) and prints the median particle diameter (set `--radius`/`DEMO_PARTICLE_DIAMETER_PX`).

> ⚠️ **EMPIAR-10025 (T20S) is not in CryoPPP** → no ground-truth labels → no picking metric. Use a
> CryoPPP-labeled protein for M1: **10017** (β-gal), **10005** (lightest, ~14 GB), or 10081 (TMV).
> Keep 10025 only as the unlabeled live-stream micrograph.

## Expected layout
```
data/
├── raw/
│   └── <EMPIAR_ID>/                # e.g. 10017
│       ├── micrographs/*.mrc       # micrographs
│       └── ground_truth/*.star     # expert particle coords (converted from CryoPPP .csv)
└── processed/
    └── <EMPIAR_ID>/
        ├── cryosegnet/*.star       # cached CryoSegNet picks (scripts/run_cryosegnet.py)
        └── junk_classifier.joblib  # trained junk model (scripts/train_junk_classifier.py)
```

## Coordinate convention (used by metrics + features)
- Particle coordinates are `(x, y)` centers in **pixels** in the micrograph frame.
- A predicted particle counts as a true positive if it is within `particle_radius` px of an
  unmatched ground-truth particle (greedy nearest-neighbor). See `src/cryoclear/metrics.py`.

## Where to get it
See `docs/03_datasets.md`. Start with EMPIAR-10025 (T20S proteasome) via the CryoPPP metadata sheet.
