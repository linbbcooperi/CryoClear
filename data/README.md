# data/

**Nothing here is committed** (see `.gitignore`). Download a small CryoPPP/EMPIAR subset locally.

## Expected layout
```
data/
├── raw/
│   └── <EMPIAR_ID>/                # e.g. 10025
│       ├── micrographs/*.mrc       # raw micrographs
│       └── ground_truth/*.star     # expert particle coords (+ junk labels if available)
└── processed/
    └── <EMPIAR_ID>/
        ├── images/*.png            # normalized 8-bit micrographs (from io_mrc)
        ├── candidates/*.csv        # picker output boxes
        └── features/*.csv          # per-candidate features for the junk classifier
```

## Coordinate convention (used by metrics + features)
- Particle coordinates are `(x, y)` centers in **pixels** in the micrograph frame.
- A predicted particle counts as a true positive if it is within `particle_radius` px of an
  unmatched ground-truth particle (greedy nearest-neighbor). See `src/cryotriage/metrics.py`.

## Where to get it
See `docs/03_datasets.md`. Start with EMPIAR-10025 (T20S proteasome) via the CryoPPP metadata sheet.
