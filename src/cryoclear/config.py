"""Central paths and constants. Tweak DEMO_* for your hero dataset."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Demo dataset.
# IMPORTANT: M1 metrics need CryoPPP ground truth. EMPIAR-10025 (T20S) is NOT in
# CryoPPP (no labels), so the labeled hero is a CryoPPP protein. 10017 = β-gal.
# Other CryoPPP options: 10005 (lightest, ~14 GB), 10081 (TMV). See docs/03_datasets.md.
DEMO_EMPIAR_ID = "10017"          # β-galactosidase — labeled in CryoPPP (has ground truth)
DEMO_PARTICLE_DIAMETER_PX = 180   # approx; tune per dataset — the download script prints
                                  # the median Diameter from CryoPPP CSVs (used as 2*radius)

# Unlabeled dataset for the visual / streaming demo only (no CryoPPP ground truth):
STREAM_EMPIAR_ID = "10025"        # T20S proteasome — direct EMPIAR averaged micrographs

# Matching tolerance for picking metrics = particle radius in pixels
def particle_radius_px(diameter_px: int = DEMO_PARTICLE_DIAMETER_PX) -> float:
    return diameter_px / 2.0

# Junk categories CryoPPP-style
JUNK_CLASSES = ("ice", "carbon_edge", "aggregate")
KEEP_CLASS = "particle"
