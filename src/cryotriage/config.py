"""Central paths and constants. Tweak DEMO_* for your hero dataset."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Demo dataset (start with a forgiving, high-contrast protein)
DEMO_EMPIAR_ID = "10025"          # T20S proteasome
DEMO_PARTICLE_DIAMETER_PX = 180   # approx; tune per dataset (used as 2*radius for matching)

# Matching tolerance for picking metrics = particle radius in pixels
def particle_radius_px(diameter_px: int = DEMO_PARTICLE_DIAMETER_PX) -> float:
    return diameter_px / 2.0

# Junk categories CryoPPP-style
JUNK_CLASSES = ("ice", "carbon_edge", "aggregate")
KEEP_CLASS = "particle"
