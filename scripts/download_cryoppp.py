"""Helper notes for downloading a SMALL CryoPPP/EMPIAR subset (do before June 27).

CryoPPP is 2.6 TB in total — DO NOT pull all of it. Grab 2-3 EMPIAR protein sets.

Steps:
  1. Open the CryoPPP metadata (per-EMPIAR Globus/FTP paths):
       http://calla.rnet.missouri.edu/cryoppp/EMPIAR_metadata_335.xlsx
  2. Pick IDs (start with 10025 = T20S proteasome). Note their download paths.
  3. Pull raw micrographs + ground-truth coords for ONLY those IDs into:
       data/raw/<EMPIAR_ID>/micrographs/   and   data/raw/<EMPIAR_ID>/ground_truth/
  4. Confirm with: python scripts/run_baseline.py --empiar 10025

EMPIAR browse/download: https://www.ebi.ac.uk/empiar/
CryoPPP repo (scripts + structure): https://github.com/BioinfoMachineLearning/cryoppp

This file intentionally does not auto-download (paths/credentials vary); it
documents the exact steps so the whole team can reproduce the setup.
"""

if __name__ == "__main__":
    print(__doc__)
