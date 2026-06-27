# Datasets & Resources

> **Download a SMALL subset (2–3 proteins, tens of GB) — NOT the full 2.6 TB.** Do this BEFORE the event.

## Primary — labeled ground truth (project backbone)
**CryoPPP** — expert-labeled cryo-EM dataset: 34 EMPIAR protein sets, ~9,893 micrographs with ground-truth particle coordinates **AND labeled false positives (ice, carbon edges)** — which is exactly what powers our junk classifier.
- GitHub (scripts + structure): https://github.com/BioinfoMachineLearning/cryoppp
- Paper (open): https://pmc.ncbi.nlm.nih.gov/articles/PMC10287764/
- Per-EMPIAR download paths (metadata): http://calla.rnet.missouri.edu/cryoppp/EMPIAR_metadata_335.xlsx

## Raw micrographs
**EMPIAR** — https://www.ebi.ac.uk/empiar/
Good high-contrast demo candidates (confirm exact IDs vs the CryoPPP list):
- **EMPIAR-10025** — T20S proteasome (forgiving "hero" demo)
- **EMPIAR-10017** — β-galactosidase
- **EMPIAR-10081** — HCN1 channel
- **EMPIAR-10028** — Plasmodium 80S ribosome
- **EMPIAR-10005** — TRPV1
Pick ONE large, high-contrast protein for the live demo; keep a 2nd/3rd for "it generalizes."

## Pretrained pickers (engine + baseline — don't train from scratch)
- **Topaz** (baseline to beat): https://github.com/tbepler/topaz
- **CryoSegNet** (SAM + U-Net): search GitHub `BioinfoMachineLearning/cryosegnet`
- **CryoFSL** (SAM2 few-shot; cite + optionally build on): https://www.biorxiv.org/content/10.1101/2025.09.19.677446 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12458156/
- **crYOLO**: https://cryolo.readthedocs.io
- **SAM2** (optional prompt route): https://github.com/facebookresearch/sam2

## Supporting libraries
- **mrcfile** — MRC IO: https://github.com/ccpem/mrcfile
- **starfile** — .star coordinate IO: https://github.com/teamtomo/starfile
- **ASPIRE-Python** — lightweight 2D classification for the "wow": https://github.com/ComputationalCryoEM/ASPIRE-Python

## File formats you'll touch
- `.mrc` / `.mrcs` — micrographs / particle stacks (use `mrcfile`)
- `.star` — coordinates + metadata (use `starfile`)
- `.box` — crYOLO/EMAN box coordinates (x, y, w, h)
- `.cs` — cryoSPARC (only if a teammate has cryoSPARC access)

## Download steps (do before June 27)
1. Open the CryoPPP metadata xlsx; pick 2–3 EMPIAR IDs (start with 10025).
2. Use the listed Globus/FTP paths to pull raw micrographs + ground-truth coords for those IDs only.
3. Put raw files in `data/raw/<EMPIAR_ID>/`; see `data/README.md` for the expected layout.
4. Run `python scripts/run_baseline.py` to confirm you can read MRC + overlay ground truth + get a Topaz baseline number.
