"""Download a SMALL labeled cryo-EM subset into the repo's folder structure.

Two sources:

  cryoppp        (DEFAULT, has ground truth — use this for M1 metrics)
    Pulls one CryoPPP per-EMPIAR tarball from the Missouri server, extracts a
    handful of micrographs + their ground-truth coordinate CSVs, and converts
    each CSV -> a RELION .star so it drops straight into the pipeline:
        data/raw/<id>/micrographs/<name>.mrc
        data/raw/<id>/ground_truth/<name>.star
    NOTE: CryoPPP has no per-file API, so the whole tarball downloads (14-19 GB);
    we then keep only --n-micrographs and delete the rest. Run this on the GPU
    box / a machine with fast network + disk, NOT a laptop.

  empiar-averaged   (NO ground truth — visual / streaming demo only)
    Fetches a few averaged micrographs directly from EBI EMPIAR over HTTPS.
    Used for the unlabeled stream demo (e.g. 10025 T20S). ~200 MB per file.

Examples:
  python scripts/download_cryoppp.py --source cryoppp --empiar 10017 --n-micrographs 15
  python scripts/download_cryoppp.py --source cryoppp --empiar 10005 --n-micrographs 10   # lightest
  python scripts/download_cryoppp.py --source empiar-averaged --empiar 10025 --n-micrographs 3
  python scripts/download_cryoppp.py --dry-run            # print the plan, download nothing

CryoPPP server : https://calla.rnet.missouri.edu/cryoppp/<id>.tar.gz
EMPIAR browse  : https://www.ebi.ac.uk/empiar/
CryoPPP repo   : https://github.com/BioinfoMachineLearning/cryoppp
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config, coords  # noqa: E402

CRYOPPP_URL = "https://calla.rnet.missouri.edu/cryoppp/{empiar}.tar.gz"
# Per-entry averaged sub-dir at EBI (varies per EMPIAR entry; 10025's is known):
EMPIAR_AVG_DIR = {
    "10025": "14sep05c_averaged_196",
}
EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/{empiar}/data/{subdir}/"


def _curl(url: str, out: Path, dry: bool, insecure: bool = False) -> None:
    cmd = ["curl", "-L", "--fail", "--retry", "3", "-o", str(out), url]
    if insecure:
        cmd.insert(1, "-k")
    print("  $", " ".join(cmd))
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)


def _download_cryoppp(empiar: str, n: int, dry: bool, insecure: bool) -> int:
    raw = config.RAW / empiar
    mic_dir, gt_dir = raw / "micrographs", raw / "ground_truth"
    tar_path = raw / f"_cryoppp_{empiar}.tar.gz"
    url = CRYOPPP_URL.format(empiar=empiar)

    print(f"[cryoppp] {url}\n          -> {mic_dir}  +  {gt_dir} (.star)\n")
    _curl(url, tar_path, dry, insecure)
    if dry:
        print("[dry-run] would extract micrographs + ground_truth, convert CSV->star, trim to", n)
        return 0

    mic_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    # tolerant extraction: grab any micrographs + coordinate CSVs in the archive
    subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(raw), "--wildcards",
         "*/micrographs/*", "*/ground_truth/*"],
        check=True,
    )

    mics = sorted(raw.rglob("micrographs/*.mrc"))[:n]
    if not mics:
        print("No micrographs found in the tarball — inspect its layout:")
        subprocess.run(["tar", "-tzf", str(tar_path)], check=False)
        return 1

    csvs = {p.stem: p for p in raw.rglob("*.csv")}
    kept, diam = 0, None
    for mic in mics:
        dest_mic = mic_dir / mic.name
        if mic.resolve() != dest_mic.resolve():
            dest_mic.write_bytes(mic.read_bytes())
        csv = csvs.get(mic.stem)
        if csv is None:
            print(f"  ! no coord CSV for {mic.name} (skipping labels)")
            continue
        n_part = coords.csv_to_star(csv, gt_dir / f"{mic.stem}.star")
        if diam is None:
            diam = coords.cryoppp_csv_diameter(csv)
        kept += 1
        print(f"  {mic.name}: {n_part} particles -> {mic.stem}.star")

    print(f"\nKept {kept} labeled micrographs in {mic_dir}")
    if diam:
        print(f"Median particle diameter ≈ {diam:.0f} px → set DEMO_PARTICLE_DIAMETER_PX / --radius {diam/2:.0f}")
    print(f"Tip: delete the {tar_path.name} tarball to reclaim space once you're happy.")
    print(f"Next: python scripts/run_baseline.py --backend blob --empiar {empiar}")
    return 0


def _download_empiar_averaged(empiar: str, n: int, subdir: str | None, dry: bool) -> int:
    subdir = subdir or EMPIAR_AVG_DIR.get(empiar)
    if not subdir:
        print(f"No known averaged sub-dir for EMPIAR-{empiar}; pass --empiar-subdir "
              f"(browse https://www.ebi.ac.uk/empiar/EMPIAR-{empiar}/).")
        return 1
    base = EMPIAR_BASE.format(empiar=empiar, subdir=subdir)
    mic_dir = config.RAW / empiar / "micrographs"
    print(f"[empiar-averaged] {base}\n                  -> {mic_dir}  (UNLABELED)\n")

    listing = subprocess.run(["curl", "-s", "--max-time", "60", base],
                             capture_output=True, text=True)
    import re
    names = re.findall(r'href="([^"]+\.mrc)"', listing.stdout)[:n]
    if not names:
        print("Could not list .mrc files — check the URL / sub-dir.")
        return 1
    print(f"  {len(names)} files (~200 MB each):", ", ".join(names[:3]), "...")
    for name in names:
        _curl(base + name, mic_dir / name, dry)
    if not dry:
        print(f"\nDownloaded {len(names)} micrographs to {mic_dir} (no ground truth — stream/visual demo).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["cryoppp", "empiar-averaged"], default="cryoppp")
    ap.add_argument("--empiar", default=None,
                    help="EMPIAR id (default: DEMO for cryoppp, STREAM for empiar-averaged)")
    ap.add_argument("--n-micrographs", type=int, default=15)
    ap.add_argument("--empiar-subdir", default=None, help="averaged sub-dir name (empiar source)")
    ap.add_argument("--insecure", action="store_true", help="curl -k (skip TLS verify if cert errors)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.source == "cryoppp":
        empiar = args.empiar or config.DEMO_EMPIAR_ID
        return _download_cryoppp(empiar, args.n_micrographs, args.dry_run, args.insecure)
    empiar = args.empiar or config.STREAM_EMPIAR_ID
    return _download_empiar_averaged(empiar, args.n_micrographs, args.empiar_subdir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
