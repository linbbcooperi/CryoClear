"""Generate Topaz picks (.star, full-res) for an EMPIAR dataset using a pretrained
Topaz detector, so the backend can offer Topaz as a selectable picker (cached, like
CryoSegNet). Runs the Topaz CLI from the GPU venv that has torch (cs_bw_venv).

  uv run python scripts/run_topaz.py --empiar 10017            # default model + scale
  TOPAZ_BIN=/workspace/cs_bw_venv/bin/topaz uv run python scripts/run_topaz.py

Topaz preprocess -s SCALE downsamples by SCALE; we keep SCALE == the webcache factor
(4) so picks land in the display space, then ×SCALE back to full-res for the .star.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoclear import config  # noqa: E402

TOPAZ_BIN = os.environ.get("TOPAZ_BIN", "/workspace/cs_bw_venv/bin/topaz")


def write_star(path: Path, xy: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n")
        for x, y in xy:
            f.write(f"{x:.2f}\t{y:.2f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--scale", type=int, default=4)       # match webcache downsample factor
    ap.add_argument("--radius", type=int, default=10)     # NMS radius in downsampled px
    ap.add_argument("--model", default="resnet16_u64")    # bundled pretrained detector
    ap.add_argument("--min-score", type=float, default=-2.0)  # permissive → over-pick → triage
    ap.add_argument("--limit", type=int, default=0, help="cap number of micrographs (0 = all)")
    ap.add_argument("--workers", type=int, default=16, help="topaz preprocess parallelism")
    args = ap.parse_args()

    raw = config.RAW / args.empiar / "micrographs"
    mics = sorted(raw.glob("*.mrc"))
    if args.limit:
        mics = mics[:args.limit]
    if not mics:
        print(f"no micrographs in {raw}")
        return 1
    out = config.PROCESSED / args.empiar / "topaz"
    proc = config.PROCESSED / args.empiar / "topaz_proc"
    out.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    print(f"preprocess {len(mics)} micrographs (scale {args.scale}, {args.workers} workers)…", flush=True)
    subprocess.run([TOPAZ_BIN, "preprocess", "-s", str(args.scale), "--num-workers",
                    str(args.workers), "-o", str(proc) + "/"] + [str(m) for m in mics], check=True)
    procs = sorted(proc.glob("*.mrc"))
    picks_txt = out / "_all_picks.txt"
    print(f"extract with {args.model} (r={args.radius})…", flush=True)
    subprocess.run([TOPAZ_BIN, "extract", "-r", str(args.radius), "-m", args.model,
                    "-o", str(picks_txt)] + [str(p) for p in procs], check=True)

    rows: dict[str, list] = {}
    with open(picks_txt) as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)                                      # header: image_name x_coord y_coord score
        for line in r:
            if len(line) < 4:
                continue
            name, x, y, score = line[0], float(line[1]), float(line[2]), float(line[3])
            if score < args.min_score:
                continue
            rows.setdefault(name, []).append((x * args.scale, y * args.scale))

    total = 0
    for m in mics:
        xy = np.asarray(rows.get(m.stem, []), dtype=float).reshape(-1, 2)
        write_star(out / f"{m.stem}.star", xy)
        total += len(xy)
    print(f"topaz: wrote {len(mics)} .star files, {total} picks (>= {args.min_score}) -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
