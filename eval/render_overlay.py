"""Render a real micrograph with green=keep / red=junk overlay to a PNG (visual proof).

  uv run python eval/render_overlay.py --empiar 10017 --index 0 --out /workspace/proof.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cryoclear import config, features, io_mrc, picker  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--backend", default="blob")
    ap.add_argument("--out", default="/workspace/proof.png")
    ap.add_argument("--dpi", type=int, default=95)
    ap.add_argument("--crop", default="", help="x0,y0,size in display px for a zoomed view")
    args = ap.parse_args()

    raw = config.RAW / args.empiar
    mics = sorted((raw / "micrographs").glob("*.mrc"))
    mic = mics[args.index]
    img = io_mrc.load_for_pipeline(mic, factor=args.factor)
    imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))

    if args.backend == "cryosegnet":
        pf = picker.pick(imgf, backend="cryosegnet", name=mic.name,
                         cache_dir=config.PROCESSED / args.empiar / "cryosegnet")
        pd = pf / args.factor
    else:
        pd = picker.pick(img, backend="blob")
        pf = pd * args.factor

    clf = JunkClassifier.load(config.PROCESSED / args.empiar / "junk_classifier.joblib")
    feats = features.extract_features(imgf, pf, box=config.DEMO_PARTICLE_DIAMETER_PX)
    isj = np.asarray(clf.predict_is_junk(feats), dtype=bool)

    s_keep, s_junk = 24, 9
    if args.crop:
        x0, y0, sz = (int(v) for v in args.crop.split(","))
        img = img[y0:y0 + sz, x0:x0 + sz]
        m = ((pd[:, 0] >= x0) & (pd[:, 0] < x0 + sz)
             & (pd[:, 1] >= y0) & (pd[:, 1] < y0 + sz))
        pd = pd[m] - np.array([x0, y0])
        isj = isj[m]
        s_keep, s_junk = 220, 90   # bigger markers in the zoomed view

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=args.dpi)
    ax.imshow(img, cmap="gray")
    k, j = pd[~isj], pd[isj]
    ax.scatter(j[:, 0], j[:, 1], s=s_junk, facecolors="none", edgecolors="#e74c3c",
               linewidths=0.6, label=f"junk ({len(j)})")
    ax.scatter(k[:, 0], k[:, 1], s=s_keep, facecolors="none", edgecolors="#2ecc71",
               linewidths=1.1, label=f"keep ({len(k)})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.4)
    ax.set_title(f"{mic.name}  [{args.backend}]", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("WROTE", args.out, Path(args.out).stat().st_size, "bytes",
          f"keep={len(k)} junk={len(j)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
