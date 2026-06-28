"""M4: 2D class averages of KEPT particles — the cryo-EM "these picks are real" proof.

Extract kept particles (after junk triage), run reference-free 2D classification
(cryoclear.class2d), render a montage of class averages.

  uv run python eval/class_averages.py --empiar 10017 --out /workspace/classes.png
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

from cryoclear import class2d, config, features, io_mrc, picker  # noqa: E402
from cryoclear.junk_classifier import JunkClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empiar", default=config.DEMO_EMPIAR_ID)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--box", type=int, default=160, help="crop box in full-res px")
    ap.add_argument("--size", type=int, default=64, help="resized particle size")
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--max-particles", type=int, default=350)
    ap.add_argument("--out", default="/workspace/class_averages.png")
    args = ap.parse_args()

    raw = config.RAW / args.empiar
    clf = JunkClassifier.load(config.PROCESSED / args.empiar / "junk_classifier.joblib")
    mics = sorted((raw / "micrographs").glob("*.mrc"))

    crops_all, n = [], 0
    for mic in mics:
        imgf = io_mrc.normalize_8bit(io_mrc.load_mrc(mic))
        img = io_mrc.load_for_pipeline(mic, factor=args.factor)
        pf = picker.pick(img, backend="blob") * float(args.factor)
        feats = features.extract_features(imgf, pf, box=config.DEMO_PARTICLE_DIAMETER_PX)
        kept = pf[~np.asarray(clf.predict_is_junk(feats), dtype=bool)]
        crops = class2d.extract_particles(imgf, kept, box=args.box, out_size=args.size)
        crops_all.append(crops)
        n += len(crops)
        if n >= args.max_particles:
            break

    stack = np.concatenate(crops_all)[:args.max_particles]
    print(f"kept particles for 2D classification: {len(stack)}")
    avgs, labels, counts = class2d.classify_2d(stack, n_classes=args.n_classes)
    mont = class2d.montage(avgs, counts)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=80)
    ax.imshow(mont, cmap="gray")
    ax.set_title(f"2D class averages — {len(stack)} kept {args.empiar} particles "
                 f"→ {len(avgs)} classes  (counts {counts.tolist()})", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("WROTE", args.out, "counts", counts.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
