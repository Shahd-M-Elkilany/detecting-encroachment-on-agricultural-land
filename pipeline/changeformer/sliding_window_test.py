"""ChangeFormer on sliding windows resized to LEVIR-CD's 0.5 m pixel size.

LEVIR-CD trains ChangeFormer on 256 x 256 crops at 0.5 m, i.e. 128 m across. This
cuts Sentinel-2 windows of the same ground size, 13 x 13 pixels at 10 m, and
resizes them to 256 x 256, so every object appears at the size the model learned.
Windows on the positive test sites are centred on the largest converted-farmland
clusters of the labelling rule; windows on the negative test sites on random
farmland away from any conversion. Writes one row per window.

    python sliding_window_test.py --code ChangeFormer-main --ckpt best_ckpt.pt \\
        --s2 ../../data/s2_tiles --tiles ../../data/tiles --sites test_sites.txt \\
        --out ../../results/changeformer_sliding_windows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from run_changeformer import load_model, predict, resize, to_rgb

WINDOW = 13  # pixels at 10 m = 130 m, about 256 pixels at 0.5 m
PER_SITE = 3


def windows(net, s2: str, tiles: str, site: str, label: str, rng) -> list[dict]:
    import rasterio
    from scipy import ndimage

    def read(folder, tag):
        with rasterio.open(os.path.join(folder, f"{site}_{tag}.tif")) as r:
            return np.nan_to_num(r.read())

    a, b = read(tiles, "before_2024"), read(tiles, "after_2025")
    sa, sb = read(s2, "before_2024"), read(s2, "after_2025")
    h, w = sa.shape[1:]
    a, b = a[:, :h, :w], b[:, :h, :w]
    converted = (a[0] > 0.25) & (b[0] < 0.25) & (b[1] > a[1] + 0.08)  # the labelling rule's pixels
    clusters, n = ndimage.label(converted)
    if label == "pos" and n:
        sizes = ndimage.sum(converted, clusters, range(1, n + 1))
        centres = [ndimage.center_of_mass(converted, clusters, i + 1) for i in np.argsort(-sizes)[:PER_SITE]]
        kind = "converted cluster"
    else:
        ys, xs = np.nonzero((a[0] > 0.25) & ~ndimage.binary_dilation(converted, iterations=8))
        centres = [(ys[i], xs[i]) for i in rng.choice(len(ys), PER_SITE, replace=False)]
        kind = "stable farmland"
    both = np.concatenate([sa[[2, 1, 0]].reshape(3, -1), sb[[2, 1, 0]].reshape(3, -1)], 1).astype(np.float32)
    valid = both.min(0) > 0
    lo, hi = np.percentile(both[:, valid], 2, 1), np.percentile(both[:, valid], 98, 1)
    ra, rb = to_rgb(sa, "pct", lo, hi), to_rgb(sb, "pct", lo, hi)
    rows = []
    for cy, cx in centres:
        y0 = int(np.clip(round(cy) - WINDOW // 2, 0, h - WINDOW))
        x0 = int(np.clip(round(cx) - WINDOW // 2, 0, w - WINDOW))
        crop = (slice(y0, y0 + WINDOW), slice(x0, x0 + WINDOW))
        prob = predict(net, resize(ra[crop], 256), resize(rb[crop], 256))
        rows.append(
            {
                "site": site,
                "label": label,
                "window": kind,
                "y0": y0,
                "x0": x0,
                "converted_px_in_window": int(converted[crop].sum()),
                "mean_prob": round(float(prob.mean()), 5),
                "changed_frac": round(float((prob > 0.5).mean()), 5),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--sites", required=True, help="text file of site names (the test sites)")
    parser.add_argument("--labels", default=str(Path(__file__).resolve().parents[2] / "data" / "labels.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    net = load_model(args.code, args.ckpt)
    labels = {r["site"]: r["label"] for r in json.loads(Path(args.labels).read_text())}
    rng = np.random.default_rng(0)
    rows = []
    for site in Path(args.sites).read_text().split():
        rows += windows(net, args.s2, args.tiles, site, labels[site], rng)
        print(site, labels[site], [r["changed_frac"] for r in rows[-PER_SITE:]], flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
