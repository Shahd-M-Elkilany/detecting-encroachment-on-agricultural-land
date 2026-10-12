"""Run the published ChangeFormerV6 LEVIR-CD checkpoint on KEMET1 Sentinel-2 tile pairs.

Needs PyTorch, timm and einops, the ChangeFormer code (github.com/wgcban/ChangeFormer)
and its LEVIR-CD checkpoint ``best_ckpt.pt``.

    python run_changeformer.py sanity --code ChangeFormer-main --ckpt best_ckpt.pt
    python run_changeformer.py run --code ChangeFormer-main --ckpt best_ckpt.pt \\
        --s2 ../../data/s2_tiles --tiles ../../data/tiles --out ../../data/changeformer pct_x1 fixed_x1

``sanity`` checks the checkpoint on the LEVIR-CD sample pairs shipped with the code.
``run`` writes, per site, ``<site>.npz`` with one uint8 change-probability map per
variant, and appends a row per site and variant to ``changeformer_scores.csv``.

Variants are ``<stretch>_x<scale>``. Stretch ``fixed`` maps reflectance 0-0.3 to
0-1; ``pct`` uses the 2-98 % range of both dates together. Scale ``x1`` resizes the
262-pixel tile to one 256 x 256 window (ChangeFormer's training size); ``x2`` and
``x4`` upsample to 512 or 1024 and run 256-pixel windows, making objects look
larger, closer to the 0.5 m pixels of LEVIR-CD.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np


def load_model(code: str, ckpt: str):
    import torch

    sys.path.insert(0, code)
    from models.ChangeFormer import ChangeFormerV6

    net = ChangeFormerV6(embed_dim=256)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_G_state_dict"])
    return net.eval()


def predict(net, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: H x W x 3 float32 in [0, 1], sides multiples of 256. Returns the change probability."""
    import torch

    def tensor(x):
        return ((torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None] - 0.5) / 0.5).float()

    A, B = tensor(a), tensor(b)
    h, w = a.shape[:2]
    prob = np.zeros((h, w), np.float32)
    with torch.no_grad():
        for y in range(0, h, 256):
            for x in range(0, w, 256):
                out = net(A[..., y : y + 256, x : x + 256], B[..., y : y + 256, x : x + 256])[-1]
                prob[y : y + 256, x : x + 256] = torch.softmax(out, 1)[0, 1].numpy()
    return prob


def to_rgb(stack: np.ndarray, stretch: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = stack[[2, 1, 0]].astype(np.float32)  # B4, B3, B2
    y = x / 3000.0 if stretch == "fixed" else (x - lo[:, None, None]) / (hi - lo)[:, None, None]
    return np.clip(np.transpose(y, (1, 2, 0)), 0, 1)


def resize(img: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    size = (size, size) if isinstance(size, int) else size
    t = torch.from_numpy(np.ascontiguousarray(img if img.ndim == 2 else np.transpose(img, (2, 0, 1))))
    t = t[None, None] if img.ndim == 2 else t[None]
    out = F.interpolate(t.float(), size=size, mode="bilinear", align_corners=False)[0].numpy()
    return out[0] if img.ndim == 2 else np.transpose(out, (1, 2, 0))


def sanity(net, code: str) -> None:
    from PIL import Image

    d = os.path.join(code, "samples_LEVIR")
    stats = {"label": [0, 0, 0], "predict_ChangeFormerV6": [0, 0, 0]}
    for name in sorted(os.listdir(os.path.join(d, "A"))):
        a = np.asarray(Image.open(os.path.join(d, "A", name)).convert("RGB"), np.float32) / 255
        b = np.asarray(Image.open(os.path.join(d, "B", name)).convert("RGB"), np.float32) / 255
        pred = predict(net, a, b) > 0.5
        for folder, s in stats.items():
            path = os.path.join(d, folder, name)
            if os.path.exists(path):
                ref = np.asarray(Image.open(path).convert("L")) > 127
                s[0] += int((pred & ref).sum())
                s[1] += int((pred & ~ref).sum())
                s[2] += int((~pred & ref).sum())
    for folder, (tp, fp, fn) in stats.items():
        print(f"LEVIR samples, F1 against {folder}: {2 * tp / max(2 * tp + fp + fn, 1):.3f}")


def run(net, s2: str, tiles: str, out: str, variants: list[str], only: set[str] | None = None) -> None:
    import rasterio

    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "changeformer_scores.csv")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {(r["site"], r["variant"]) for r in csv.DictReader(f)}
    sites = sorted({f.split("_")[0] for f in os.listdir(s2) if f.endswith(".tif")}, key=lambda s: int(s[4:]))
    if only:
        sites = [s for s in sites if s in only]
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        writer = csv.DictWriter(f, ["site", "variant", "changed_ha", "changed_farmland_ha", "mean_prob", "seconds"])
        if new:
            writer.writeheader()
        for site in sites:
            todo = [v for v in variants if (site, v) not in done]
            if not todo:
                continue
            with rasterio.open(os.path.join(s2, f"{site}_before_2024.tif")) as r:
                sa, pixel_ha = r.read(), abs(r.transform.a * r.transform.e) / 1e4
            with rasterio.open(os.path.join(s2, f"{site}_after_2025.tif")) as r:
                sb = r.read()
            with rasterio.open(os.path.join(tiles, f"{site}_before_2024.tif")) as r:
                farmland = np.nan_to_num(r.read(1)) > 0.25  # NDVI before, as in the labelling rule
            both = np.concatenate([sa[[2, 1, 0]].reshape(3, -1), sb[[2, 1, 0]].reshape(3, -1)], 1).astype(np.float32)
            valid = both.min(0) > 0
            lo, hi = np.percentile(both[:, valid], 2, 1), np.percentile(both[:, valid], 98, 1)
            h, w = sa.shape[1:]
            path = os.path.join(out, f"{site}.npz")
            maps = dict(np.load(path)) if os.path.exists(path) else {}
            for v in todo:
                stretch, scale = v.split("_")
                size = 256 * int(scale[1:])
                t0 = time.time()
                prob = predict(
                    net, resize(to_rgb(sa, stretch, lo, hi), size), resize(to_rgb(sb, stretch, lo, hi), size)
                )
                prob = resize(prob, (h, w))
                changed = prob > 0.5
                maps[v] = np.clip(prob * 255, 0, 255).astype(np.uint8)
                writer.writerow(
                    {
                        "site": site,
                        "variant": v,
                        "changed_ha": round(float(changed.sum() * pixel_ha), 3),
                        "changed_farmland_ha": round(float((changed & farmland).sum() * pixel_ha), 3),
                        "mean_prob": round(float(prob.mean()), 5),
                        "seconds": round(time.time() - t0, 2),
                    }
                )
                f.flush()
            np.savez_compressed(path, **maps)
            print(site, {v: int((maps[v] > 127).sum()) for v in todo}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["sanity", "run"])
    parser.add_argument("variants", nargs="*", default=["pct_x1", "fixed_x1"])
    parser.add_argument("--code", required=True, help="ChangeFormer repository folder")
    parser.add_argument("--ckpt", required=True, help="LEVIR-CD best_ckpt.pt")
    parser.add_argument("--s2", help="Sentinel-2 tiles from download_s2.py")
    parser.add_argument("--tiles", help="KEMET1 index tiles (for the farmland mask)")
    parser.add_argument("--out", help="output folder")
    parser.add_argument("--sites", help="optional text file of site names to restrict the run to")
    args = parser.parse_args()
    net = load_model(args.code, args.ckpt)
    if args.mode == "sanity":
        sanity(net, args.code)
    else:
        only = set(open(args.sites).read().split()) if args.sites else None
        run(net, args.s2, args.tiles, args.out, args.variants, only)


if __name__ == "__main__":
    main()
