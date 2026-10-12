"""Fine-tune the LEVIR-CD ChangeFormerV6 on OSCD (Sentinel-2 change pairs) and score the KEMET1 sites.

Used by ``finetune_oscd.ipynb`` (Google Colab, GPU). The steps:

1. ``download_oscd`` fetches the Onera Satellite Change Detection dataset: 24
   Sentinel-2 pairs at 10 m with hand-drawn urban-change masks, 14 for training
   and 10 for testing.
2. ``build_model`` loads the LEVIR-CD checkpoint and, for more than 3 bands, widens
   the first layer (the RGB filters are kept; the new near-infrared and SWIR
   filters start from their mean).
3. ``train`` fine-tunes on random 256-pixel crops of the OSCD training pairs with
   ChangeFormer's multi-scale cross-entropy; ``evaluate_oscd`` reports F1 of the
   change class on the OSCD test pairs.
4. ``download_kemet_s2`` rebuilds the Sentinel-2 composites behind the 300 KEMET1
   tiles in Earth Engine (same recipe as ``download_s2.py``) on the grids in
   ``data/tile_grids.json``; ``score_kemet`` runs the model on every pair at its
   native 10 m and appends one row per site to ``changeformer_scores.csv``, the
   file ``compare_models.py`` reads.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time
import urllib.request
import zipfile

import numpy as np

KEMET_BANDS = ["B2", "B3", "B4", "B8", "B11"]  # order of the KEMET1 Sentinel-2 tiles
OSCD_BANDS = ["B02", "B03", "B04", "B08", "B11"]
CHANNELS = {"rgb": [2, 1, 0], "msi5": [2, 1, 0, 3, 4]}  # indices into the 5-band stacks
OSCD_URLS = {
    "Onera Satellite Change Detection dataset - Images.zip": "https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download",
    "Onera Satellite Change Detection dataset - Train Labels.zip": "https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download",
    "Onera Satellite Change Detection dataset - Test Labels.zip": "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download",
}
MULTI_SCALE_WEIGHTS = [0.5, 0.5, 0.5, 0.8, 1.0]  # ChangeFormer's own training setting


# --------------------------------------------------------------------------- data


def download_oscd(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    for name, url in OSCD_URLS.items():
        folder = os.path.join(root, name[:-4])
        if os.path.isdir(folder):
            continue
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(url, path)
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
        # Some archives contain the folder, some only its contents.
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
    return root


def _read_band_stack(folder: str, size: tuple[int, int] | None = None) -> np.ndarray:
    import rasterio
    import torch
    import torch.nn.functional as F

    bands = []
    for band in OSCD_BANDS:
        files = [f for f in glob.glob(os.path.join(folder, "*.tif")) if f.endswith(f"{band}.tif")]
        if not files:
            raise FileNotFoundError(f"no {band} file in {folder}")
        with rasterio.open(files[0]) as r:
            bands.append(r.read(1).astype(np.float32))
    size = size or bands[0].shape
    out = []
    for b in bands:
        if b.shape != size:
            b = F.interpolate(torch.from_numpy(b)[None, None], size=size, mode="bilinear", align_corners=False)
            b = b[0, 0].numpy()
        out.append(b)
    return np.stack(out)


def load_oscd(root: str) -> dict[str, list[dict]]:
    """{'train': [...], 'test': [...]}, each item with 5-band 'before', 'after' and boolean 'mask'."""
    from PIL import Image

    images = os.path.join(root, "Onera Satellite Change Detection dataset - Images")
    out = {}
    for split in ("train", "test"):
        labels = os.path.join(root, f"Onera Satellite Change Detection dataset - {split.capitalize()} Labels")
        items = []
        for region in sorted(os.listdir(labels)):
            cm = glob.glob(os.path.join(labels, region, "cm", "*.png"))
            if not cm:
                continue
            mask = np.asarray(Image.open(cm[0]).convert("L")) > 127
            before = _read_band_stack(os.path.join(images, region, "imgs_1_rect"), mask.shape)
            after = _read_band_stack(os.path.join(images, region, "imgs_2_rect"), mask.shape)
            items.append({"region": region, "before": before, "after": after, "mask": mask})
        out[split] = items
        print(f"OSCD {split}: {len(items)} pairs, change {np.mean([i['mask'].mean() for i in items]):.3f}", flush=True)
    return out


def stretch(before: np.ndarray, after: np.ndarray, channels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Both dates scaled per band with one shared 2-98 % range, as at inference; returns C x H x W in [0, 1]."""
    a, b = before[channels].astype(np.float32), after[channels].astype(np.float32)
    both = np.concatenate([a.reshape(len(channels), -1), b.reshape(len(channels), -1)], 1)
    valid = both.min(0) > 0
    sample = both[:, valid] if valid.sum() > 100 else both
    lo, hi = np.percentile(sample, 2, 1)[:, None, None], np.percentile(sample, 98, 1)[:, None, None]
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1), np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)


# -------------------------------------------------------------------------- model


def build_model(code: str, ckpt: str, n_channels: int, device: str = "cpu"):
    import sys

    import torch

    if code not in sys.path:
        sys.path.insert(0, code)
    from models.ChangeFormer import ChangeFormerV6

    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_G_state_dict"]
    key = "Tenc_x2.patch_embed1.proj.weight"
    if n_channels != 3:
        w = state[key]
        extra = w.mean(1, keepdim=True).repeat(1, n_channels - 3, 1, 1)
        state[key] = torch.cat([w, extra], 1) * (3 / n_channels)
    net = ChangeFormerV6(input_nc=n_channels, embed_dim=256)
    net.load_state_dict(state)
    return net.to(device)


def _tensor(x: np.ndarray):
    import torch

    return (torch.from_numpy(np.ascontiguousarray(x)).float() - 0.5) / 0.5


# ----------------------------------------------------------------------- training


def _crops(items, channels, batch, size, rng, change_bias=0.5):
    xs1, xs2, ys = [], [], []
    for _ in range(batch):
        it = items[rng.integers(len(items))]
        a, b = it["_stretched"]
        m = it["mask"]
        h, w = m.shape
        ph, pw = max(0, size - h), max(0, size - w)
        if ph or pw:
            a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            b = np.pad(b, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            m = np.pad(m, ((0, ph), (0, pw)))
            h, w = m.shape
        if rng.random() < change_bias and m.any():
            ys_, xs_ = np.nonzero(m)
            i = rng.integers(len(ys_))
            y0 = int(np.clip(ys_[i] - rng.integers(size), 0, h - size))
            x0 = int(np.clip(xs_[i] - rng.integers(size), 0, w - size))
        else:
            y0, x0 = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
        ca, cb, cm = (
            a[:, y0 : y0 + size, x0 : x0 + size],
            b[:, y0 : y0 + size, x0 : x0 + size],
            m[y0 : y0 + size, x0 : x0 + size],
        )
        k = int(rng.integers(4))
        ca, cb, cm = np.rot90(ca, k, (1, 2)), np.rot90(cb, k, (1, 2)), np.rot90(cm, k)
        if rng.random() < 0.5:
            ca, cb, cm = ca[:, :, ::-1], cb[:, :, ::-1], cm[:, ::-1]
        if rng.random() < 0.5:  # change is symmetric in time
            ca, cb = cb, ca
        xs1.append(ca.copy())
        xs2.append(cb.copy())
        ys.append(cm.copy())
    return np.stack(xs1), np.stack(xs2), np.stack(ys)


def train(net, items, channels, iters=2000, batch=8, lr=6e-5, change_weight=None, device="cpu", seed=0, log_every=100):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for it in items:
        it["_stretched"] = stretch(it["before"], it["after"], channels)
    frac = float(np.mean([it["mask"].mean() for it in items]))
    change_weight = change_weight or float(min(10.0, np.sqrt((1 - frac) / max(frac, 1e-3))))
    weight = torch.tensor([1.0, change_weight], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / iters)
    net.train()
    t0, running = time.time(), 0.0
    for step in range(1, iters + 1):
        a, b, m = _crops(items, channels, batch, 256, rng)
        a, b = _tensor(a).to(device), _tensor(b).to(device)
        target = torch.from_numpy(m.astype(np.int64)).to(device)
        outs = net(a, b)
        loss = 0.0
        for wgt, out in zip(MULTI_SCALE_WEIGHTS, outs, strict=True):
            out = F.interpolate(out, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + wgt * F.cross_entropy(out, target, weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss)
        if step % log_every == 0 or step == iters:
            print(
                f"step {step}/{iters}  loss {running / min(step, log_every):.4f}  {time.time() - t0:.0f}s", flush=True
            )
            running = 0.0
    net.eval()
    return net


def predict_full(net, a: np.ndarray, b: np.ndarray, device="cpu") -> np.ndarray:
    """Change probability for C x H x W inputs in [0, 1], in 256-pixel windows over a reflect-padded image."""
    import torch

    _, h, w = a.shape
    H, W = -(-h // 256) * 256, -(-w // 256) * 256
    pad = ((0, 0), (0, H - h), (0, W - w))
    a, b = np.pad(a, pad, mode="reflect"), np.pad(b, pad, mode="reflect")
    A, B = _tensor(a)[None].to(device), _tensor(b)[None].to(device)
    prob = np.zeros((H, W), np.float32)
    with torch.no_grad():
        for y in range(0, H, 256):
            for x in range(0, W, 256):
                out = net(A[..., y : y + 256, x : x + 256], B[..., y : y + 256, x : x + 256])[-1]
                prob[y : y + 256, x : x + 256] = torch.softmax(out, 1)[0, 1].float().cpu().numpy()
    return prob[:h, :w]


def evaluate_oscd(net, items, channels, device="cpu") -> dict:
    tp = fp = fn = 0
    for it in items:
        a, b = stretch(it["before"], it["after"], channels)
        pred = predict_full(net, a, b, device) > 0.5
        m = it["mask"]
        tp += int((pred & m).sum())
        fp += int((pred & ~m).sum())
        fn += int((~pred & m).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


# -------------------------------------------------------------------------- KEMET1


def download_kemet_s2(grids_json: str, out: str, project: str, workers: int = 8) -> None:
    """Rebuild the 600 KEMET1 Sentinel-2 tiles (B2, B3, B4, B8, B11) on their original grids."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import ee
    import rasterio
    from rasterio.transform import Affine

    ee.Initialize(project=project)
    os.makedirs(out, exist_ok=True)

    def composite(year):
        def mask(img):
            scl = img.select("SCL")
            return img.updateMask(scl.neq(3).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))

        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(f"{year}-07-01", f"{year}-09-15")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(mask)
            .select(KEMET_BANDS)
            .median()
        )

    images = {"before_2024": composite(2024), "after_2025": composite(2025)}
    grids = json.load(open(grids_json))

    def one(name, g):
        dst = os.path.join(out, f"{name}.tif")
        if os.path.exists(dst):
            return
        a, b, c, d, e, f = g["transform"]
        grid = {
            "dimensions": {"width": g["width"], "height": g["height"]},
            "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
            "crsCode": g["crs"],
        }
        tag = name.split("_", 1)[1]
        for attempt in range(4):
            try:
                arr = ee.data.computePixels({"expression": images[tag], "fileFormat": "NUMPY_NDARRAY", "grid": grid})
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        stack = np.clip(np.nan_to_num(np.stack([arr[k] for k in KEMET_BANDS]).astype(np.float32)), 0, 65535)
        profile = {
            "driver": "GTiff",
            "width": g["width"],
            "height": g["height"],
            "count": 5,
            "dtype": "uint16",
            "crs": g["crs"],
            "transform": Affine(a, b, c, d, e, f),
            "nodata": 0,
            "compress": "deflate",
        }
        with rasterio.open(dst, "w", **profile) as w:
            w.write(stack.astype(np.uint16))

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, n, g) for n, g in grids.items()]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 100 == 0:
                print(i, "tiles", flush=True)


def score_kemet(net, s2_dir: str, channels: list[int], variant: str, scores_csv: str, maps_dir: str, device="cpu"):
    """Run the fine-tuned model on every KEMET1 pair at 10 m and append rows to ``scores_csv``."""
    import rasterio

    os.makedirs(maps_dir, exist_ok=True)
    sites = sorted(
        {os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(s2_dir, "*.tif"))}, key=lambda s: int(s[4:])
    )
    existing = []
    if os.path.exists(scores_csv):
        with open(scores_csv) as f:
            existing = [r for r in csv.DictReader(f) if r["variant"] != variant]
    rows = []
    for site in sites:
        with rasterio.open(os.path.join(s2_dir, f"{site}_before_2024.tif")) as r:
            sa, pixel_ha = r.read().astype(np.float32), abs(r.transform.a * r.transform.e) / 1e4
        with rasterio.open(os.path.join(s2_dir, f"{site}_after_2025.tif")) as r:
            sb = r.read().astype(np.float32)
        t0 = time.time()
        a, b = stretch(sa, sb, channels)
        prob = predict_full(net, a, b, device)
        ndvi = (sa[3] - sa[2]) / (sa[3] + sa[2] + 1e-6)
        changed = prob > 0.5
        rows.append(
            {
                "site": site,
                "variant": variant,
                "changed_ha": round(float(changed.sum() * pixel_ha), 3),
                "changed_farmland_ha": round(float((changed & (ndvi > 0.25)).sum() * pixel_ha), 3),
                "mean_prob": round(float(prob.mean()), 5),
                "seconds": round(time.time() - t0, 2),
            }
        )
        np.savez_compressed(os.path.join(maps_dir, f"{site}_{variant}.npz"), prob=(prob * 255).astype(np.uint8))
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, ["site", "variant", "changed_ha", "changed_farmland_ha", "mean_prob", "seconds"])
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"{variant}: {len(rows)} sites, any change at {sum(r['changed_ha'] > 0 for r in rows)}", flush=True)
    return rows
