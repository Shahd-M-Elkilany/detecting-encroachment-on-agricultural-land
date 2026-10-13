"""Download Sentinel-2 B2, B3, B4, B8, B11 composites on the exact grid of each KEMET1 index tile.

ChangeFormer needs colour images; the KEMET1 tiles hold spectral indices. This
rebuilds the composites the tiles were made from (S2_SR_HARMONIZED, 1 July to
15 September, scenes under 20 % cloud, SCL cloud/shadow mask, median) and writes
one 5-band uint16 GeoTIFF per tile, plus a log of how well the NDVI recomputed
from the download matches band 1 of the original tile.

    python download_s2.py --project <earth-engine-project> --tiles ../../data/tiles --out ../../data/s2_tiles
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

BANDS = ["B2", "B3", "B4", "B8", "B11"]
YEARS = (("before", 2024), ("after", 2025))


def composite(ee, year: int):
    def mask(img):
        scl = img.select("SCL")
        return img.updateMask(scl.neq(3).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{year}-07-01", f"{year}-09-15")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask)
        .select(BANDS)
        .median()
    )


def download_tile(ee, image, src: str, dst: str) -> dict:
    import rasterio

    with rasterio.open(src) as r:
        ndvi_tile = r.read(1).astype(np.float32)
        profile, t = r.profile, r.transform
        grid = {
            "dimensions": {"width": r.width, "height": r.height},
            "affineTransform": {
                "scaleX": t.a,
                "shearX": t.b,
                "translateX": t.c,
                "shearY": t.d,
                "scaleY": t.e,
                "translateY": t.f,
            },
            "crsCode": r.crs.to_string(),
        }
    for attempt in range(4):
        try:
            arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    stack = np.stack([arr[b].astype(np.float32) for b in BANDS])
    ndvi = (stack[3] - stack[2]) / (stack[3] + stack[2] + 1e-6)
    ok = np.isfinite(ndvi_tile) & (stack[3] > 0)
    r_ndvi = float(np.corrcoef(ndvi_tile[ok], ndvi[ok])[0, 1]) if ok.sum() > 100 else float("nan")
    profile.update(count=5, dtype="uint16", nodata=0, compress="deflate")
    with rasterio.open(dst, "w", **profile) as w:
        w.write(np.clip(np.nan_to_num(stack), 0, 65535).astype(np.uint16))
        w.descriptions = tuple(BANDS)
    return {"ndvi_r": round(r_ndvi, 4), "missing_frac": round(float(1 - (stack[3] > 0).mean()), 4)}


def main() -> None:
    import ee

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="Earth Engine cloud project")
    parser.add_argument("--tiles", required=True, help="folder with site{N}_before_2024.tif index tiles")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    os.makedirs(args.out, exist_ok=True)
    images = {year: composite(ee, year) for _, year in YEARS}
    log = os.path.join(args.out, "download_log.csv")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {(r["site"], r["tile"]) for r in csv.DictReader(f)}
    sites = sorted({f.split("_")[0] for f in os.listdir(args.tiles) if f.endswith(".tif")}, key=lambda s: int(s[4:]))
    jobs = [(s, t, y) for s in sites for t, y in YEARS if (s, f"{t}_{y}") not in done]
    print(f"{len(jobs)} tiles to download", flush=True)
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, ["site", "tile", "ndvi_r", "missing_frac"])
        if new:
            writer.writeheader()
        futures = {
            pool.submit(
                download_tile,
                ee,
                images[y],
                os.path.join(args.tiles, f"{s}_{t}_{y}.tif"),
                os.path.join(args.out, f"{s}_{t}_{y}.tif"),
            ): (s, t, y)
            for s, t, y in jobs
        }
        for n, fut in enumerate(as_completed(futures), 1):
            s, t, y = futures[fut]
            try:
                writer.writerow({"site": s, "tile": f"{t}_{y}", **fut.result()})
                f.flush()
            except Exception as e:  # noqa: BLE001
                print("FAILED", s, t, repr(e)[:200], flush=True)
            if n % 50 == 0:
                print(n, "done", flush=True)


if __name__ == "__main__":
    main()
