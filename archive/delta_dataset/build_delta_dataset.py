"""
build_delta_dataset.py
======================
Build a proper co-registered encroachment dataset for the Nile Delta.

Each tile = one fixed geographic location (2.6 km × 2.6 km, 260×260 px @ 10 m).
For every tile we export 5 annual Sentinel-2 composites: 2021 → 2025.
All five images cover the exact same bounding box → true before/after comparison.

Outputs (exported to Google Drive → folder "NileDelta_Encroachment"):
  {year}/tile_{id:03d}_{year}_{label}.tif   (6-band spectral index GeoTIFF)

Bands (same order as KEMET1):
  1 NDVI   (vegetation)
  2 NDBI   (built-up)
  3 MNDWI  (water)
  4 SAVI   (soil-adjusted vegetation)
  5 BSI    (bare soil)
  6 NDWI   (general water / moisture)

Labels:
  neg  = stable agricultural land in 2021 (used as negative class)
  unk  = not yet labelled (you annotate after download)
  Positive labels are assigned manually or via NDVI-loss check after download.

Season: July 1 – September 15 (summer peak NDVI for Egypt's delta).
        Same window every year → consistent phenology.

Steps
-----
1. pip install earthengine-api geopandas shapely tqdm  (run once)
2. earthengine authenticate                            (run once, opens browser)
3. python build_delta_dataset.py --dry-run             (check tile count, no export)
4. python build_delta_dataset.py                       (submit GEE export tasks)
5. Monitor tasks at https://code.earthengine.google.com/tasks
6. Download from Google Drive when done
7. python build_delta_dataset.py --label               (auto-label by NDVI loss)

Usage
-----
  python build_delta_dataset.py [--dry-run] [--label] [--max-tiles N]
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
TILE_SIZE_M   = 2600        # 260 px × 10 m
YEARS         = [2021, 2022, 2023, 2024, 2025]
SEASON_START  = "07-01"     # July 1
SEASON_END    = "09-15"     # September 15
CLOUD_MAX_PCT = 20          # discard scenes with > 20 % cloud cover
SCALE_M       = 10          # Sentinel-2 native resolution
DRIVE_FOLDER  = "NileDelta_Encroachment"

# Nile Delta bounding box (WGS-84)
DELTA_WEST  = 29.5
DELTA_EAST  = 32.4
DELTA_SOUTH = 29.8
DELTA_NORTH = 31.65

# ── Known encroachment hotspot centres (lon, lat) ─────────────────────────────
# These are cities / roads whose agricultural fringes show rapid encroachment.
# We sample extra tiles within HOTSPOT_RADIUS_KM of each centre.
HOTSPOT_CENTRES = [
    # (lon,   lat,    name)
    (31.002, 30.794, "Tanta"),
    (30.935, 31.113, "Kafr_El_Sheikh"),
    (30.468, 30.468, "Damanhur"),
    (31.378, 31.037, "Mansoura"),
    (31.502, 30.584, "Zagazig"),
    (30.994, 30.559, "Shebin_El_Kom"),
    (31.178, 30.468, "Banha"),
    (31.263, 30.724, "Mitt_Ghamr"),
    (31.371, 31.052, "Talkha"),
    (31.114, 30.325, "Cairo_NE_fringe"),
    (30.650, 31.320, "Kafr_El_Dawar"),
    (31.650, 31.250, "Damietta"),
    (32.300, 31.500, "Port_Said_fringe"),
    (30.120, 30.855, "Alexandria_E_fringe"),
    (31.800, 30.950, "Ismailia_fringe"),
    (31.200, 30.900, "El_Gharbia"),
    (30.800, 30.600, "Menofia"),
    (31.500, 30.780, "Sharkia_west"),
]
HOTSPOT_RADIUS_KM = 15   # sample tiles within this radius

# ── Sampling grid parameters ───────────────────────────────────────────────────
# Tile stride (centre-to-centre distance). We use 1.5× tile size to avoid
# overlap while keeping dense sampling.
GRID_STRIDE_DEG = TILE_SIZE_M / 111_000 * 1.5   # ~0.0351°

# Target mix: 60 % hotspot tiles, 40 % random agricultural background
TARGET_TOTAL     = 220
TARGET_HOTSPOT   = 130
TARGET_BACKGROUND = 90


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Generate tile centres (pure Python / shapely, no GEE needed)
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def in_delta(lon, lat) -> bool:
    return DELTA_WEST < lon < DELTA_EAST and DELTA_SOUTH < lat < DELTA_NORTH


def near_hotspot(lon, lat) -> bool:
    return any(
        haversine_km(lon, lat, hlon, hlat) <= HOTSPOT_RADIUS_KM
        for hlon, hlat, _ in HOTSPOT_CENTRES
    )


def generate_tile_centres(max_tiles: int = TARGET_TOTAL) -> list[dict]:
    """
    Build a list of tile centre dicts:
      {id, lon, lat, hotspot: bool}
    Mix: hotspot-zone tiles + background tiles.
    """
    import random
    random.seed(42)

    hotspot_tiles, background_tiles = [], []

    lon = DELTA_WEST + GRID_STRIDE_DEG / 2
    while lon < DELTA_EAST:
        lat = DELTA_SOUTH + GRID_STRIDE_DEG / 2
        while lat < DELTA_NORTH:
            if in_delta(lon, lat):
                if near_hotspot(lon, lat):
                    hotspot_tiles.append((lon, lat))
                else:
                    background_tiles.append((lon, lat))
            lat += GRID_STRIDE_DEG
        lon += GRID_STRIDE_DEG

    print(f"Grid candidates: {len(hotspot_tiles)} hotspot, {len(background_tiles)} background")

    # Sample to target mix
    n_hot = min(TARGET_HOTSPOT, len(hotspot_tiles))
    n_bg  = min(TARGET_BACKGROUND, len(background_tiles))
    selected = (
        random.sample(hotspot_tiles, n_hot) +
        random.sample(background_tiles, n_bg)
    )
    random.shuffle(selected)
    selected = selected[:max_tiles]

    tiles = []
    for i, (lon, lat) in enumerate(selected):
        tiles.append({
            "id":      i + 1,
            "lon":     round(lon, 6),
            "lat":     round(lat, 6),
            "hotspot": near_hotspot(lon, lat),
            "label":   "unk",   # filled in after download / annotation
        })
    print(f"Selected {len(tiles)} tiles ({sum(t['hotspot'] for t in tiles)} hotspot)")
    return tiles


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — Google Earth Engine export
# ══════════════════════════════════════════════════════════════════════════════

def mask_s2_clouds(image):
    """Mask clouds using Sentinel-2 SCL band (Scene Classification Layer)."""
    import ee
    scl = image.select("SCL")
    # SCL classes to mask: 3=shadow, 7=unclassified, 8=cloud medium, 9=cloud high, 10=cirrus
    mask = scl.neq(3).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000)  # scale to [0,1]


def compute_indices(image):
    """Add 6 spectral index bands to a Sentinel-2 SR image."""
    import ee
    B8  = image.select("B8")   # NIR
    B4  = image.select("B4")   # Red
    B3  = image.select("B3")   # Green
    B11 = image.select("B11")  # SWIR1
    B2  = image.select("B2")   # Blue

    eps = 0.0001  # avoid division by zero

    ndvi  = B8.subtract(B4).divide(B8.add(B4).add(eps)).rename("NDVI")
    ndbi  = B11.subtract(B8).divide(B11.add(B8).add(eps)).rename("NDBI")
    mndwi = B3.subtract(B11).divide(B3.add(B11).add(eps)).rename("MNDWI")
    savi  = B8.subtract(B4).multiply(1.5).divide(B8.add(B4).add(0.5).add(eps)).rename("SAVI")
    bsi   = (B11.add(B4)).subtract(B8.add(B2)).divide(
             (B11.add(B4)).add(B8.add(B2)).add(eps)).rename("BSI")
    ndwi  = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename("NDWI")

    return ee.Image([ndvi, ndbi, mndwi, savi, bsi, ndwi])


def get_annual_composite(lon: float, lat: float, year: int):
    """Return a 6-band spectral index image for (lon, lat) tile, given year."""
    import ee

    half = TILE_SIZE_M / 2
    point = ee.Geometry.Point([lon, lat])

    # Tile bounding box (approximate using degrees — GEE clips to exact metres)
    # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km × cos(lat)
    dlat = half / 111_000
    dlon = half / (111_000 * math.cos(math.radians(lat)))
    roi = ee.Geometry.Rectangle([lon - dlon, lat - dlat, lon + dlon, lat + dlat])

    start = f"{year}-{SEASON_START}"
    end   = f"{year}-{SEASON_END}"

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX_PCT))
        .map(mask_s2_clouds)
        .select(["B2", "B3", "B4", "B8", "B11", "SCL"])
    )

    # Median composite → robust to remaining clouds / outliers
    composite = col.median()
    indices   = compute_indices(composite)
    return indices.clip(roi), roi


def submit_export_tasks(tiles: list[dict], dry_run: bool = False) -> None:
    """Submit one GEE export task per (tile, year)."""
    import ee
    ee.Initialize()

    total = len(tiles) * len(YEARS)
    print(f"\nSubmitting {total} export tasks ({len(tiles)} tiles × {len(YEARS)} years)...")
    if dry_run:
        print("[DRY RUN] No tasks submitted.")
        return

    submitted = 0
    for tile in tiles:
        tid  = tile["id"]
        lon  = tile["lon"]
        lat  = tile["lat"]
        lbl  = tile["label"]

        for year in YEARS:
            image, roi = get_annual_composite(lon, lat, year)
            desc = f"tile_{tid:03d}_{year}_{lbl}"
            folder = f"{DRIVE_FOLDER}/{year}"

            task = ee.batch.Export.image.toDrive(
                image       = image,
                description = desc,
                folder      = folder,
                fileNamePrefix = desc,
                region      = roi,
                scale       = SCALE_M,
                crs         = "EPSG:32636",   # UTM 36N — same as KEMET1
                maxPixels   = 1e8,
                fileFormat  = "GeoTIFF",
            )
            task.start()
            submitted += 1
            if submitted % 50 == 0:
                print(f"  {submitted}/{total} tasks submitted...")

    print(f"\nDone. {submitted} tasks submitted.")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Files will appear in Google Drive → {DRIVE_FOLDER}/{{year}}/")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — Auto-labelling by NDVI loss (run after download)
# ══════════════════════════════════════════════════════════════════════════════

def auto_label(download_dir: Path, tiles: list[dict],
               ndvi_loss_thresh: float = -0.10,
               min_pct_pixels:   float = 0.05) -> list[dict]:
    """
    Compare 2021 NDVI to 2024/2025 NDVI per tile.
    Label tile as 'pos' if ≥ min_pct_pixels fraction of pixels lost
    ≥ ndvi_loss_thresh NDVI between 2021 and the latest available year.
    Otherwise label 'neg'.

    Requires rasterio.  Run:  pip install rasterio
    """
    try:
        import rasterio
        import numpy as np
    except ImportError:
        print("[ERROR] pip install rasterio numpy  then re-run --label")
        return tiles

    print(f"\nAuto-labelling {len(tiles)} tiles ...")
    pos_count = neg_count = skip_count = 0

    for tile in tiles:
        tid = tile["id"]
        lbl = "unk"

        # Find 2021 file
        t2021 = next(download_dir.glob(f"2021/tile_{tid:03d}_2021_*.tif"), None)
        # Use 2025 if available, else 2024
        t_late = next(download_dir.glob(f"2025/tile_{tid:03d}_2025_*.tif"), None) or \
                 next(download_dir.glob(f"2024/tile_{tid:03d}_2024_*.tif"), None)

        if t2021 is None or t_late is None:
            skip_count += 1
            tile["label"] = "unk"
            continue

        with rasterio.open(t2021) as s:
            ndvi_2021 = s.read(1).astype(float)  # band 1 = NDVI
        with rasterio.open(t_late) as s:
            ndvi_late = s.read(1).astype(float)

        loss = ndvi_late - ndvi_2021           # negative = vegetation lost
        valid = (~np.isnan(loss)) & (~np.isnan(ndvi_2021))
        if valid.sum() == 0:
            skip_count += 1
            tile["label"] = "unk"
            continue

        pct_lost = (loss[valid] < ndvi_loss_thresh).mean()
        lbl = "pos" if pct_lost >= min_pct_pixels else "neg"
        tile["label"] = lbl

        if lbl == "pos":
            pos_count += 1
        else:
            neg_count += 1

    print(f"  pos={pos_count}  neg={neg_count}  skipped={skip_count}")
    return tiles


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — Rename downloaded files to include correct label
# ══════════════════════════════════════════════════════════════════════════════

def rename_with_labels(download_dir: Path, tiles: list[dict]) -> None:
    """Rename tile_{id}_{year}_unk.tif → tile_{id}_{year}_{label}.tif"""
    import os
    label_map = {t["id"]: t["label"] for t in tiles}
    renamed = 0
    for tif in download_dir.rglob("*.tif"):
        parts = tif.stem.split("_")
        # expected: tile_NNN_YEAR_LABEL
        if len(parts) == 4 and parts[0] == "tile":
            tid  = int(parts[1])
            year = parts[2]
            old_lbl = parts[3]
            new_lbl = label_map.get(tid, old_lbl)
            if new_lbl != old_lbl:
                new_name = tif.parent / f"tile_{tid:03d}_{year}_{new_lbl}.tif"
                os.rename(tif, new_name)
                renamed += 1
    print(f"Renamed {renamed} files.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build Nile Delta encroachment dataset via GEE")
    parser.add_argument("--dry-run",   action="store_true", help="Generate tiles but don't submit exports")
    parser.add_argument("--label",     action="store_true", help="Auto-label by NDVI loss after download")
    parser.add_argument("--max-tiles", type=int, default=TARGET_TOTAL, help="Max number of tiles")
    parser.add_argument("--download-dir", type=Path, default=Path("data/NileDelta_Encroachment"),
                        help="Local folder where GDrive files were downloaded")
    parser.add_argument("--tiles-json", type=Path, default=Path("data/delta_tiles.json"),
                        help="JSON file to save/load tile metadata")
    args = parser.parse_args()

    # ── Step 1: generate or load tile centres ─────────────────────────────────
    if args.tiles_json.exists() and not args.label:
        tiles = json.loads(args.tiles_json.read_text())
        print(f"Loaded {len(tiles)} tiles from {args.tiles_json}")
    else:
        tiles = generate_tile_centres(max_tiles=args.max_tiles)
        args.tiles_json.parent.mkdir(parents=True, exist_ok=True)
        args.tiles_json.write_text(json.dumps(tiles, indent=2))
        print(f"Saved tile metadata → {args.tiles_json}")

    if args.label:
        # ── Step 3: auto-label after download ─────────────────────────────────
        tiles = auto_label(args.download_dir, tiles)
        args.tiles_json.write_text(json.dumps(tiles, indent=2))
        rename_with_labels(args.download_dir, tiles)
        print(f"Updated labels saved → {args.tiles_json}")
        return

    # ── Step 2: submit GEE export tasks ───────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Nile Delta Encroachment Dataset Builder")
    print("="*60)
    print(f"  Tiles        : {len(tiles)}")
    print(f"  Years        : {YEARS}")
    print(f"  Season       : {SEASON_START} – {SEASON_END} (July–Sept)")
    print(f"  Tile size    : {TILE_SIZE_M} m × {TILE_SIZE_M} m ({TILE_SIZE_M//SCALE_M}×{TILE_SIZE_M//SCALE_M} px @ {SCALE_M} m)")
    print(f"  Total exports: {len(tiles) * len(YEARS)}")
    print(f"  Drive folder : {DRIVE_FOLDER}/{{year}}/")
    print("="*60)

    if args.dry_run:
        print("\n[DRY RUN] — no GEE tasks submitted.")
        print("Run without --dry-run to submit exports.")
        # Print first 10 tiles for inspection
        print("\nFirst 10 tiles:")
        for t in tiles[:10]:
            hs = "★ hotspot" if t["hotspot"] else "  background"
            print(f"  tile_{t['id']:03d}  lat={t['lat']:.4f}  lon={t['lon']:.4f}  {hs}")
        return

    # Authenticate + submit
    try:
        import ee
    except ImportError:
        print("\n[ERROR] earthengine-api not installed.")
        print("Run:  pip install earthengine-api")
        sys.exit(1)

    submit_export_tasks(tiles, dry_run=False)


if __name__ == "__main__":
    main()
