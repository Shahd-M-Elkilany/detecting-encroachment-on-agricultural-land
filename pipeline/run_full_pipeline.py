"""Run the 8-step pipeline on a KEMET1 tile pair (six pre-computed index bands).

The KEMET1 tiles carry spectral indices rather than raw Sentinel-2 bands, so the
deep-learning models in Steps 02, 05, 06 and 07 (which expect raw bands or RGB)
are replaced by the spectral rules each step falls back to:

  Step 01  Data acquisition    read the two GeoTIFFs
  Step 02  Cloud detection     skipped (composites are cloud-free)
  Step 03  Cloud removal       skipped
  Step 04  Spectral indices    NDVI / NDBI / MNDWI taken from bands 0-2
  Step 05  Change detection    mean absolute band difference above 0.5 of the tile maximum
  Step 06  Agriculture mask    NDVI before > 0.2
  Step 07  Building mask       NDBI rise > 0.05 inside changed farmland, morphologically cleaned
  Step 08  Final output        red / yellow / green map, regions, GeoJSON, map, JSON report

Usage:
    python run_full_pipeline.py --site site264                 # tiles in ../data/tiles
    python run_full_pipeline.py --before b.tif --after a.tif --site mysite
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# Pre-computed index mode must be set before any step module is imported.
from config import settings as _settings

_settings.SPECTRAL_INDICES_CONFIG["precomputed_indices"] = True

from src.utils.geo_utils import pixel_area_m2, read_geotiff  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("run_full_pipeline")

TILES_DIR = Path(__file__).resolve().parent.parent / "data" / "tiles"


def quiet() -> None:
    """Silence the per-step INFO logging (used when scoring many sites)."""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("step_") or name in {"pipeline", "run_full_pipeline"}:
            logging.getLogger(name).setLevel(logging.WARNING)


def pseudo_rgb(image: np.ndarray) -> np.ndarray:
    """Displayable [H, W, 3] uint8 image: R = NDVI, G = NDBI, B = MNDWI."""

    def norm(band, lo, hi):
        return ((np.clip(band.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)

    return np.stack([norm(image[0], -0.2, 1.0), norm(image[1], -0.5, 0.5), norm(image[2], -0.5, 0.5)], axis=-1)


def load_pair(before_path: Path, after_path: Path):
    t1, meta = read_geotiff(before_path)
    t2, _ = read_geotiff(after_path)
    t1, t2 = np.nan_to_num(t1), np.nan_to_num(t2)
    if t2.shape[1:] != t1.shape[1:]:
        import cv2

        h, w = t1.shape[1:]
        t2 = np.stack([cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR) for b in t2])
    return t1, t2, meta


def analyse(t1: np.ndarray, t2: np.ndarray, meta: dict) -> dict:
    """Steps 04-07 plus the Step 08 alert scoring, without writing any files."""
    from src.step_04_spectral_indices.compute_indices import run as step04
    from src.step_05_change_detection.detect_changes import run as step05
    from src.step_06_agriculture_segmentation.segment_agriculture import run as step06
    from src.step_07_building_detection.detect_buildings import run as step07
    from src.step_08_final_output.generate_output import _compute_red_score, _extract_regions

    s04 = step04(t1, t2, meta, meta)
    s05 = step05(t1, t2, meta)
    s06 = step06(t1, meta)
    s07 = step07(t2, s05["change_map"], s06["agri_mask"], meta, t1_image=t1)
    red_score = _compute_red_score(s05["change_confidence"], s04["spectral_signal"], s05["change_map"])
    regions = _extract_regions(
        red_score,
        s07["building_mask"],
        s06["agri_mask"],
        meta.get("transform"),
        meta,
        None,
        np.zeros(t1.shape[1:] + (3,), np.uint8),
        s04["yellow_mask"],
        s04["spectral_signal"],
    )
    pixel_ha = (pixel_area_m2(meta["transform"]) if meta.get("transform") else 100.0) / 10_000
    return {
        "s04": s04,
        "s05": s05,
        "s06": s06,
        "s07": s07,
        "regions": regions,
        "summary": {
            "regions": len(regions),
            "red_ha": round(sum(r["area_ha"] for r in regions), 4),
            "building_ha": round(float(s07["building_mask"].sum()) * pixel_ha, 4),
            "yellow_ha": round(float(s04["yellow_mask"].sum()) * pixel_ha, 4),
            "changed_pct": round(float(s05["change_map"].mean() * 100), 3),
            "agri_pct": round(float(s06["agri_mask"].mean() * 100), 3),
        },
    }


def run_pipeline(before_path: Path, after_path: Path, site: str, out_dir: Path | None = None) -> dict:
    """Run all eight steps and write the Step 08 outputs to ``outputs/<site>/``."""
    import src.step_08_final_output.generate_output as step08

    t1, t2, meta = load_pair(before_path, after_path)
    logger.info(f"{site}: T1 {t1.shape}, T2 {t2.shape}; Steps 02-03 skipped (cloud-free composites)")
    result = analyse(t1, t2, meta)

    step08.OUTPUT_DIR = Path(out_dir) if out_dir else Path(_settings.OUTPUT_DIR) / site
    s04, s05, s06, s07 = result["s04"], result["s05"], result["s06"], result["s07"]
    out = step08.run(
        t2_rgb=pseudo_rgb(t2),
        change_map=s05["change_map"],
        agri_mask=s06["agri_mask"],
        building_mask=s07["building_mask"],
        polygons=s07["polygons"],
        meta=meta,
        change_confidence=s05["change_confidence"],
        spectral_signal=s04["spectral_signal"],
        yellow_mask=s04["yellow_mask"],
        t1_rgb=pseudo_rgb(t1),
    )
    summary = {"site": site, **result["summary"], "outputs": str(step08.OUTPUT_DIR)}
    logger.info(f"{site}: {summary}")
    return {"summary": summary, "paths": out["paths"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="8-step pipeline on a KEMET1 index-tile pair")
    parser.add_argument("--site", help="site name, e.g. site264 (reads ../data/tiles unless --before/--after)")
    parser.add_argument("--before", help="before GeoTIFF")
    parser.add_argument("--after", help="after GeoTIFF")
    parser.add_argument("--tiles", default=str(TILES_DIR), help="folder with site{N}_before_2024.tif tiles")
    args = parser.parse_args()

    if args.before and args.after:
        before, after = Path(args.before), Path(args.after)
        site = args.site or Path(args.before).stem.split("_before")[0]
    elif args.site:
        before = Path(args.tiles) / f"{args.site}_before_2024.tif"
        after = Path(args.tiles) / f"{args.site}_after_2025.tif"
        site = args.site
    else:
        parser.error("give --site, or --before and --after")
    print(run_pipeline(before, after, site)["summary"])


if __name__ == "__main__":
    main()
