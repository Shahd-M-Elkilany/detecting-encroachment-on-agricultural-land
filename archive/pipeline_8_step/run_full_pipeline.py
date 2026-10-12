"""
run_full_pipeline.py — Run the 8-step KEMET1 pipeline on a before/after GeoTIFF pair.

All steps use fallback modes (no GPU, no deep learning weights required):
  Step 02  Cloud detection  → skipped (precomputed index input)
  Step 03  Cloud removal    → skipped (0% cloud coverage)
  Step 04  Spectral indices → reads Band 0/1/2 directly (precomputed mode)
  Step 05  Change detection → mean-abs-difference fallback (no ChangeFormer)
  Step 06  Agri mask        → NDVI > 0.2 threshold on Band 0 (no SegFormer)
  Step 07  Building mask    → NDBI > 0.1 morphological fallback on Band 1 (no YOLO)
  Step 08  Final output     → full alert map, geocoded regions, HTML + JSON

Usage:
    python run_full_pipeline.py --site site3
    python run_full_pipeline.py --before path/to/before.tif --after path/to/after.tif --site mysite
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ── Patch precomputed_indices BEFORE any step imports ────────────────────────
from config import settings as _s
_s.SPECTRAL_INDICES_CONFIG["precomputed_indices"] = True
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from src.utils.geo_utils import read_geotiff
from src.utils.logger import get_logger

logger = get_logger("run_full_pipeline")

DATA_DIR = Path("data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles")


def make_pseudorgb(image: np.ndarray) -> np.ndarray:
    """
    Create a displayable [H,W,3] uint8 RGB from 6-band spectral index GeoTIFF.
    Mapping: R=NDVI, G=NDBI, B=MNDWI  (normalised to 0-255)
    High NDVI (healthy farmland) → bright red channel.
    High NDBI (built-up)         → bright green channel.
    """
    def norm(band, lo, hi):
        clipped = np.clip(band.astype(np.float32), lo, hi)
        return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    r = norm(image[0], -0.2, 1.0)   # NDVI
    g = norm(image[1], -0.5, 0.5)   # NDBI
    b = norm(image[2], -0.5, 0.5)   # MNDWI
    return np.stack([r, g, b], axis=-1)


def run_pipeline(before_path: Path, after_path: Path, site_name: str) -> dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"  KEMET1 FULL PIPELINE — {site_name}")
    logger.info(f"{'='*70}")

    # ── Step 01: Load GeoTIFFs ────────────────────────────────────────────
    logger.info("\n  STEP 01 — DATA ACQUISITION (offline)")
    t1_data, t1_meta = read_geotiff(before_path)
    t2_data, t2_meta = read_geotiff(after_path)
    logger.info(f"  T1: {t1_data.shape}  T2: {t2_data.shape}  CRS: {t1_meta.get('crs','?')}")

    # Align T2 to T1 if shapes differ
    if t2_data.shape[1:] != t1_data.shape[1:]:
        import cv2
        h, w = t1_data.shape[1], t1_data.shape[2]
        t2_data = np.stack([
            cv2.resize(t2_data[b], (w, h), interpolation=cv2.INTER_LINEAR)
            for b in range(t2_data.shape[0])
        ], axis=0)
        logger.info(f"  T2 aligned to {t2_data.shape}")

    # ── Step 02: Cloud detection — skipped (precomputed indices) ──────────
    logger.info("\n  STEP 02 — CLOUD DETECTION [SKIPPED — precomputed index input]")

    # ── Step 03: Cloud removal — skipped (0% cloud) ───────────────────────
    logger.info("  STEP 03 — CLOUD REMOVAL [SKIPPED — 0% cloud coverage]")

    # ── Step 04: Spectral indices (precomputed mode) ───────────────────────
    logger.info("\n  STEP 04 — SPECTRAL INDICES (precomputed mode)")
    from src.step_04_spectral_indices.compute_indices import run as step04
    s04 = step04(t1_data, t2_data, t1_meta, t2_meta)
    logger.info(f"  Yellow pixels: {s04['yellow_pct']:.2f}%")

    # ── Step 05: Change detection (difference fallback) ────────────────────
    logger.info("\n  STEP 05 — CHANGE DETECTION (mean-abs-diff fallback)")
    from src.step_05_change_detection.detect_changes import run as step05
    s05 = step05(t1_data, t2_data, t1_meta)
    pct = s05['change_map'].mean() * 100
    logger.info(f"  Changed pixels: {pct:.2f}%")

    # ── Step 06: Agriculture segmentation (NDVI fallback) ─────────────────
    logger.info("\n  STEP 06 — AGRICULTURE SEGMENTATION (NDVI > 0.2 fallback)")
    from src.step_06_agriculture_segmentation.segment_agriculture import run as step06
    s06 = step06(t1_data, t1_meta)
    logger.info(f"  Agricultural land: {s06['agri_pct']:.2f}%")

    # ── Step 07: Building detection (NDBI morphological fallback) ─────────
    logger.info("\n  STEP 07 — BUILDING DETECTION (NDBI > 0.1 morphological fallback)")
    from src.step_07_building_detection.detect_buildings import run as step07
    s07 = step07(t2_data, s05["change_map"], s06["agri_mask"], t2_meta, t1_image=t1_data)
    logger.info(f"  Building pixels: {s07['building_mask'].sum():,}  |  Polygons: {len(s07['polygons'])}")

    # ── Step 08: Final output ──────────────────────────────────────────────
    logger.info("\n  STEP 08 — FINAL OUTPUT")
    from src.step_08_final_output.generate_output import run as step08
    t1_rgb = make_pseudorgb(t1_data)
    t2_rgb = make_pseudorgb(t2_data)
    s08 = step08(
        t2_rgb          = t2_rgb,
        change_map      = s05["change_map"],
        agri_mask       = s06["agri_mask"],
        building_mask   = s07["building_mask"],
        polygons        = s07["polygons"],
        meta            = t1_meta,
        change_confidence = s05["change_confidence"],
        spectral_signal   = s04["spectral_signal"],
        yellow_mask       = s04["yellow_mask"],
        t1_rgb            = t1_rgb,
    )

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE — {site_name}")
    logger.info(f"  Encroachment regions : {len(s08.get('regions', []))}")
    logger.info(f"  Total area (red)     : {s08.get('total_ha', 0):.2f} ha")
    logger.info(f"  Yellow area          : {s08.get('yellow_ha', 0):.2f} ha")
    logger.info(f"  Alert tier           : {s08.get('alert_level', 'unknown')}")
    logger.info(f"{'='*70}\n")

    return {
        "site": site_name,
        "regions": len(s08.get("regions", [])),
        "total_ha": s08.get("total_ha", 0),
        "yellow_ha": s08.get("yellow_ha", 0),
        "alert_level": s08.get("alert_level", "unknown"),
        "changed_pct": float(pct),
        "agri_pct": float(s06["agri_pct"]),
        "yellow_pct": float(s04["yellow_pct"]),
    }


def main():
    parser = argparse.ArgumentParser(description="KEMET1 full 8-step pipeline (fallback mode)")
    parser.add_argument("--site", help="Site name — loads data/KEMET1_BeforeAfter/siteN_before/after.tif")
    parser.add_argument("--before", help="Path to before GeoTIFF")
    parser.add_argument("--after",  help="Path to after GeoTIFF")
    args = parser.parse_args()

    if args.site:
        before = DATA_DIR / f"{args.site}_before_2024.tif"
        after  = DATA_DIR / f"{args.site}_after_2025.tif"
        site   = args.site
    elif args.before and args.after:
        before = Path(args.before)
        after  = Path(args.after)
        site   = args.site or "custom"
    else:
        parser.error("Provide --site or both --before and --after")

    result = run_pipeline(before, after, site)
    print("\nSummary:", result)


if __name__ == "__main__":
    main()
