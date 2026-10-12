"""
Step 01 — Data Acquisition
Downloads satellite imagery from GEE, or loads existing GeoTIFFs offline.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

from config.settings import GEE_CONFIG, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("step_01")


def run() -> Dict[str, Path]:
    """Download T1 and T2 imagery from Google Earth Engine."""
    try:
        import ee
        ee.Initialize(project=GEE_CONFIG["project"])
    except Exception as e:
        raise RuntimeError(
            f"GEE initialisation failed: {e}\n"
            "Run `earthengine authenticate` and set your project in config/settings.py"
        )

    region = ee.Geometry.Rectangle(GEE_CONFIG["region"])
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(GEE_CONFIG["bands"])
    )

    dates = sorted(
        collection.aggregate_array("system:time_start").getInfo()
    )
    if len(dates) < 2:
        raise ValueError("Not enough Sentinel-2 scenes found for this region/period.")

    t1_img = collection.filter(
        ee.Filter.date(ee.Date(dates[0]), ee.Date(dates[len(dates) // 2]))
    ).median()
    t2_img = collection.filter(
        ee.Filter.date(ee.Date(dates[len(dates) // 2]), ee.Date(dates[-1]))
    ).median()

    t1_path = RAW_DIR / "T1" / "T1.tif"
    t2_path = RAW_DIR / "T2" / "T2.tif"
    t1_path.parent.mkdir(parents=True, exist_ok=True)
    t2_path.parent.mkdir(parents=True, exist_ok=True)

    geemap_export(t1_img, region, GEE_CONFIG["scale"], str(t1_path))
    geemap_export(t2_img, region, GEE_CONFIG["scale"], str(t2_path))

    logger.info(f"T1 saved: {t1_path}")
    logger.info(f"T2 saved: {t2_path}")
    return {"T1": t1_path, "T2": t2_path}


def geemap_export(image, region, scale: int, out_path: str) -> None:
    import geemap
    geemap.ee_export_image(image, filename=out_path, scale=scale, region=region, file_per_band=False)


def run_offline(t1_path: str | Path, t2_path: str | Path) -> Dict[str, Path]:
    """Use existing GeoTIFF files instead of downloading."""
    t1 = Path(t1_path)
    t2 = Path(t2_path)
    if not t1.exists():
        raise FileNotFoundError(f"T1 image not found: {t1}")
    if not t2.exists():
        raise FileNotFoundError(f"T2 image not found: {t2}")
    logger.info(f"Offline mode — T1: {t1}, T2: {t2}")
    return {"T1": t1, "T2": t2}
