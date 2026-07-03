"""
Step 03 — Cloud Removal
Uses OpenCV Telea inpainting to fill cloud-masked pixels.
This step is only called by the pipeline when cloud coverage >= 15%.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config.settings import CLOUD_REMOVAL_CONFIG, PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("step_03")


def run(
    t1_path: Path,
    t2_path: Path,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Remove clouds from T1 and T2 images.

    Returns dict with cleaned images and their metadata:
        {"T1": {"image": ndarray, "meta": dict}, "T2": {...}}
    """
    from src.utils.geo_utils import read_geotiff, write_geotiff

    t1_data, t1_meta = read_geotiff(t1_path)
    t2_data, t2_meta = read_geotiff(t2_path)

    t1_clean = _inpaint_image(t1_data, t1_mask, label="T1")
    t2_clean = _inpaint_image(t2_data, t2_mask, label="T2")

    # Persist to disk for downstream steps / resuming
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_geotiff(PROCESSED_DIR / "T1_cloud_free.tif", t1_clean, t1_meta)
    write_geotiff(PROCESSED_DIR / "T2_cloud_free.tif", t2_clean, t2_meta)
    logger.info("Cloud-free images written to data/processed/")

    return {
        "T1": {"image": t1_clean, "meta": t1_meta},
        "T2": {"image": t2_clean, "meta": t2_meta},
    }


def _inpaint_image(image: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    """Inpaint each band independently using the cloud mask."""
    import cv2

    method = (
        cv2.INPAINT_TELEA
        if CLOUD_REMOVAL_CONFIG["inpaint_method"] == "telea"
        else cv2.INPAINT_NS
    )
    radius = CLOUD_REMOVAL_CONFIG["inpaint_radius"]

    # Mask must be uint8 for cv2
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    cloudy_pixels = int(mask_u8.sum() / 255)
    logger.info(f"[{label}] Inpainting {cloudy_pixels:,} cloud pixels across {image.shape[0]} bands")

    result = np.zeros_like(image)
    for b in range(image.shape[0]):
        band = image[b]
        # cv2 inpaint expects 8-bit or 16-bit; normalise to 16-bit
        band_min, band_max = band.min(), band.max()
        rng = band_max - band_min
        if rng > 0:
            band_16 = ((band - band_min) / rng * 65535).astype(np.uint16)
        else:
            band_16 = np.zeros_like(band, dtype=np.uint16)

        inpainted_16 = cv2.inpaint(band_16, mask_u8, radius, method)
        # Back to float32
        result[b] = (inpainted_16.astype(np.float32) / 65535) * rng + band_min

    return result
