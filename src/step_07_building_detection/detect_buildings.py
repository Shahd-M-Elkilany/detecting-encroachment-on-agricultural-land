"""
Step 07 — Building Detection
Uses YOLOv8-seg to detect buildings, then optionally refines with SAM.
Only runs on pixels flagged by both the change map AND the agri mask.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from config.settings import BUILDING_DETECTION_CONFIG, PROCESSED_DIR
from src.utils.logger import get_logger
from src.utils.geo_utils import write_geotiff

logger = get_logger("step_07")
CFG = BUILDING_DETECTION_CONFIG


def run(
    t2_image:   np.ndarray,
    change_map: np.ndarray,
    agri_mask:  np.ndarray,
    meta:       Dict[str, Any],
    t1_image:   np.ndarray = None,
) -> Dict[str, Any]:
    """
    Detect buildings on changed agricultural land.

    Args:
        t1_image: optional before-image for NDBI_delta fallback (5-10x more sensitive).

    Returns:
        building_mask: binary [H,W] uint8
        polygons:      list of dicts with pixel coordinates and confidence
    """
    roi_mask = (change_map > 0) & (agri_mask > 0)
    roi_pct = float(roi_mask.mean() * 100)
    logger.info(f"Region of interest (changed agri land): {roi_pct:.2f}% of image")

    if roi_mask.sum() == 0:
        logger.warning("No changed agricultural pixels -- skipping building detection")
        H, W = change_map.shape
        return {"building_mask": np.zeros((H, W), dtype=np.uint8), "polygons": []}

    try:
        building_mask, polygons = _run_yolo(t2_image, roi_mask, meta)
    except Exception as e:
        logger.warning(f"YOLO detection failed ({e}). Using morphological fallback.")
        building_mask, polygons = _morphological_fallback(t2_image, roi_mask, t1_image)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_geotiff(PROCESSED_DIR / "building_mask.tif", building_mask, meta)
    logger.info(f"Buildings detected: {building_mask.sum():,} pixels, {len(polygons)} polygons")

    return {"building_mask": building_mask, "polygons": polygons}


def _run_yolo(image, roi_mask, meta):
    from ultralytics import YOLO
    import cv2

    weights = Path(CFG["yolo_weights"])
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))
    rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    rgb = np.clip(rgb / (rgb.max() + 1e-8) * 255, 0, 255).astype(np.uint8)

    H, W = image.shape[1], image.shape[2]
    building_mask = np.zeros((H, W), dtype=np.uint8)
    polygons = []

    results = model.predict(rgb, conf=CFG["yolo_conf"], iou=CFG["yolo_iou"], verbose=False)
    for res in results:
        if res.masks is None:
            continue
        for mask_xy, box, conf in zip(res.masks.xy, res.boxes.xyxy, res.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            if roi_mask[y1:y2, x1:x2].mean() < 0.1:
                continue
            pts = mask_xy.astype(np.int32)
            cv2.fillPoly(building_mask, [pts], 1)
            polygons.append({"bbox_px": [x1, y1, x2, y2], "confidence": float(conf), "contour": pts.tolist()})

    building_mask = (building_mask & roi_mask).astype(np.uint8)
    return building_mask, polygons


def _morphological_fallback(image, roi_mask, t1_image=None):
    """
    Fallback: NDBI-based built-up detection within the ROI.
    With t1_image: NDBI_delta (after-before) > 0.05  [5-10x more sensitive]
    Without:       NDBI_after > 0.1
    """
    import cv2
    from config.settings import SPECTRAL_INDICES_CONFIG
    precomputed = SPECTRAL_INDICES_CONFIG.get("precomputed_indices", False)

    if precomputed:
        ndbi_after = image[1].astype(np.float32)
        if t1_image is not None:
            ndbi_before = t1_image[1].astype(np.float32)
            ndbi = ndbi_after - ndbi_before
            threshold = 0.05
            logger.info("Building fallback: NDBI_delta > 0.05 (precomputed bands)")
        else:
            ndbi = ndbi_after
            threshold = 0.1
            logger.info("Building fallback: NDBI_after > 0.1 (precomputed bands, no t1)")
    else:
        def _calc_ndbi(img):
            nir = img[3] if img.shape[0] > 3 else img[-1]
            swir1 = img[4] if img.shape[0] > 4 else img[-1]
            denom = nir + swir1
            return np.where(denom > 0, (swir1 - nir) / denom, 0.0).astype(np.float32)

        if t1_image is not None:
            ndbi = _calc_ndbi(image) - _calc_ndbi(t1_image)
            threshold = 0.05
            logger.info("Building fallback: NDBI_delta > 0.05 (raw bands)")
        else:
            ndbi = _calc_ndbi(image)
            threshold = 0.1
            logger.info("Building fallback: NDBI_after > 0.1 (raw bands, no t1)")

    raw = ((ndbi > threshold) & (roi_mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 25:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        polygons.append({"bbox_px": [x, y, x + w, y + h], "confidence": 0.5, "contour": cnt.squeeze().tolist()})

    return cleaned, polygons
