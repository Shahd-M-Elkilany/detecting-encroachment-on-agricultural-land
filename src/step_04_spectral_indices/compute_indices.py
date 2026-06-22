"""
Step 04 — Spectral Indices
Computes NDVI, NDBI, MNDWI for T1 and T2, and derives a spectral
degradation signal used later as the yellow-alert input.
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np

from config.settings import SPECTRAL_INDICES_CONFIG
from src.utils.logger import get_logger

logger = get_logger("step_04")

CFG = SPECTRAL_INDICES_CONFIG


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), masked where denominator is zero."""
    denom = a + b
    return np.where(denom != 0, (a - b) / denom, 0.0).astype(np.float32)


def compute_indices(image: np.ndarray) -> Dict[str, np.ndarray]:
    nir   = image[CFG["band_nir"]]
    red   = image[CFG["band_red"]]
    green = image[CFG["band_green"]]
    swir1 = image[CFG["band_swir1"]]

    ndvi  = _safe_ratio(nir, red)      # vegetation health: high = healthy
    ndbi  = _safe_ratio(swir1, nir)    # built-up: high = buildings
    mndwi = _safe_ratio(green, swir1)  # water: high = water

    return {"ndvi": ndvi, "ndbi": ndbi, "mndwi": mndwi}


def run(
    t1_image: np.ndarray,
    t2_image: np.ndarray,
    t1_meta: Dict[str, Any],
    t2_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute indices for T1 and T2, then derive a spectral degradation signal.

    The degradation signal is used as the yellow-alert input in Step 08:
      - NDVI drop  (vegetation loss)
      - NDBI rise  (built-up increase)
    Both components normalised to [0,1] and averaged.
    """
    logger.info("Computing spectral indices for T1 and T2 ...")
    t1_idx = compute_indices(t1_image)
    t2_idx = compute_indices(t2_image)

    ndvi_drop = t1_idx["ndvi"] - t2_idx["ndvi"]   # positive = vegetation lost
    ndbi_rise = t2_idx["ndbi"] - t1_idx["ndbi"]   # positive = more built-up

    def norm01(arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, 0, None)
        mx = arr.max()
        return (arr / mx).astype(np.float32) if mx > 0 else arr

    spectral_signal = (norm01(ndvi_drop) + norm01(ndbi_rise)) / 2.0

    # Yellow alert mask: pixels where degradation threshold is exceeded
    ndvi_threshold = abs(CFG["ndvi_degradation_threshold"])
    yellow_mask = (ndvi_drop > ndvi_threshold).astype(np.uint8)

    yellow_pct = float(yellow_mask.mean() * 100)
    logger.info(f"Spectral degradation: {yellow_pct:.2f}% of pixels flagged yellow")

    return {
        "T1": t1_idx,
        "T2": t2_idx,
        "ndvi_drop":        ndvi_drop,
        "ndbi_rise":        ndbi_rise,
        "spectral_signal":  spectral_signal,   # [0,1] float map
        "yellow_mask":      yellow_mask,        # binary
        "yellow_pct":       yellow_pct,
    }
