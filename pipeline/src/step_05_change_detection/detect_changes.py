"""
Step 05 — Change Detection

With ChangeFormer weights (``weights/ChangeFormer_LEVIR.pth``, LEVIR-CD) and the
ChangeFormer code on the path, a Siamese transformer compares the T1 and T2
images. Without them, and always for pre-computed index tiles (which ChangeFormer
was not trained on), a difference rule is used: the mean absolute change across
all bands, normalised by its tile maximum, thresholded at ``threshold``.
Returns a binary change map and a [0, 1] confidence map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from config.settings import CHANGE_DETECTION_CONFIG

from src.utils.logger import get_logger

logger = get_logger("step_05")
CFG = CHANGE_DETECTION_CONFIG


def run(
    t1_image: np.ndarray,
    t2_image: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect land-use changes between T1 and T2.

    Returns:
        change_map:        binary [H,W] uint8
        change_confidence: float [H,W] in [0,1]  (used for red-alert weight)
    """
    from config.settings import SPECTRAL_INDICES_CONFIG

    weights = Path(CFG["model_weights"])

    if SPECTRAL_INDICES_CONFIG.get("precomputed_indices", False):
        logger.info("Pre-computed index mode -> mean absolute band difference")
        change_map, confidence = _difference_fallback(t1_image, t2_image)
    elif weights.exists():
        logger.info("Running ChangeFormer model ...")
        change_map, confidence = _run_changeformer(t1_image, t2_image, weights)
    else:
        logger.warning(f"ChangeFormer weights not found at {weights}. Using difference-based fallback.")
        change_map, confidence = _difference_fallback(t1_image, t2_image)

    changed = int(change_map.sum())
    total = change_map.size
    logger.info(f"Changed pixels: {changed:,} / {total:,} ({100 * changed / total:.2f}%)")

    return {
        "change_map": change_map,
        "change_confidence": confidence,
        "meta": meta,
    }


def _run_changeformer(t1: np.ndarray, t2: np.ndarray, weights: Path) -> tuple[np.ndarray, np.ndarray]:
    """ChangeFormerV6 on the RGB of two raw-band images (bands B2, B3, B4, ...), in 256-pixel windows.

    Uses the loader and pre-processing checked in changeformer/run_changeformer.py: the
    checkpoint's ``model_G_state_dict``, a shared 2-98 % stretch of both dates and
    ChangeFormer's own (0.5, 0.5) normalisation.
    """
    import sys

    here = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(here / "changeformer"))
    from run_changeformer import load_model, predict

    net = load_model(CFG["code"], str(weights))
    rgb = [img[[2, 1, 0]].astype(np.float32) for img in (t1, t2)]
    both = np.concatenate([x.reshape(3, -1) for x in rgb], 1)
    lo, hi = np.percentile(both, 2, 1), np.percentile(both, 98, 1)
    h, w = t1.shape[1:]
    H, W = -(-h // 256) * 256, -(-w // 256) * 256
    a, b = (
        np.pad(
            np.clip(np.transpose((x - lo[:, None, None]) / (hi - lo + 1e-9)[:, None, None], (1, 2, 0)), 0, 1),
            ((0, H - h), (0, W - w), (0, 0)),
            mode="reflect",
        )
        for x in rgb
    )
    probs = predict(net, a, b)[:h, :w]
    mask = (probs > CFG["threshold"]).astype(np.uint8)
    return mask, probs.astype(np.float32)


def _difference_fallback(t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple mean-absolute-difference fallback when ChangeFormer weights are absent.
    Normalised to [0,1]; threshold applied to create binary mask.
    """
    diff = np.abs(t2 - t1).mean(axis=0)
    confidence = diff / (diff.max() + 1e-8)
    mask = (confidence > CFG["threshold"]).astype(np.uint8)
    return mask, confidence.astype(np.float32)
