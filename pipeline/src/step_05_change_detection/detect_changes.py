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
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ChangeFormer expects [B,C,H,W] pairs
    t1_t = torch.from_numpy(t1).unsqueeze(0).to(device)
    t2_t = torch.from_numpy(t2).unsqueeze(0).to(device)

    # Dynamically import ChangeFormer (must be on PYTHONPATH)
    from models.ChangeFormer import ChangeFormerV6

    model = ChangeFormerV6()
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    with torch.no_grad():
        logits = model(t1_t, t2_t)  # [B,2,H,W]
        probs = torch.softmax(logits, dim=1)[:, 1].squeeze().cpu().numpy()

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
