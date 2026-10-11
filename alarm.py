"""alarm.py - The one definition of the alarm score and its tiers.

app.py, run_inference.py, upload_case.py and batch_upload.py all import from
here, so a site gets the same score and the same risk level whichever way it
is processed.

    fusion = 0.65 x RF probability + 0.35 x mean NDVI loss

Thresholds were chosen on the 45-site validation split (never the test split);
see results/repeated_splits.json and the README for what they deliver.
"""
from __future__ import annotations

import numpy as np

RF_WEIGHT = 0.65
SPECTRAL_WEIGHT = 0.35

#: "High" / red alert. On validation 0.30 catches 7 of 12 flagged sites with
#: 4 false alarms; the previous 0.40 caught 4 with 2, i.e. the same precision
#: for far lower recall.
ALERT_THRESHOLD = 0.30
#: "Medium" / yellow alert. Close to the F1-optimal point on validation.
YELLOW_THRESHOLD = 0.23


def spectral_score(d1: np.ndarray, d2: np.ndarray) -> float:
    """Mean NDVI loss over the tile (band 0 is NDVI); gains count as zero."""
    return float(np.nanmean(np.maximum(d1[0] - d2[0], 0)))


def fusion_score(rf_probability: float, d1: np.ndarray, d2: np.ndarray) -> float:
    return RF_WEIGHT * float(rf_probability) + SPECTRAL_WEIGHT * spectral_score(d1, d2)


def risk_level(fusion: float) -> str:
    """'High', 'Medium' or 'Low' for a fusion score."""
    if fusion >= ALERT_THRESHOLD:
        return "High"
    if fusion >= YELLOW_THRESHOLD:
        return "Medium"
    return "Low"
