"""Risk tiers. The web app, the site reports and the backend uploader all use these.

A site's risk comes from the model's probability of encroachment alone.
Earlier versions blended in the tile's mean NDVI loss (0.65 x RF + 0.35 x NDVI
loss); with the logistic model that blend no longer changes the ranking
(cross-validated AUC 0.8822 without it, 0.8825 with it), so it was dropped.

Both thresholds were chosen on cross-validated predictions for the 255
development sites, before the test sites were scored (results/evaluation.json):

* Medium ("check this site") - the highest threshold that still catches 90 %
  of flagged sites.
* High ("likely encroachment") - the lowest threshold above that at which 75 %
  of alarms are correct.
"""

from __future__ import annotations

import numpy as np

MEDIUM_THRESHOLD = 0.1627
HIGH_THRESHOLD = 0.5149


def risk_level(probability: float) -> str:
    """'High', 'Medium' or 'Low'."""
    if probability >= HIGH_THRESHOLD:
        return "High"
    if probability >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def mean_ndvi_loss(before: np.ndarray, after: np.ndarray) -> float:
    """Tile-average NDVI loss (gains count as zero). Reported for context, not used in the risk."""
    return float(np.nanmean(np.maximum(before[0] - after[0], 0)))
