"""Tile reading and the 44 spectral-change features.

Every tile is a 6-band GeoTIFF of spectral indices at 10 m:
NDVI, NDBI, MNDWI, SAVI, BSI, NDWI (band order below). A site is a
co-registered pair: before (2024) and after (2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BANDS = ("NDVI", "NDBI", "MNDWI", "SAVI", "BSI", "NDWI")
STATS = ("mean", "std", "p10", "p25", "p50", "p75", "p90")
FEATURE_NAMES = tuple(f"d_{b}_{s}" for b in BANDS for s in STATS) + ("mean_dNDVI", "mean_dNDBI")
N_FEATURES = len(FEATURE_NAMES)  # 44

#: Thresholds of the automatic labelling rule (see docs/DATA.md).
LABEL_PCT_CONVERTED = 0.02576
LABEL_MAX_CLUSTER_HA = 3.13


@dataclass(frozen=True)
class TilePair:
    before: np.ndarray  # (6, H, W) float32
    after: np.ndarray  # (6, H, W) float32
    pixel_area_ha: float
    bounds_wgs84: tuple[float, float, float, float]  # west, south, east, north
    before_year: str = "before"
    after_year: str = "after"

    @property
    def centre(self) -> tuple[float, float]:
        """(lat, lon) of the tile centre."""
        w, s, e, n = self.bounds_wgs84
        return (s + n) / 2, (w + e) / 2

    @property
    def tile_area_ha(self) -> float:
        return float(self.before.shape[1] * self.before.shape[2] * self.pixel_area_ha)


def _year(path: Path, default: str) -> str:
    tail = Path(path).stem.split("_")[-1]
    return tail if tail.isdigit() else default


def read_pair(before_path: str | Path, after_path: str | Path) -> TilePair:
    """Read a before/after GeoTIFF pair and check that they line up."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(before_path) as src:
        before = src.read().astype(np.float32)
        res = src.res
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    with rasterio.open(after_path) as src:
        after = src.read().astype(np.float32)
    if before.shape != after.shape:
        raise ValueError(f"before and after tiles differ in shape: {before.shape} vs {after.shape}")
    if before.shape[0] < len(BANDS):
        raise ValueError(f"expected {len(BANDS)} index bands, got {before.shape[0]}")
    return TilePair(
        before,
        after,
        abs(res[0] * res[1]) / 10_000,
        tuple(bounds),
        _year(before_path, "before"),
        _year(after_path, "after"),
    )


def band_stats(arr: np.ndarray) -> np.ndarray:
    """Mean, std and 10/25/50/75/90th percentiles of every band, ignoring non-finite pixels."""
    out: list[float] = []
    for band in arr:
        values = band[np.isfinite(band)]
        out += [values.mean(), values.std(), *np.percentile(values, (10, 25, 50, 75, 90))]
    return np.asarray(out, dtype=np.float64)


def pair_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """The 44 features: statistics of every band difference, then mean dNDVI and dNDBI."""
    diff = after - before
    return np.concatenate([band_stats(diff), [np.nanmean(diff[0]), np.nanmean(diff[1])]])


def conversion_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Pixels that went from vegetated to built-up: NDVI falls below 0.25 and NDBI rises by > 0.08."""
    return (before[0] > 0.25) & (after[0] < 0.25) & (after[1] > before[1] + 0.08)


def label_rule(pct_converted: float, max_cluster_ha: float) -> bool:
    """The automatic labelling rule that produced data/ba_labels.json."""
    return pct_converted > LABEL_PCT_CONVERTED or max_cluster_ha > LABEL_MAX_CLUSTER_HA
