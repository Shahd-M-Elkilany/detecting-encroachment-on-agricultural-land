"""Connected areas of farmland-to-built-up conversion within a tile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .features import conversion_mask

#: Clusters smaller than this are ignored (0.5 ha = 50 pixels at 10 m).
MIN_CLUSTER_HA = 0.5


@dataclass(frozen=True)
class Cluster:
    area_ha: float
    box: tuple[int, int, int, int]  # row0, col0, row1, col1 (end-exclusive) in pixels
    score: float  # 0.65 x built-up increase + 0.35 x NDVI drop, cluster mean
    centre_px: tuple[float, float]  # row, col

    def colour(self, max_score: float) -> str:
        """Red / orange / yellow by score relative to the strongest cluster in the tile."""
        if max_score <= 1e-9:
            return "#d29922"
        ratio = self.score / max_score
        if ratio >= 0.67:
            return "#da3633"
        if ratio >= 0.33:
            return "#ff6b00"
        return "#d29922"


def find_clusters(
    before: np.ndarray, after: np.ndarray, pixel_area_ha: float = 0.01, min_ha: float = MIN_CLUSTER_HA
) -> list[Cluster]:
    """Converted-pixel clusters of at least ``min_ha``, largest first."""
    labeled, n = ndimage.label(conversion_mask(before, after))
    out: list[Cluster] = []
    for index, slices in enumerate(ndimage.find_objects(labeled), start=1):
        pixels = labeled == index
        area = round(float(pixels.sum()) * pixel_area_ha, 2)
        if area < min_ha:
            continue
        ndvi_drop = float(np.nanmean(np.maximum(before[0][pixels] - after[0][pixels], 0)))
        builtup_gain = float(np.nanmean(np.maximum(after[1][pixels] - before[1][pixels], 0)))
        rows, cols = slices
        r, c = ndimage.center_of_mass(pixels)
        out.append(
            Cluster(
                area,
                (rows.start, cols.start, rows.stop, cols.stop),
                0.65 * builtup_gain + 0.35 * ndvi_drop,
                (float(r), float(c)),
            )
        )
    return sorted(out, key=lambda cl: -cl.area_ha)


def pixel_to_lonlat(
    row: float, col: float, shape: tuple[int, int], bounds_wgs84: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Approximate (lat, lon) of a pixel position, by linear interpolation over the tile bounds."""
    west, south, east, north = bounds_wgs84
    height, width = shape
    return north - row * (north - south) / height, west + col * (east - west) / width
