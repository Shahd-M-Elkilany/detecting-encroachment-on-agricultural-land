"""Tests run from the repository root and need no satellite tiles."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tile_pair():
    """A synthetic before/after pair shaped like the real tiles: 6 index bands, 10 m pixels.

    A 30 x 30 pixel block (9 ha) turns from vegetation into built-up land.
    """
    rng = np.random.default_rng(0)
    before = rng.normal(0.0, 0.02, size=(6, 120, 120)).astype(np.float32)
    before[0] += 0.6   # NDVI: vegetated everywhere
    before[1] -= 0.2   # NDBI: not built-up
    after = before + rng.normal(0.0, 0.01, size=before.shape).astype(np.float32)
    after[0, 40:70, 40:70] = 0.1    # vegetation lost
    after[1, 40:70, 40:70] = 0.2    # built-up signal gained
    return before, after


def write_geotiff(path, data):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype="float32", crs="EPSG:32636", transform=from_origin(300000, 3400000, 10, 10),
    ) as dst:
        dst.write(data)
    return path
