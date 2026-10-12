"""Shared fixtures. Everything runs on synthetic tiles plus the committed feature table."""

from __future__ import annotations

import numpy as np
import pytest


def make_pair(seed: int = 0, size: int = 120, block=(40, 70)):
    """A vegetated tile in which a block of pixels turns into built-up land (9 ha by default)."""
    rng = np.random.default_rng(seed)
    before = rng.normal(0.0, 0.02, size=(6, size, size)).astype(np.float32)
    before[0] += 0.6  # NDVI high: vegetated
    before[1] -= 0.2  # NDBI low: not built-up
    after = before + rng.normal(0.0, 0.01, size=before.shape).astype(np.float32)
    a, b = block
    after[0, a:b, a:b] = 0.1
    after[1, a:b, a:b] = 0.2
    return before, after


@pytest.fixture
def tile_pair():
    return make_pair()


def write_geotiff(path, data, west=300_000, north=3_400_000):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(west, north, 10, 10),
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def tile_files(tmp_path, tile_pair):
    before, after = tile_pair
    return (
        write_geotiff(tmp_path / "siteX_before_2024.tif", before),
        write_geotiff(tmp_path / "siteX_after_2025.tif", after),
    )
