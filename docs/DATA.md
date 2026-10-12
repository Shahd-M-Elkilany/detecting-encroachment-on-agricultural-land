# Data

## Sites and tiles

300 sites across the Nile Delta, each a 262 × 262 pixel tile at 10 m (about 684 ha), imaged once in 2024 and
once in 2025. Every tile is a GeoTIFF (EPSG:32636) with six spectral-index bands derived from Sentinel-2 L2A:

| Band | Index | Formula (Sentinel-2 bands) |
| --- | --- | --- |
| 1 | NDVI | (B8 − B4) / (B8 + B4) |
| 2 | NDBI | (B11 − B8) / (B11 + B8) |
| 3 | MNDWI | (B3 − B11) / (B3 + B11) |
| 4 | SAVI | 1.5 (B8 − B4) / (B8 + B4 + 0.5) |
| 5 | BSI | bare-soil index |
| 6 | NDWI | water index |

The tiles were prepared by another member of the project team; the exact acquisition dates and compositing
settings were not recorded with them. `scripts/gee_sample_export.js` computes the same six indices in Google
Earth Engine for five sample locations. The tiles total 1.1 GB and are not stored in this repository; to run
the tile-level commands, place them in `data/tiles/` as `site{N}_before_2024.tif` and
`site{N}_after_2025.tif`.

## Files in this repository

| File | Contents |
| --- | --- |
| `data/labels.json` | Site name, label (`pos` / `neg`) and the two quantities the label was computed from |
| `data/features.csv` | One row per site: name, centre latitude and longitude, label, `pct_converted`, `max_cluster_ha`, `mean_ndvi_loss`, and the 44 model features |

`data/features.csv` is what training and evaluation read, so every number in the README can be reproduced
without the tiles. `kemet1 extract` regenerates it from the tiles; on the full set it reproduces the committed
table to within 5 × 10⁻⁶ relative error.

## Labels

Labels were produced automatically:

- `pct_converted`: share of pixels where NDVI was above 0.25 in 2024, below 0.25 in 2025, and NDBI rose by
  more than 0.08.
- `max_cluster_ha`: area of the largest 4-connected patch of such pixels.
- A site is positive when `pct_converted > 0.02576` or `max_cluster_ha > 3.13`.

This gives 78 positive and 222 negative sites. `kemet1 extract` recomputes both quantities from the tiles
and the rule then reproduces all 300 labels. No site has been checked by hand against very-high-resolution
imagery or field visits.

## Splits

- **Test:** 45 sites (11 positive), from a stratified 70/15/15 split with seed 42. These are the same test
  sites the original (v1) model was evaluated on, which makes the v1/v2 comparison like for like.
- **Development:** the other 255 sites (67 positive). Model choice, regularisation and alarm thresholds were
  all chosen here by repeated stratified 5-fold cross-validation.
- **Spatial folds:** for the robustness check, all 300 sites are grouped into five regions by k-means on their
  coordinates (`encroachment/evaluation.py`, `spatial_folds`), numbered south to north.

## Known issues

- Neighbouring sites can share land-cover patterns, so random splits are optimistic; the spatial
  cross-validation in the README is the better estimate for new regions.
- A tile-wide harvest or fallow between the two dates looks like vegetation loss.
- Building on land that was already bare in 2024 barely changes NDVI and is under-detected.
