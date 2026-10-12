import numpy as np
import pytest

from encroachment import features


def test_feature_names_and_count(tile_pair):
    values = features.pair_features(*tile_pair)
    assert values.shape == (44,) == (len(features.FEATURE_NAMES),)
    assert features.FEATURE_NAMES[0] == "d_NDVI_mean" and features.FEATURE_NAMES[-1] == "mean_dNDBI"
    assert np.isfinite(values).all()


def test_band_stats_known_values():
    band = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
    mean, std, p10, p25, p50, p75, p90 = features.band_stats(band)
    assert mean == pytest.approx(50.5) and std == pytest.approx(np.std(np.arange(1, 101)))
    assert (p10, p25, p50, p75, p90) == pytest.approx((10.9, 25.75, 50.5, 75.25, 90.1))


def test_band_stats_ignore_missing_pixels():
    band = np.array([[[1.0, np.nan], [3.0, np.inf]]], dtype=np.float32)
    assert features.band_stats(band)[0] == pytest.approx(2.0)


def test_features_describe_change_not_state(tile_pair):
    before, _ = tile_pair
    np.testing.assert_allclose(features.pair_features(before, before), 0.0, atol=1e-7)


def test_mean_index_changes(tile_pair):
    before, after = tile_pair
    values = dict(zip(features.FEATURE_NAMES, features.pair_features(before, after), strict=True))
    assert values["mean_dNDVI"] == pytest.approx(float((after[0] - before[0]).mean()), rel=1e-5)
    assert values["mean_dNDVI"] < 0 < values["mean_dNDBI"]


def test_conversion_mask_marks_the_converted_block(tile_pair):
    mask = features.conversion_mask(*tile_pair)
    assert mask.sum() == 900 and mask[40:70, 40:70].all()


def test_label_rule():
    assert features.label_rule(0.03, 0.0) and features.label_rule(0.0, 3.5)
    assert not features.label_rule(0.02, 3.0)


def test_read_pair(tile_files):
    pair = features.read_pair(*tile_files)
    assert pair.before.shape == (6, 120, 120) and pair.pixel_area_ha == pytest.approx(0.01)
    assert (pair.before_year, pair.after_year) == ("2024", "2025")
    lat, lon = pair.centre
    assert 30 < lat < 31 and 30 < lon < 32
    assert pair.tile_area_ha == pytest.approx(144.0)


def test_read_pair_rejects_bad_input(tmp_path, tile_pair):
    from conftest import write_geotiff

    before, after = tile_pair
    b = write_geotiff(tmp_path / "b.tif", before)
    with pytest.raises(ValueError, match="shape"):
        features.read_pair(b, write_geotiff(tmp_path / "small.tif", after[:, :60, :60]))
    three = write_geotiff(tmp_path / "rgb.tif", before[:3])
    with pytest.raises(ValueError, match="bands"):
        features.read_pair(three, three)
