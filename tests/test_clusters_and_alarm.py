import pytest

from encroachment import alarm, clusters


def test_one_cluster_with_area_box_and_score(tile_pair):
    found = clusters.find_clusters(*tile_pair, pixel_area_ha=0.01)
    assert len(found) == 1
    c = found[0]
    assert c.area_ha == pytest.approx(9.0) and c.box == (40, 40, 70, 70)
    assert c.centre_px == pytest.approx((54.5, 54.5))
    # built-up gain ~0.4, NDVI drop ~0.5
    assert c.score == pytest.approx(0.65 * 0.4 + 0.35 * 0.5, abs=0.02)


def test_small_patches_are_ignored(tile_pair):
    before, _ = tile_pair
    after = before.copy()
    after[0, 10:15, 10:15], after[1, 10:15, 10:15] = 0.1, 0.2  # 0.25 ha
    assert clusters.find_clusters(before, after) == []
    assert len(clusters.find_clusters(before, after, min_ha=0.2)) == 1


def test_largest_cluster_first():
    from conftest import make_pair

    before, after = make_pair(block=(40, 50))
    after[0, 80:110, 80:110], after[1, 80:110, 80:110] = 0.1, 0.2
    found = clusters.find_clusters(before, after)
    assert [c.area_ha for c in found] == [9.0, 1.0]


def test_cluster_colours_are_relative_to_the_strongest():
    make = lambda s: clusters.Cluster(1.0, (0, 0, 1, 1), s, (0.0, 0.0))  # noqa: E731
    assert make(0.9).colour(1.0) == "#da3633"
    assert make(0.5).colour(1.0) == "#ff6b00"
    assert make(0.1).colour(1.0) == "#d29922"
    assert make(0.0).colour(0.0) == "#d29922"


def test_pixel_to_lonlat():
    lat, lon = clusters.pixel_to_lonlat(0, 0, (100, 200), (31.0, 30.0, 31.2, 30.1))
    assert (lat, lon) == pytest.approx((30.1, 31.0))
    lat, lon = clusters.pixel_to_lonlat(100, 200, (100, 200), (31.0, 30.0, 31.2, 30.1))
    assert (lat, lon) == pytest.approx((30.0, 31.2))


def test_risk_levels_at_the_boundaries():
    assert 0 < alarm.MEDIUM_THRESHOLD < alarm.HIGH_THRESHOLD < 1
    assert alarm.risk_level(alarm.HIGH_THRESHOLD) == "High"
    assert alarm.risk_level(alarm.HIGH_THRESHOLD - 1e-6) == "Medium"
    assert alarm.risk_level(alarm.MEDIUM_THRESHOLD) == "Medium"
    assert alarm.risk_level(alarm.MEDIUM_THRESHOLD - 1e-6) == "Low"


def test_mean_ndvi_loss_ignores_gains():
    import numpy as np

    before = np.full((6, 10, 10), 0.6, np.float32)
    after = before.copy()
    after[0, :5], after[0, 5:] = 0.2, 0.9
    assert alarm.mean_ndvi_loss(before, after) == pytest.approx(0.2)
