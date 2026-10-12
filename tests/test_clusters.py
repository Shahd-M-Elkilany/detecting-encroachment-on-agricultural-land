"""Converted-pixel clustering used for the report boxes and area figures."""
import pytest

import run_inference


def test_report_clusters(tile_pair):
    clusters = run_inference.find_clusters(*tile_pair)
    assert len(clusters) == 1
    row0, row1, col0, col1, hectares = clusters[0]
    assert (row0, row1, col0, col1) == (40, 69, 40, 69)
    assert hectares == pytest.approx(9.0)  # 900 pixels x 100 m2


def test_no_change_gives_no_clusters(tile_pair):
    before, _ = tile_pair
    assert run_inference.find_clusters(before, before) == []


def test_small_patches_are_dropped(tile_pair):
    before, after = tile_pair
    after = before.copy()
    after[0, 10:15, 10:15] = 0.1  # 25 pixels = 0.25 ha, below the 0.5 ha minimum
    after[1, 10:15, 10:15] = 0.2
    assert run_inference.find_clusters(before, after) == []
    assert len(run_inference.find_clusters(before, after, min_ha=0.2)) == 1


def test_web_app_clusters_match_the_report(tile_pair):
    app = pytest.importorskip("app")
    areas, boxes = app._find_clusters(*tile_pair, px_ha=0.01)
    assert areas == [pytest.approx(9.0)] and boxes == [(40, 40, 70, 70)]
