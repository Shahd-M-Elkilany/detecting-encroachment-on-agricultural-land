"""The 44-feature vector is re-implemented in five scripts. They must never drift apart."""
import numpy as np
import pytest

import ablation_no_circular
import evaluate_ba
import geographic_holdout
import run_inference

IMPLEMENTATIONS = {
    "evaluate_ba": evaluate_ba.pair_features,
    "geographic_holdout": geographic_holdout.pair_features,
    "ablation_no_circular": ablation_no_circular.pair_features_ablation,
    "run_inference": run_inference.pair_features,
}


def test_feature_vector_has_44_values(tile_pair):
    features = evaluate_ba.pair_features(*tile_pair)
    assert features.shape == (44,) and np.isfinite(features).all()


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_script_computes_the_same_features(name, tile_pair):
    reference = evaluate_ba.pair_features(*tile_pair)
    np.testing.assert_allclose(IMPLEMENTATIONS[name](*tile_pair), reference, rtol=1e-6)


def test_web_app_computes_the_same_features(tile_pair):
    app = pytest.importorskip("app")
    np.testing.assert_allclose(app._pair_features(*tile_pair), evaluate_ba.pair_features(*tile_pair), rtol=1e-6)


def test_extract_stats_known_values():
    band = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
    mean, std, p10, p25, p50, p75, p90 = evaluate_ba.extract_stats(band)
    assert mean == pytest.approx(50.5) and std == pytest.approx(np.std(np.arange(1, 101)))
    assert (p10, p25, p50, p75, p90) == pytest.approx((10.9, 25.75, 50.5, 75.25, 90.1))


def test_extract_stats_ignores_missing_pixels():
    band = np.array([[[1.0, np.nan], [3.0, np.inf]]], dtype=np.float32)
    assert evaluate_ba.extract_stats(band)[0] == pytest.approx(2.0)


def test_features_describe_change_not_state(tile_pair):
    """Identical tiles give all-zero features, whatever the land cover is."""
    before, _ = tile_pair
    np.testing.assert_allclose(evaluate_ba.pair_features(before, before), 0.0, atol=1e-7)


def test_last_two_features_are_mean_ndvi_and_ndbi_change(tile_pair):
    before, after = tile_pair
    features = evaluate_ba.pair_features(before, after)
    assert features[42] == pytest.approx(float((after[0] - before[0]).mean()), rel=1e-5)
    assert features[43] == pytest.approx(float((after[1] - before[1]).mean()), rel=1e-5)
    assert features[42] < 0 < features[43], "vegetation fell and built-up signal rose"


def test_leaky_features_are_only_in_the_ablation_baseline(tile_pair):
    full = ablation_no_circular.pair_features_full(*tile_pair)
    assert full.shape == (46,)
    np.testing.assert_allclose(full[:44], ablation_no_circular.pair_features_ablation(*tile_pair), rtol=1e-6)
    assert full[44] == pytest.approx(900 / 14400, abs=1e-3), "pct_conv is the converted share of the tile"


def test_bootstrap_interval():
    y = np.array([0] * 20 + [1] * 20)
    perfect = np.concatenate([np.linspace(0, 0.4, 20), np.linspace(0.6, 1, 20)])
    assert evaluate_ba.bootstrap_auc(y, perfect, n_boot=200) == (1.0, 1.0)
    noisy = np.random.default_rng(0).random(40)
    lo, hi = evaluate_ba.bootstrap_auc(y, noisy, n_boot=200)
    assert 0 <= lo < hi <= 1
    assert evaluate_ba.bootstrap_auc(y, noisy, n_boot=200) == (lo, hi), "seeded, so repeatable"
