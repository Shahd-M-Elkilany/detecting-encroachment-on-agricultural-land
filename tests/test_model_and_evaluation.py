"""Checks against the committed data: the shipped model must reproduce the published numbers."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from encroachment import alarm, data, evaluation, features
from encroachment import model as model_lib

REPORT = json.loads((data.RESULTS / "evaluation.json").read_text())


@pytest.fixture(scope="module")
def table():
    return data.load_features()


def test_feature_table_matches_labels(table):
    labels = data.load_labels()
    assert table["site"].tolist() == [r["site"] for r in labels]
    assert table["label"].tolist() == [int(r["label"] == "pos") for r in labels]
    assert len(table) == 300 and table["label"].sum() == 78


def test_labels_are_exactly_the_threshold_rule(table):
    rule = [features.label_rule(p, m) for p, m in zip(table["pct_converted"], table["max_cluster_ha"], strict=True)]
    assert table["label"].astype(bool).tolist() == rule


def test_test_split_is_the_original_one(table):
    """Same 45 test sites as v1 (results/v1/ba_eval.json), so the comparison is like for like."""
    _, y = data.xy(table)
    dev, test = evaluation.dev_test_split(y)
    v1 = {s["site"] for s in json.loads((data.RESULTS / "v1/ba_eval.json").read_text())["per_site"]}
    assert set(table["site"].to_numpy()[test]) == v1
    assert len(dev) == 255 and len(set(dev) & set(test)) == 0


def test_shipped_model_reproduces_the_reported_test_auc(table):
    X, y = data.xy(table)
    _, test = evaluation.dev_test_split(y)
    shipped = model_lib.load(data.MODEL)
    p = model_lib.probability(shipped, X[test])
    assert roc_auc_score(y[test], p) == pytest.approx(REPORT["test"]["auc"], abs=1e-4)
    meta = json.loads(data.MODEL.with_suffix(".json").read_text())
    assert sorted(table["site"].to_numpy()[test]) == meta["test_sites"]
    assert meta["features"] == list(features.FEATURE_NAMES)


def test_training_is_deterministic(table):
    X, y = data.xy(table)
    dev, _ = evaluation.dev_test_split(y)
    shipped = model_lib.load(data.MODEL)
    refit = model_lib.fit(X[dev], y[dev])
    np.testing.assert_allclose(refit[-1].coef_, shipped[-1].coef_, rtol=1e-6)


def test_alarm_thresholds_are_the_ones_chosen_on_development_sites():
    chosen = REPORT["development_out_of_fold"]["thresholds"]
    assert (chosen["medium"], chosen["high"]) == (alarm.MEDIUM_THRESHOLD, alarm.HIGH_THRESHOLD)


def test_counts():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.6])
    c = evaluation.counts(y, s, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["precision"] == pytest.approx(2 / 3, abs=1e-4) and c["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_choose_thresholds_meets_its_targets():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    p = np.clip(0.35 * y + rng.normal(0.3, 0.15, 400), 0, 1)
    t = evaluation.choose_thresholds(y, p)
    assert t["medium"] < t["high"]
    assert evaluation.counts(y, p, t["medium"])["recall"] >= evaluation.MEDIUM_TARGET_RECALL
    assert evaluation.counts(y, p, t["high"])["precision"] >= evaluation.HIGH_TARGET_PRECISION


def test_spatial_folds_are_compact_and_ordered(table):
    lat, lon = table["lat"].to_numpy(), table["lon"].to_numpy()
    folds = evaluation.spatial_folds(lat, lon)
    assert set(folds) == set(range(5))
    assert np.array_equal(folds, evaluation.spatial_folds(lat, lon)), "deterministic"
    assert REPORT["robustness_all_sites"]["spatial_folds"]["sites_per_fold"] == np.bincount(folds).tolist()


def test_bootstrap_interval():
    y = np.array([0] * 20 + [1] * 20)
    perfect = np.linspace(0, 1, 40)
    assert evaluation.bootstrap_auc(y, perfect, n_boot=200) == (1.0, 1.0)
