"""Integrity of the files that ship with the repository."""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_labels_file():
    records = load("data/ba_labels.json")
    assert len(records) == 300 and len({r["site"] for r in records}) == 300
    assert sum(r["label"] == "pos" for r in records) == 78


def test_labels_are_exactly_the_documented_threshold_rule():
    """The README says the labels are automatic. This is the rule, and it explains every label."""
    for r in load("data/ba_labels.json"):
        rule = r["pct_converted"] > 0.02576 or r["max_cluster_ha"] > 3.13
        assert (r["label"] == "pos") == rule, r["site"]


def test_stored_evaluations_agree_with_each_other():
    clean, evaluation, geo = load("data/ba_clean_results.json"), load("data/ba_eval.json"), load("data/geo_holdout.json")
    assert evaluation["val"]["auc"] == pytest.approx(clean["val_auc"], abs=1e-4)
    assert evaluation["test"]["auc"] == pytest.approx(clean["test_auc"], abs=1e-4)
    assert (evaluation["n_train"], evaluation["n_val"], evaluation["n_test"]) == (210, 45, 45)
    assert geo["train_south_test_north"]["auc"] == pytest.approx(clean["geo_auc_s2n"], abs=1e-3)
    assert geo["train_north_test_south"]["auc"] == pytest.approx(clean["geo_auc_n2s"], abs=3e-3)
    for half in geo.values():
        assert half["train_n"] == half["test_n"] == 150


def test_repeated_splits_contain_the_reported_split():
    repeated = load("results/repeated_splits.json")
    clean = load("data/ba_clean_results.json")
    shipped = repeated["shipped_model_seed42"]
    assert shipped["test"]["rf_auc"] == pytest.approx(clean["test_auc"], abs=1e-4)
    assert shipped["validation"]["rf_auc"] == pytest.approx(clean["val_auc"], abs=1e-4)
    assert repeated["test_auc"]["min"] <= repeated["test_auc"]["mean"] <= repeated["test_auc"]["max"]
    for block in shipped.values():
        rows = list(block["alarm"].values()) + [block["previous_high_threshold_0.40"]]
        for row in rows:
            assert row["tp"] + row["fn"] == block["positives"]
            assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == block["n"]


def test_model_bundle_loads_and_scores(tile_pair):
    import evaluate_ba

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scikit-learn version differences when unpickling
        bundle = pickle.load(open(ROOT / "models/ba_rf_model.pkl", "rb"))
        assert {"model", "calibrated_model", "imputer"} <= set(bundle)
        assert bundle["model"].n_features_in_ == 44
        features = evaluate_ba.pair_features(*tile_pair).reshape(1, -1)
        probability = bundle["calibrated_model"].predict_proba(features)[0, 1]
        unchanged = evaluate_ba.pair_features(tile_pair[0], tile_pair[0]).reshape(1, -1)
        baseline = bundle["calibrated_model"].predict_proba(unchanged)[0, 1]
    assert 0.0 <= baseline < probability <= 1.0, "a 9 ha conversion must score above no change at all"
    assert np.isfinite(probability)
