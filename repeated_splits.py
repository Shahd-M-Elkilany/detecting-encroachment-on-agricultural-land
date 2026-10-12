"""repeated_splits.py - How much do the headline metrics depend on the split?

The reported validation / test AUC come from one stratified 70/15/15 split
(seed 42) with 45 test sites. This script repeats the same protocol, with the
production model's hyper-parameters, over many seeds and reports the spread.
It also scores the deployed alarm (alarm.py) on the seed-42 validation and
test splits, at the thresholds the scripts use.

Run from the repository root (needs the tiles, see README):
    python repeated_splits.py

Output:
  results/repeated_splits.json
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

import alarm

ROOT = Path(__file__).parent
BA_DIR = ROOT / "data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles"
LABELS = ROOT / "data/ba_labels.json"
MODEL_PATH = ROOT / "models/ba_rf_model.pkl"
OUT_JSON = ROOT / "results/repeated_splits.json"
N_SEEDS = 30


def load_features(records):
    """44 features, the spectral term of the fusion score and the label, per site."""
    import rasterio

    from evaluate_ba import pair_features

    X, spec, y = [], [], []
    for r in records:
        with rasterio.open(BA_DIR / (r["site"] + "_before_2024.tif")) as s:
            d1 = s.read().astype(np.float32)
        with rasterio.open(BA_DIR / (r["site"] + "_after_2025.tif")) as s:
            d2 = s.read().astype(np.float32)
        X.append(pair_features(d1, d2))
        spec.append(alarm.spectral_score(d1, d2))
        y.append(1 if r["label"] == "pos" else 0)
    return np.array(X), np.array(spec), np.array(y)


def split_70_15_15(n, y, seed):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, val, test


def counts(y_true, score, threshold):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, (score >= threshold).astype(int), labels=[0, 1])
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
    }


def summarise(X, spec, y, bundle, n_seeds=N_SEEDS):
    stored = bundle["model"]
    params = dict(n_estimators=stored.n_estimators, max_depth=stored.max_depth,
                  min_samples_leaf=stored.min_samples_leaf, class_weight=stored.class_weight)
    val_auc, test_auc = [], []
    for seed in range(n_seeds):
        tr, val, test = split_70_15_15(len(y), y, seed)
        imp = SimpleImputer(strategy="median").fit(X[tr])
        rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params).fit(imp.transform(X[tr]), y[tr])
        val_auc.append(roc_auc_score(y[val], rf.predict_proba(imp.transform(X[val]))[:, 1]))
        test_auc.append(roc_auc_score(y[test], rf.predict_proba(imp.transform(X[test]))[:, 1]))

    def spread(values):
        v = np.array(values)
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                "p05": round(float(np.percentile(v, 5)), 4), "p95": round(float(np.percentile(v, 95)), 4)}

    # The shipped model on the split it was trained with (seed 42).
    tr, val, test = split_70_15_15(len(y), y, 42)
    clf = bundle.get("calibrated_model") or bundle["model"]
    prob = clf.predict_proba(X)[:, 1]
    fusion = alarm.RF_WEIGHT * prob + alarm.SPECTRAL_WEIGHT * spec
    tiers = {"medium": alarm.YELLOW_THRESHOLD, "high": alarm.ALERT_THRESHOLD}

    def block(idx):
        return {
            "n": int(len(idx)), "positives": int(y[idx].sum()),
            "rf_auc": round(float(roc_auc_score(y[idx], prob[idx])), 4),
            "fusion_auc": round(float(roc_auc_score(y[idx], fusion[idx])), 4),
            "alarm": {name: counts(y[idx], fusion[idx], t) for name, t in tiers.items()},
            "previous_high_threshold_0.40": counts(y[idx], fusion[idx], 0.40),
        }

    return {
        "protocol": "stratified 70/15/15, production hyper-parameters, one fit per seed",
        "n_seeds": n_seeds,
        "hyper_parameters": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()},
        "val_auc": spread(val_auc),
        "test_auc": spread(test_auc),
        "shipped_model_seed42": {"validation": block(val), "test": block(test)},
    }


def main():
    records = json.load(open(LABELS))
    print(f"Extracting features for {len(records)} sites ...")
    X, spec, y = load_features(records)
    bundle = pickle.load(open(MODEL_PATH, "rb"))
    result = summarise(X, spec, y, bundle)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("val_auc", "test_auc")}, indent=2))
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
