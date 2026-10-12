"""Evaluation protocol: model selection, threshold choice, held-out test, robustness.

Sites are split once into a development set (255 sites) and a test set (45
sites; the same test sites as the original project, stratified split, seed 42).
Everything that is chosen - the model, its settings, the alarm thresholds - is
chosen on the development set by repeated cross-validation. The test set is
scored once, at the end.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from . import model as model_lib

SEED = 42
#: The "check this site" tier must catch at least this share of flagged sites (cross-validated).
MEDIUM_TARGET_RECALL = 0.90
#: The "likely encroachment" tier must be right at least this often (cross-validated).
HIGH_TARGET_PRECISION = 0.75


# ----------------------------------------------------------------------- splits
def dev_test_split(y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Indices of development (train + validation of the original 70/15/15 split) and test sites."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return np.sort(np.concatenate([train, val])), np.sort(test)


def spatial_folds(lat: np.ndarray, lon: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per site from k-means on coordinates: whole regions are held out together."""
    coords = np.column_stack([lat, lon * np.cos(np.radians(lat.mean()))])
    folds = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords)
    order = np.argsort([lat[folds == f].mean() for f in range(k)])  # number folds south to north
    return np.argsort(order)[folds]


# ---------------------------------------------------------------------- metrics
def counts(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    flagged = score >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(len(y) - tp - fp - fn),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
    }


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            values.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def spread(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std()), 4),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
        "n": int(len(v)),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.any():
            out.append(
                {"predicted": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4), "n": int(m.sum())}
            )
    return out


# ------------------------------------------------------------------- procedures
def cv_scores(X, y, kind: str, splitter, groups=None) -> tuple[list[float], list[float]]:
    aucs, aps = [], []
    for a, b in splitter.split(X, y, groups):
        p = model_lib.fit(X[a], y[a], kind).predict_proba(X[b])[:, 1]
        aucs.append(roc_auc_score(y[b], p))
        aps.append(average_precision_score(y[b], p))
    return aucs, aps


def out_of_fold(X, y, kind: str = "logreg", repeats: int = 5, seed: int = 0) -> np.ndarray:
    """Mean out-of-fold probability per site over repeated stratified 5-fold CV."""
    total, seen = np.zeros(len(y)), np.zeros(len(y))
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(X, y):
        total[b] += model_lib.fit(X[a], y[a], kind).predict_proba(X[b])[:, 1]
        seen[b] += 1
    return total / seen


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> dict:
    """Medium: highest threshold reaching the target recall. High: lowest reaching the target precision."""
    grid = np.unique(np.round(p, 4))
    medium = max(t for t in grid if counts(y, p, t)["recall"] >= MEDIUM_TARGET_RECALL)
    precise = [
        t
        for t in grid
        if t > medium and counts(y, p, t)["tp"] > 0 and counts(y, p, t)["precision"] >= HIGH_TARGET_PRECISION
    ]
    high = min(precise) if precise else float(grid[-1])
    return {"medium": round(float(medium), 4), "high": round(float(high), 4)}


def model_selection(X, y, repeats: int = 5) -> dict:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    out = {}
    for kind in model_lib.CANDIDATES:
        aucs, aps = cv_scores(X, y, kind, splitter)
        out[kind] = {"auc": spread(aucs), "average_precision": spread(aps)}
    return out


def robustness(X, y, lat, lon, kinds=("logreg", "random_forest")) -> dict:
    """Spread over repeated random CV and over spatially held-out regions, on all sites."""
    folds = spatial_folds(lat, lon)
    north = lat >= np.median(lat)
    out: dict = {
        "spatial_folds": {
            "k": int(folds.max() + 1),
            "sites_per_fold": np.bincount(folds).tolist(),
            "positives_per_fold": np.bincount(folds, weights=y).astype(int).tolist(),
        }
    }
    for kind in kinds:
        rnd, _ = cv_scores(X, y, kind, RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1))
        spatial = []
        for f in range(folds.max() + 1):
            test = folds == f
            p = model_lib.fit(X[~test], y[~test], kind).predict_proba(X[test])[:, 1]
            spatial.append(roc_auc_score(y[test], p))
        halves = {}
        for name, test in (("train_south_test_north", north), ("train_north_test_south", ~north)):
            p = model_lib.fit(X[~test], y[~test], kind).predict_proba(X[test])[:, 1]
            halves[name] = round(float(roc_auc_score(y[test], p)), 4)
        out[kind] = {
            "random_5fold_x10": spread(rnd),
            "spatial_5fold": [round(float(a), 4) for a in spatial],
            "spatial_5fold_mean": round(float(np.mean(spatial)), 4),
            **halves,
        }
    return out


def full_report(table) -> tuple[dict, object, dict]:
    """Run the whole protocol. Returns (report, final model, thresholds)."""
    from .data import xy

    X, y = xy(table)
    lat, lon = table["lat"].to_numpy(), table["lon"].to_numpy()
    ndvi_loss = table["mean_ndvi_loss"].to_numpy()
    dev, test = dev_test_split(y)

    selection = model_selection(X[dev], y[dev])
    oof = out_of_fold(X[dev], y[dev])
    # v1 blended the model score with mean NDVI loss; checked here, it adds nothing.
    fusion_oof = 0.65 * oof + 0.35 * ndvi_loss[dev]
    thresholds = choose_thresholds(y[dev], oof)

    final = model_lib.fit(X[dev], y[dev])
    p_test = final.predict_proba(X[test])[:, 1]
    rf = model_lib.fit(X[dev], y[dev], "random_forest")
    p_rf = rf.predict_proba(X[test])[:, 1]

    report = {
        "protocol": __doc__.strip(),
        "sites": {
            "development": int(len(dev)),
            "development_positives": int(y[dev].sum()),
            "test": int(len(test)),
            "test_positives": int(y[test].sum()),
        },
        "model_selection_cv_on_development": selection,
        "development_out_of_fold": {
            "auc": round(float(roc_auc_score(y[dev], oof)), 4),
            "auc_with_ndvi_fusion": round(float(roc_auc_score(y[dev], fusion_oof)), 4),
            "thresholds": thresholds,
            "medium": counts(y[dev], oof, thresholds["medium"]),
            "high": counts(y[dev], oof, thresholds["high"]),
        },
        "test": {
            "auc": round(float(roc_auc_score(y[test], p_test)), 4),
            "auc_ci95": list(bootstrap_auc(y[test], p_test)),
            "average_precision": round(float(average_precision_score(y[test], p_test)), 4),
            "brier": round(float(brier_score_loss(y[test], p_test)), 4),
            "medium": counts(y[test], p_test, thresholds["medium"]),
            "high": counts(y[test], p_test, thresholds["high"]),
            "random_forest_v1_auc": round(float(roc_auc_score(y[test], p_rf)), 4),
            "random_forest_v1_auc_ci95": list(bootstrap_auc(y[test], p_rf)),
            "calibration": calibration_bins(y[test], p_test, bins=3),
            "per_site": [
                {"site": s, "label": int(lbl), "probability": round(float(p), 4)}
                for s, lbl, p in zip(table["site"].to_numpy()[test], y[test], p_test, strict=True)
            ],
        },
        "robustness_all_sites": robustness(X, y, lat, lon),
        "leaky_rule_reference": {
            "note": "AUC of pct_converted, the quantity the labels are thresholded on. Not a usable feature.",
            "auc": round(float(roc_auc_score(y, table["pct_converted"])), 4),
        },
    }
    return report, final, thresholds
