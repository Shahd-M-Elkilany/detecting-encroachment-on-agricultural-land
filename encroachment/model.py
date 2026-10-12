"""The classifier: standardised features + L2 logistic regression.

Chosen over the earlier Random Forest by repeated cross-validation on the
development sites (results/model_selection.json): CV AUC 0.87 vs 0.79. The
labels come from a threshold rule on spectral change, which a linear model
over spectral-change statistics fits well.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, N_FEATURES

#: Inverse regularisation strength, chosen by cross-validation (flat between 1 and 3).
C = 1.0


def make_model(kind: str = "logreg", seed: int = 0):
    """A fresh, unfitted model. ``kind`` names one of the candidates compared in model selection."""
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    if kind == "random_forest":  # the v1 production model
        return RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    if kind == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed
        )
    raise ValueError(f"unknown model kind: {kind}")


CANDIDATES = ("logreg", "random_forest", "gradient_boosting")


def fit(X: np.ndarray, y: np.ndarray, kind: str = "logreg", seed: int = 0):
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")
    return make_model(kind, seed).fit(X, y)


def save(model, path: Path, metadata: dict) -> None:
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {"features": list(FEATURE_NAMES), "scikit_learn": sklearn.__version__, **metadata}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load(path: Path):
    import joblib

    model = joblib.load(path)
    if getattr(model, "n_features_in_", N_FEATURES) != N_FEATURES:
        raise ValueError(f"{path} expects {model.n_features_in_} features, not {N_FEATURES}")
    return model


def probability(model, features: np.ndarray) -> np.ndarray:
    """Probability of encroachment for one (44,) vector or an (n, 44) matrix."""
    X = np.nan_to_num(np.atleast_2d(np.asarray(features, dtype=np.float64)), nan=0.0)
    return model.predict_proba(X)[:, 1]
