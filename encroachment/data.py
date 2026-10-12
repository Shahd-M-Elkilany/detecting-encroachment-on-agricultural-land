"""Paths and the shipped data tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TILES_DIR = DATA / "tiles"  # site{N}_before_2024.tif / site{N}_after_2025.tif (not committed)
LABELS = DATA / "labels.json"
FEATURES = DATA / "features.csv"  # per-site features extracted from the tiles (committed)
MODEL = ROOT / "models" / "encroachment_logreg.joblib"
RESULTS = ROOT / "results"


def tile_paths(site: str, tiles_dir: Path = TILES_DIR) -> tuple[Path, Path]:
    return tiles_dir / f"{site}_before_2024.tif", tiles_dir / f"{site}_after_2025.tif"


def load_labels(path: Path = LABELS) -> list[dict]:
    return json.loads(Path(path).read_text())


def load_features(path: Path = FEATURES) -> pd.DataFrame:
    """One row per site: identifiers, coordinates, label, labelling inputs and the 44 features."""
    table = pd.read_csv(path)
    missing = [c for c in FEATURE_NAMES if c not in table]
    if missing:
        raise ValueError(f"{path} lacks feature columns, e.g. {missing[:3]}")
    return table


def xy(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return table[list(FEATURE_NAMES)].to_numpy(dtype=np.float64), table["label"].to_numpy(dtype=int)
