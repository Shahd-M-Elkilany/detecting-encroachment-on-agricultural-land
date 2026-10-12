"""Score one site: features -> probability -> risk tier, plus change clusters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import alarm
from . import model as model_lib
from .clusters import Cluster, find_clusters
from .data import MODEL
from .features import TilePair, pair_features, read_pair


@dataclass
class SiteResult:
    name: str
    pair: TilePair
    probability: float
    risk: str
    ndvi_loss: float
    clusters: list[Cluster] = field(default_factory=list)

    @property
    def changed_ha(self) -> float:
        return round(sum(c.area_ha for c in self.clusters), 2)

    def summary(self) -> dict:
        lat, lon = self.pair.centre
        return {
            "site": self.name,
            "probability": round(self.probability, 4),
            "risk_level": self.risk,
            "mean_ndvi_loss": round(self.ndvi_loss, 4),
            "clusters": len(self.clusters),
            "changed_ha": self.changed_ha,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        }


class Scorer:
    """Loads the model once and scores any number of tile pairs."""

    def __init__(self, model_path: Path = MODEL):
        self.model = model_lib.load(model_path)

    def score(self, pair: TilePair, name: str = "site") -> SiteResult:
        probability = float(model_lib.probability(self.model, pair_features(pair.before, pair.after))[0])
        return SiteResult(
            name=name,
            pair=pair,
            probability=probability,
            risk=alarm.risk_level(probability),
            ndvi_loss=alarm.mean_ndvi_loss(pair.before, pair.after),
            clusters=find_clusters(pair.before, pair.after, pair.pixel_area_ha),
        )

    def score_files(self, before: Path, after: Path, name: str | None = None) -> SiteResult:
        return self.score(read_pair(before, after), name or Path(before).stem.split("_before")[0])
