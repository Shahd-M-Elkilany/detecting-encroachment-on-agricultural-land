"""
Temporal Manager -- Same-Season Year-Over-Year Strategy + Adaptive Baseline
===========================================================================

Problem with a fixed old baseline (e.g. always 2021):
  Crops grow and die, soil moisture changes, vegetation dries out seasonally.
  A 2021 vs 2026 comparison generates enormous noise from natural cycles --
  not real encroachment.

Problem with only comparing 20-days-prior:
  A building finished 40 days ago and unchanged since looks like "no change".

Solution -- two-step comparison:
  1. PRIMARY  -- same-season, 1 year prior (+-SEASON_WINDOW_DAYS)
       Eliminates seasonal noise. Only structural permanent changes survive.
       Example: March 2026 vs March 2025.

  2. RECENCY CHECK -- most recent image before the new one (<= revisit gap)
       Only runs when the primary flags a change.
       Answers "is this brand-new right now, or was it already there last month?"

Adaptive Baseline (AdaptiveBaseline class):
  When processing a sequence of images D1, D2, D3... for the same site:

  - PROMOTE threshold (default 0.60): if the model score exceeds this, encroachment
    is confirmed. The "after" image becomes the new baseline for future comparisons
    and t1_is_pos is set True (baseline is now an encroached site).

  - ALERT threshold (default 0.40): if score is between alert and promote thresholds,
    a yellow alert is raised but the baseline does NOT move.

  - No change (< alert threshold): baseline stays, t1_is_pos stays False.

State is persisted in outputs/temporal_state.json.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

from src.utils.logger import get_logger

logger = get_logger("temporal")

STATE_FILE = Path("outputs/temporal_state.json")

SEASON_WINDOW_DAYS = 45

# Adaptive baseline thresholds
PROMOTE_THRESHOLD = 0.60   # score >= this -> RED alert, baseline moves
ALERT_THRESHOLD   = 0.40   # score >= this -> YELLOW alert, baseline stays


# ---- State I/O ---------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        n = len(state.get("archive", []))
        logger.info(f"Loaded temporal state: {n} images in archive")
        return state
    return {"archive": [], "runs": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_state() -> None:
    if STATE_FILE.exists():
        backup = STATE_FILE.with_suffix(".json.bak")
        shutil.copy2(STATE_FILE, backup)
        logger.info(f"State backed up to {backup}")
    save_state({"archive": [], "runs": []})
    logger.info("Temporal state reset.")


# ---- Archive management ------------------------------------------------------

def register_image(state: Dict[str, Any], image_path: str, date_str: str) -> Dict[str, Any]:
    parsed = _parse_date(date_str)
    path   = str(Path(image_path).resolve())
    for entry in state["archive"]:
        if entry["path"] == path:
            logger.info(f"Image already registered: {date_str}")
            return state
    state["archive"].append({"date": parsed.isoformat(), "path": path, "label": date_str})
    state["archive"].sort(key=lambda e: e["date"])
    logger.info(f"Registered image: {date_str} -> {path}")
    return state


# ---- Image finders -----------------------------------------------------------

def find_same_season_image(
    archive:      List[Dict],
    new_date_str: str,
    exclude_path: Optional[str] = None,
    window_days:  int = SEASON_WINDOW_DAYS,
) -> Optional[Dict]:
    new_date    = _parse_date(new_date_str)
    target_date = new_date - timedelta(days=365)
    best        = None
    best_delta  = timedelta(days=window_days + 1)
    for entry in archive:
        if exclude_path and entry["path"] == exclude_path:
            continue
        entry_date = datetime.fromisoformat(entry["date"])
        if entry_date >= new_date:
            continue
        delta = abs(entry_date - target_date)
        if delta <= timedelta(days=window_days) and delta < best_delta:
            best       = entry
            best_delta = delta
    if best:
        logger.info(f"Same-season baseline: {best['label']} (delta={best_delta.days}d)")
    else:
        logger.warning(f"No same-season image found within +-{window_days} days of {target_date.date()}.")
    return best


def find_most_recent_image(
    archive:      List[Dict],
    new_date_str: str,
    exclude_path: Optional[str] = None,
) -> Optional[Dict]:
    new_date = _parse_date(new_date_str)
    past = [
        e for e in archive
        if datetime.fromisoformat(e["date"]) < new_date
        and (not exclude_path or e["path"] != exclude_path)
    ]
    if not past:
        return None
    recent = max(past, key=lambda e: e["date"])
    logger.info(f"Most recent prior image: {recent['label']}")
    return recent


# ---- Comparison plan ---------------------------------------------------------

def get_comparison_plan(
    new_image_path: str,
    new_date_str:   str,
    state:          Dict[str, Any],
) -> Dict[str, Any]:
    archive  = state.get("archive", [])
    new_path = str(Path(new_image_path).resolve())

    if len(archive) == 0:
        return {
            "mode": "no_archive", "primary": None, "recency": None,
            "explanation": "No images in archive yet.",
        }

    same_season = find_same_season_image(archive, new_date_str, exclude_path=new_path)
    most_recent = find_most_recent_image(archive, new_date_str, exclude_path=new_path)

    if same_season:
        primary = _make_comp(same_season, new_image_path, new_date_str, "same_season_primary")
        recency_candidate = find_most_recent_image(
            archive, new_date_str, exclude_path=same_season["path"]
        )
        recency = None
        if recency_candidate and recency_candidate["path"] != same_season["path"]:
            recency = _make_comp(recency_candidate, new_image_path, new_date_str, "recency_check")
        explanation = (
            f"Primary: {same_season['label']} -> {new_date_str} "
            f"({_days_between(same_season['date'], new_date_str)} days apart)\n"
        )
        explanation += (
            f"Recency: {recency_candidate['label']} -> {new_date_str}" if recency
            else "Recency: not available"
        )
        return {"mode": "same_season", "primary": primary, "recency": recency,
                "explanation": explanation}

    elif most_recent:
        primary = _make_comp(most_recent, new_image_path, new_date_str, "recent_fallback")
        return {
            "mode": "recent_only", "primary": primary, "recency": None,
            "explanation": (
                f"No same-season image found. Falling back to most recent: "
                f"{most_recent['label']} -> {new_date_str}. May include seasonal noise."
            ),
        }

    else:
        return {"mode": "no_archive", "primary": None, "recency": None,
                "explanation": "Archive has no images prior to the new date."}


# ---- Run recorder ------------------------------------------------------------

def record_run(
    state:           Dict[str, Any],
    new_image_path:  str,
    new_date_str:    str,
    primary_result:  Optional[Dict],
    recency_result:  Optional[Dict],
    change_detected: bool,
    encroachment_ha: float,
    regions:         List[Dict],
) -> Dict[str, Any]:
    state["runs"].append({
        "new_date":        new_date_str,
        "new_path":        new_image_path,
        "change_detected": change_detected,
        "encroachment_ha": encroachment_ha,
        "region_count":    len(regions),
        "had_recency":     recency_result is not None,
    })
    return state


# ---- Region merger -----------------------------------------------------------

def merge_results(
    primary_regions: List[Dict],
    recency_regions: Optional[List[Dict]],
    iou_threshold:   float = 0.3,
) -> List[Dict]:
    if not recency_regions:
        for r in primary_regions:
            r["encroachment_type"] = "unconfirmed_timing"
            r["timing_note"] = "Recency check not run"
        return primary_regions

    def iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    matched = set()
    merged  = []
    for p in primary_regions:
        p_bbox   = p.get("bbox_px", [0, 0, 0, 0])
        best_iou = 0.0; best_idx = -1
        for i, r in enumerate(recency_regions):
            v = iou(p_bbox, r.get("bbox_px", [0, 0, 0, 0]))
            if v > best_iou:
                best_iou = v; best_idx = i
        if best_iou >= iou_threshold and best_idx >= 0:
            p["encroachment_type"] = "new_encroachment"
            p["timing_note"]       = "Visible in both comparisons -- appeared within last revisit cycle"
            matched.add(best_idx)
        else:
            p["encroachment_type"] = "existing_encroachment"
            p["timing_note"]       = "In same-season only -- was already present before last revisit"
        merged.append(p)
    for i, r in enumerate(recency_regions):
        if i not in matched:
            r["encroachment_type"] = "new_encroachment"
            r["timing_note"]       = "Only in recency check -- very fresh encroachment"
            merged.append(r)
    n_new   = sum(1 for r in merged if r["encroachment_type"] == "new_encroachment")
    n_exist = sum(1 for r in merged if r["encroachment_type"] == "existing_encroachment")
    logger.info(f"Regions -- new: {n_new}, existing: {n_exist}")
    return merged


# ---- Adaptive Baseline -------------------------------------------------------

class AdaptiveBaseline:
    """
    Processes a time-ordered image sequence for a single site using an anchored
    baseline strategy instead of a rolling window.

    State machine transitions
    -------------------------
    score >= promote_threshold  ->  RED alert  : baseline moves to current image,
                                                 t1_is_pos becomes True.
    score >= alert_threshold    ->  YELLOW alert: baseline stays (uncertain signal).
    score <  alert_threshold    ->  no change   : baseline stays.

    Why this beats rolling windows
    --------------------------------
    Rolling (D1->D2, D2->D3, ...): noise compounds. A barely-threshold fluctuation
    in D1->D2 pollutes the D2 baseline; subsequent comparisons lose signal.

    Adaptive: always compares against the last *confirmed-clean* state, maximising
    the delta when real encroachment appears. The t1_is_pos flag transfers naturally
    after promotion, so the classifier knows the baseline image contains a building.

    Example
    -------
    D1->D2: score=0.25  -> none    (baseline stays D1)
    D1->D3: score=0.50  -> yellow  (baseline stays D1)
    D1->D4: score=0.72  -> RED     (baseline moves to D4, t1_is_pos=True)
    D4->D5: score=0.30  -> none    (t1_is_pos=True, baseline stays D4)
    """

    def __init__(
        self,
        model,
        calibrator,
        feature_names: list,
        alert_threshold:   float = ALERT_THRESHOLD,
        promote_threshold: float = PROMOTE_THRESHOLD,
    ) -> None:
        self.model             = model
        self.calibrator        = calibrator
        self.feature_names     = feature_names
        self.alert_threshold   = alert_threshold
        self.promote_threshold = promote_threshold

        self.baseline_path:   Optional[Path] = None
        self.baseline_date:   Optional[str]  = None
        self.baseline_is_pos: bool           = False
        self.history:         list           = []

    def step(self, image_path, date_str: str) -> dict:
        """
        Process one new image. Feed images in chronological order.

        First call registers the image as the initial baseline (no score).
        Subsequent calls score and apply state-machine transitions.

        Result keys: baseline_path, baseline_date, current_path, current_date,
                     t1_is_pos, score, alert, baseline_moved, note
        """
        image_path = Path(image_path)

        if self.baseline_path is None:
            self.baseline_path   = image_path
            self.baseline_date   = date_str
            self.baseline_is_pos = False
            result = {
                "baseline_path":  str(image_path),
                "baseline_date":  date_str,
                "current_path":   str(image_path),
                "current_date":   date_str,
                "t1_is_pos":      False,
                "score":          None,
                "alert":          "none",
                "baseline_moved": True,
                "note":           "Initial baseline registered -- no comparison run.",
            }
            self.history.append(result)
            logger.info(f"AdaptiveBaseline -- initial baseline: {date_str}")
            return result

        old_path = self.baseline_path
        old_date = self.baseline_date
        t1_is_pos_now = self.baseline_is_pos

        score = self._score(self.baseline_path, image_path, t1_is_pos_now)

        if score >= self.promote_threshold:
            self.baseline_path   = image_path
            self.baseline_date   = date_str
            self.baseline_is_pos = True
            alert          = "red"
            baseline_moved = True
            note = (
                f"RED alert (score={score:.3f} >= {self.promote_threshold}). "
                f"Baseline promoted from {old_date} to {date_str}. t1_is_pos=True."
            )
            logger.warning(f"AdaptiveBaseline -- RED at {date_str}: score={score:.3f}")

        elif score >= self.alert_threshold:
            alert          = "yellow"
            baseline_moved = False
            note = (
                f"YELLOW alert (score={score:.3f}). "
                f"Baseline held at {old_date} -- awaiting stronger signal."
            )
            logger.info(f"AdaptiveBaseline -- YELLOW at {date_str}: score={score:.3f}")

        else:
            alert          = "none"
            baseline_moved = False
            note = (
                f"No change (score={score:.3f} < {self.alert_threshold}). "
                f"Baseline stays at {old_date}."
            )
            logger.info(f"AdaptiveBaseline -- clear at {date_str}: score={score:.3f}")

        result = {
            "baseline_path":  str(old_path),   # what was compared against (pre-promotion)
            "baseline_date":  old_date,
            "current_path":   str(image_path),
            "current_date":   date_str,
            "t1_is_pos":      t1_is_pos_now,
            "score":          round(float(score), 4),
            "alert":          alert,
            "baseline_moved": baseline_moved,
            "note":           note,
        }
        self.history.append(result)
        return result

    def reset(self, image_path, date_str: str) -> None:
        """Manually reset baseline (e.g. after confirmed remediation)."""
        self.baseline_path   = Path(image_path)
        self.baseline_date   = date_str
        self.baseline_is_pos = False
        logger.info(f"AdaptiveBaseline -- manual reset to {date_str}")

    def summary(self) -> dict:
        """Compact summary of all steps processed so far."""
        return {
            "steps":                 len(self.history),
            "red_alerts":            sum(1 for r in self.history if r["alert"] == "red"),
            "yellow_alerts":         sum(1 for r in self.history if r["alert"] == "yellow"),
            "current_baseline_date": self.baseline_date,
            "baseline_is_pos":       self.baseline_is_pos,
            "history":               self.history,
        }

    def _score(self, t1_path: Path, t2_path: Path, t1_is_pos: bool) -> float:
        """Extract features and return calibrated probability score."""
        import sys as _sys
        project_root = Path(__file__).resolve().parent.parent.parent
        _sys.path.insert(0, str(project_root))
        from train_classifier import extract_features  # noqa: PLC0415
        feats    = extract_features(t1_path, t2_path, t1_is_pos=t1_is_pos)
        raw_prob = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
        if self.calibrator is not None:
            return float(self.calibrator.predict_proba([[raw_prob]])[0, 1])
        return raw_prob


# ---- Helpers -----------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """Parse flexible date strings: '2025', '2025-03', '2025-03-15'."""
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{date_str}'. Use YYYY, YYYY-MM, or YYYY-MM-DD.")


def _make_comp(t1_entry: Dict, t2_path: str, t2_date: str, label: str) -> Dict:
    return {
        "t1_path":  t1_entry["path"],
        "t2_path":  str(Path(t2_path).resolve()),
        "t1_date":  t1_entry["label"],
        "t2_date":  t2_date,
        "label":    label,
    }


def _days_between(iso_date: str, date_str: str) -> int:
    d1 = datetime.fromisoformat(iso_date)
    d2 = _parse_date(date_str)
    return abs((d2 - d1).days)
