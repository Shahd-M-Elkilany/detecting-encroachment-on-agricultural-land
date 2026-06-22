"""
Temporal Manager
================
Tracks multi-date pipeline runs and decides which comparisons to make
when a new image arrives.

State is persisted in outputs/temporal_state.json so the pipeline
remembers what happened in previous runs across sessions.

Decision logic
--------------
When a new image for date D arrives:

  Case A — no change was ever detected in history
    → compare previous image vs D  (rolling window)

  Case B — change WAS detected in a previous period
    → compare baseline (earliest clean image) vs D  → total cumulative change
    → compare previous image vs D                   → new encroachment this period
    → merge and tag each region as "existing" or "new"

The result is two layers of encroachment, letting you see:
  • How much total farmland has been lost since the original state
  • What is brand-new encroachment vs what was already flagged before
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logger import get_logger

logger = get_logger("temporal")

STATE_FILE = Path("outputs/temporal_state.json")


# ── State I/O ────────────────────────────────────────────────────────────────

def load_state() -> Dict[str, Any]:
    """Load persisted temporal state, or return empty state if first run."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        logger.info(f"Loaded temporal state: {len(state.get('history', []))} previous runs")
        return state
    return {"baseline": None, "history": [], "last_image": None}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Temporal state saved → {STATE_FILE}")


def reset_state() -> None:
    """Clear all history and start fresh (use when re-baselining)."""
    if STATE_FILE.exists():
        backup = STATE_FILE.with_suffix(".json.bak")
        shutil.copy2(STATE_FILE, backup)
        logger.info(f"State backed up to {backup}")
    save_state({"baseline": None, "history": [], "last_image": None})
    logger.info("Temporal state reset.")


# ── Decision engine ──────────────────────────────────────────────────────────

def get_comparison_plan(
    new_image_path: str,
    new_date:       str,
    state:          Dict[str, Any],
) -> Dict[str, Any]:
    """
    Given the current state and a new image, return what comparisons to run.

    Returns
    -------
    {
      "mode": "rolling" | "dual",
      "comparisons": [
        {"t1_path": ..., "t2_path": ..., "t1_date": ..., "t2_date": ..., "label": "cumulative"|"incremental"},
        ...
      ],
      "is_first_run": bool,
    }
    """
    history  = state.get("history", [])
    baseline = state.get("baseline")
    last     = state.get("last_image")

    # ── First run ever ───────────────────────────────────────────────────────
    if not history or last is None:
        logger.info("First run — no history. Need a T1 (before) image to compare against.")
        return {
            "mode": "first_run",
            "comparisons": [],
            "is_first_run": True,
        }

    # ── Was change ever detected? ────────────────────────────────────────────
    change_ever_detected = any(r.get("change_detected", False) for r in history)

    if not change_ever_detected:
        # Case A: no change detected yet — rolling window only
        logger.info(
            f"No change detected in any previous run. "
            f"Comparing {last['date']} → {new_date} (rolling window)."
        )
        return {
            "mode": "rolling",
            "comparisons": [
                {
                    "t1_path":  last["path"],
                    "t2_path":  new_image_path,
                    "t1_date":  last["date"],
                    "t2_date":  new_date,
                    "label":    "incremental",
                }
            ],
            "is_first_run": False,
        }

    else:
        # Case B: change was detected before → dual comparison
        logger.info(
            f"Previous change detected. Running DUAL comparison:\n"
            f"  1. Baseline {baseline['date']} → {new_date}  (total cumulative)\n"
            f"  2. Previous {last['date']} → {new_date}      (new this period)"
        )
        return {
            "mode": "dual",
            "comparisons": [
                {
                    "t1_path":  baseline["path"],
                    "t2_path":  new_image_path,
                    "t1_date":  baseline["date"],
                    "t2_date":  new_date,
                    "label":    "cumulative",
                },
                {
                    "t1_path":  last["path"],
                    "t2_path":  new_image_path,
                    "t1_date":  last["date"],
                    "t2_date":  new_date,
                    "label":    "incremental",
                },
            ],
            "is_first_run": False,
        }


# ── State updater ─────────────────────────────────────────────────────────────

def record_run(
    state:            Dict[str, Any],
    t1_path:          str,
    t2_path:          str,
    t1_date:          str,
    t2_date:          str,
    change_detected:  bool,
    encroachment_ha:  float,
    regions:          List[Dict],
) -> Dict[str, Any]:
    """
    Record the result of a pipeline run and update the state.

    - Sets baseline on the very first run (T1 of first comparison).
    - Updates last_image to the new image (T2).
    - Appends to history.
    """
    # Set baseline once — the T1 of the very first run
    if state["baseline"] is None:
        state["baseline"] = {"date": t1_date, "path": t1_path}
        logger.info(f"Baseline set: {t1_date} → {t1_path}")

    state["history"].append({
        "t1_date":         t1_date,
        "t1_path":         t1_path,
        "t2_date":         t2_date,
        "t2_path":         t2_path,
        "change_detected": change_detected,
        "encroachment_ha": encroachment_ha,
        "region_count":    len(regions),
    })

    # Advance the "last image" pointer to T2
    state["last_image"] = {"date": t2_date, "path": t2_path}

    return state


# ── Region merger (dual mode) ─────────────────────────────────────────────────

def merge_dual_results(
    cumulative_regions:   List[Dict],
    incremental_regions:  List[Dict],
    iou_threshold:        float = 0.3,
) -> List[Dict]:
    """
    Merge cumulative and incremental region lists into a single annotated list.

    A region in cumulative that also appears in incremental → "new_encroachment"
    A region in cumulative only (not in incremental) → "existing_encroachment"
    A region in incremental only → "new_encroachment" (edge case)

    Matching is done by IoU of bounding boxes.
    """
    merged: List[Dict] = []

    def bbox_iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (area1 + area2 - inter + 1e-8)

    matched_incremental = set()

    for c_region in cumulative_regions:
        best_iou  = 0.0
        best_idx  = -1
        c_bbox    = c_region.get("bbox_px", [0, 0, 0, 0])

        for i, i_region in enumerate(incremental_regions):
            iou = bbox_iou(c_bbox, i_region.get("bbox_px", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            # Region exists in BOTH → new encroachment this period
            c_region["encroachment_type"] = "new_encroachment"
            c_region["incremental_score"] = incremental_regions[best_idx].get("red_score", 0)
            matched_incremental.add(best_idx)
        else:
            # Only in cumulative → pre-existing encroachment
            c_region["encroachment_type"] = "existing_encroachment"
            c_region["incremental_score"] = 0.0

        merged.append(c_region)

    # Add any incremental-only regions (shouldn't happen often, but handle it)
    for i, i_region in enumerate(incremental_regions):
        if i not in matched_incremental:
            i_region["encroachment_type"] = "new_encroachment"
            i_region["incremental_score"] = i_region.get("red_score", 0)
            merged.append(i_region)

    n_new      = sum(1 for r in merged if r["encroachment_type"] == "new_encroachment")
    n_existing = sum(1 for r in merged if r["encroachment_type"] == "existing_encroachment")
    logger.info(
        f"Merged regions: {n_new} new encroachment, {n_existing} existing encroachment"
    )
    return merged
