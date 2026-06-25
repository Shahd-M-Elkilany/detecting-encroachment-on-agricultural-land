#!/usr/bin/env python3
"""
predict.py — Inference script for the KEMET1 encroachment classifier.

Usage (single pair):
    python predict.py --t1 path/to/T1.tif --t2 path/to/T2.tif

Usage (multiple pairs for the same tile, enables temporal consistency):
    python predict.py \\
        --t1 tile_01_T1.tif --t2 tile_01_T2.tif \\
        --t1 tile_01_T2.tif --t2 tile_01_T3.tif \\
        --t1 tile_01_T3.tif --t2 tile_01_T4.tif

Options:
    --model PATH     path to .pkl bundle (default: weights/encroachment_classifier_rf.pkl)
    --t1 PATH        path to "before" image (repeatable)
    --t2 PATH        path to "after"  image (repeatable, must match --t1 count)
    --t1-is-pos      treat T1 as already-encroached (raises t1_is_pos feature flag)
                     if omitted, inferred from filename suffix (*_pos.tif)
    --no-consistency skip majority temporal consistency dampening
    --json           output machine-readable JSON instead of human-readable text

Temporal consistency:
    When ≥ MAJORITY_THRESH pairs (default 2) across all supplied image pairs
    score positive, all scores for that tile are multiplied by SEASONAL_DAMPEN (0.6).
    This suppresses seasonally-drifting false positives.

Exit codes:
    0  no encroachment detected
    1  encroachment detected in at least one pair
    2  error
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Constants (must match evaluate.py) ───────────────────────────────────────
SEASONAL_DAMPEN = 0.6
MAJORITY_THRESH = 2


def _infer_t1_is_pos(path: Path) -> bool:
    """Guess t1_is_pos from filename suffix (*_pos.tif → True)."""
    return path.stem.endswith("pos")


def load_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_pairs(
    pairs: list[tuple[Path, Path, bool]],
    model,
    calibrator,
    feature_names: list[str],
) -> list[float]:
    """Return raw probability scores for each pair."""
    # Import extract_features from train_classifier (same project)
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    from train_classifier import extract_features  # noqa: PLC0415

    scores = []
    for t1_path, t2_path, t1_is_pos in pairs:
        feats = extract_features(t1_path, t2_path, t1_is_pos=t1_is_pos)
        if len(feats) != len(feature_names):
            print(
                f"[WARN] Feature count mismatch: got {len(feats)}, "
                f"expected {len(feature_names)}",
                file=sys.stderr,
            )
        raw_prob = float(model.predict_proba(feats.reshape(1, -1))[0, 1])
        if calibrator is not None:
            prob = float(calibrator.predict_proba([[raw_prob]])[0, 1])
        else:
            prob = raw_prob
        scores.append(prob)
    return scores


def apply_temporal_consistency(scores: list[float], threshold: float) -> list[float]:
    """Dampen all scores if ≥ MAJORITY_THRESH pairs exceed the threshold."""
    n_positive = sum(s >= threshold for s in scores)
    if n_positive >= MAJORITY_THRESH:
        scores = [s * SEASONAL_DAMPEN for s in scores]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the KEMET1 encroachment classifier on one or more image pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "encroachment_classifier_rf.pkl",
        help="Path to the saved model bundle (.pkl)",
    )
    parser.add_argument(
        "--t1",
        dest="t1_paths",
        metavar="PATH",
        action="append",
        required=True,
        type=Path,
        help='Path to the "before" image. Repeat for multiple pairs.',
    )
    parser.add_argument(
        "--t2",
        dest="t2_paths",
        metavar="PATH",
        action="append",
        required=True,
        type=Path,
        help='Path to the "after" image. Repeat for multiple pairs.',
    )
    parser.add_argument(
        "--t1-is-pos",
        dest="t1_is_pos_flag",
        action="store_true",
        default=None,
        help="Force t1_is_pos=True for all pairs. "
             "Default: inferred from filename (*_pos.tif).",
    )
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="Disable temporal consistency dampening.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    # Validate pair counts
    if len(args.t1_paths) != len(args.t2_paths):
        print(
            f"[ERROR] --t1 and --t2 must be given the same number of times "
            f"(got {len(args.t1_paths)} vs {len(args.t2_paths)}).",
            file=sys.stderr,
        )
        return 2

    # Load model bundle
    bundle = load_bundle(args.model)
    model         = bundle["model"]
    calibrator    = bundle.get("calibrator", None)
    threshold     = bundle["threshold"]
    feature_names = bundle.get("feature_names", [])
    model_name    = bundle.get("model_name", "RF")
    val_auc       = bundle.get("val_auc", float("nan"))
    test_auc      = bundle.get("test_auc", float("nan"))

    # Build pair list with t1_is_pos flags
    pairs: list[tuple[Path, Path, bool]] = []
    for t1p, t2p in zip(args.t1_paths, args.t2_paths):
        if args.t1_is_pos_flag is not None:
            tip = args.t1_is_pos_flag
        else:
            tip = _infer_t1_is_pos(t1p)
        pairs.append((t1p, t2p, tip))

    # Score
    scores = score_pairs(pairs, model, calibrator, feature_names)

    # Temporal consistency
    if not args.no_consistency and len(scores) > 1:
        scores = apply_temporal_consistency(scores, threshold)

    # Decisions
    decisions = [s >= threshold for s in scores]
    encroachment_detected = any(decisions)

    if args.json:
        result = {
            "model": model_name,
            "threshold": float(threshold),
            "val_auc": float(val_auc) if not (val_auc != val_auc) else None,
            "test_auc": float(test_auc) if not (test_auc != test_auc) else None,
            "temporal_consistency_applied": bool(not args.no_consistency and len(scores) > 1),
            "encroachment_detected": bool(encroachment_detected),
            "pairs": [
                {
                    "t1": str(t1p),
                    "t2": str(t2p),
                    "t1_is_pos": bool(tip),
                    "score": round(float(score), 4),
                    "decision": bool(decision),
                    "label": "ENCROACHMENT" if decision else "no encroachment",
                }
                for (t1p, t2p, tip), score, decision in zip(pairs, scores, decisions)
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        bar_width = 20

        print()
        print("══════════════════════════════════════════════════════")
        print("  KEMET1 — Encroachment Prediction")
        print("══════════════════════════════════════════════════════")
        print(f"  Model     : {model_name}")
        print(f"  Threshold : {threshold:.2f}")
        if not (val_auc != val_auc):
            print(f"  Val AUC   : {val_auc:.4f}")
        if not (test_auc != test_auc):
            print(f"  Test AUC  : {test_auc:.4f}")
        tc_applied = not args.no_consistency and len(scores) > 1
        print(f"  Temporal consistency : {'ON' if tc_applied else 'OFF'}")
        print()

        for (t1p, t2p, tip), score, decision in zip(pairs, scores, decisions):
            bar = "█" * int(score * bar_width)
            bar = bar.ljust(bar_width, "░")
            marker = "✗" if decision else "✓"
            label  = "← ENCROACHMENT" if decision else ""
            t1_tag = "(pos)" if tip else "(neg)"
            print(
                f"  {marker}  {t1p.name} {t1_tag} → {t2p.name}"
                f"  p={score:.3f} {bar}  {label}"
            )

        print()
        print("──────────────────────────────────────────────────────")
        if encroachment_detected:
            print("  RESULT: ⚠  ENCROACHMENT DETECTED")
        else:
            print("  RESULT: ✓  No encroachment detected")
        print("══════════════════════════════════════════════════════")
        print()

    return 1 if encroachment_detected else 0


if __name__ == "__main__":
    sys.exit(main())
