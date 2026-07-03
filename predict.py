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

Usage (adaptive baseline — chronological sequence for ONE site):
    python predict.py --adaptive \\
        --images D1.tif D2.tif D3.tif D4.tif \\
        --dates  2024-01 2024-07 2025-01 2025-07

    The adaptive baseline strategy:
    - Starts with D1 as the anchor baseline.
    - Compares baseline vs each new image in order.
    - RED alert (score >= promote-threshold, default 0.60):
        baseline moves to the current image; t1_is_pos becomes True.
    - YELLOW alert (score >= alert-threshold, default 0.40):
        alert raised but baseline stays — anchor held at last clean image.
    - No change (score < alert-threshold):
        baseline stays; next image still compared against the same anchor.

Options:
    --model PATH              path to .pkl bundle
    --t1 PATH                 "before" image (repeatable, standard mode)
    --t2 PATH                 "after" image (repeatable, standard mode)
    --t1-is-pos               force t1_is_pos=True (standard mode)
    --no-consistency          skip temporal consistency dampening (standard mode)
    --adaptive                switch to adaptive baseline mode
    --images PATH [PATH …]    chronological image list (adaptive mode)
    --dates STR [STR …]       matching date labels, e.g. 2024-01 (adaptive mode)
    --alert-threshold FLOAT   yellow-alert threshold (default: 0.40, adaptive mode)
    --promote-threshold FLOAT baseline-promotion threshold (default: 0.60, adaptive mode)
    --json                    output machine-readable JSON

Temporal consistency (standard mode):
    When ≥ MAJORITY_THRESH pairs (default 2) across all supplied image pairs
    score positive, all scores for that tile are multiplied by SEASONAL_DAMPEN (0.6).
    This suppresses seasonally-drifting false positives.

Exit codes:
    0  no encroachment detected
    1  encroachment detected (red alert in adaptive mode, or any pair in standard mode)
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
SEASONAL_DAMPEN  = 0.6
MAJORITY_THRESH  = 2

# Adaptive baseline thresholds (mirrored from temporal_manager for CLI defaults)
ALERT_THRESHOLD   = 0.40
PROMOTE_THRESHOLD = 0.60


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


def apply_temporal_consistency(
    scores: list[float],
    threshold: float,
    t1_is_pos_flags: list[bool],
) -> list[float]:
    """
    Suppress seasonal-drift false positives while preserving genuine signals.

    Case 1 — drift in established-encroachment windows:
        If ≥ MAJORITY_THRESH pos→pos pairs (t1_is_pos=True) score above the
        threshold, the 'already encroached' windows show anomalous spectral
        change (seasonal drift rather than new encroachment). Dampen only
        those pos→pos pairs; leave the primary neg→pos signal pair intact.

    Case 2 — all-negative tile with global drift:
        If there are no pos→pos pairs (all transitions are neg→neg) and
        every pair fires above the threshold, the whole tile is drifting.
        Dampen all pairs.
    """
    if len(scores) <= 1:
        return scores

    n_pos_pairs  = sum(t1_is_pos_flags)
    n_neg_pairs  = len(scores) - n_pos_pairs
    n_total_fire = sum(s >= threshold for s in scores)
    n_pos_fire   = sum(
        s >= threshold for s, is_pos in zip(scores, t1_is_pos_flags) if is_pos
    )
    n_neg_fire   = sum(
        s >= threshold for s, is_pos in zip(scores, t1_is_pos_flags) if not is_pos
    )

    # Case 1 — encroachment tile with drifting pos→pos windows:
    #   ≥ 1 pos→pos pair fires AND total fires across all pairs ≥ MAJORITY_THRESH.
    #   Count neg→pos fires in the majority to catch tiles where the signal pair
    #   fires alongside just one anomalous pos→pos pair (e.g. only 2 total fires).
    #   Dampen only the pos→pos pairs; leave the neg→pos signal intact.
    if n_pos_pairs > 0 and n_pos_fire >= 1 and n_total_fire >= MAJORITY_THRESH:
        return [
            s * SEASONAL_DAMPEN if is_pos else s
            for s, is_pos in zip(scores, t1_is_pos_flags)
        ]

    # Case 2 — all-negative tile with majority drift:
    #   No pos→pos pairs exist and ≥ MAJORITY_THRESH neg→neg pairs fire.
    #   Dampen all (there is no protected signal pair here).
    if n_pos_pairs == 0 and n_neg_pairs > 1 and n_neg_fire >= MAJORITY_THRESH:
        return [s * SEASONAL_DAMPEN for s in scores]

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
    # ── Standard mode args ────────────────────────────────────────────────────
    parser.add_argument(
        "--t1",
        dest="t1_paths",
        metavar="PATH",
        action="append",
        default=[],
        type=Path,
        help='Path to the "before" image. Repeat for multiple pairs.',
    )
    parser.add_argument(
        "--t2",
        dest="t2_paths",
        metavar="PATH",
        action="append",
        default=[],
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
    # ── Adaptive baseline mode args ───────────────────────────────────────────
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive baseline mode for a chronological sequence of images.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="PATH",
        type=Path,
        default=[],
        help="Chronological image paths (adaptive mode).",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        metavar="DATE",
        default=[],
        help="Date labels matching --images, e.g. 2024-01 2024-07 (adaptive mode).",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help=f"Yellow-alert threshold (adaptive mode, default: {ALERT_THRESHOLD}).",
    )
    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=None,
        help=f"Baseline-promotion threshold (adaptive mode, default: {PROMOTE_THRESHOLD}).",
    )
    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    # Load model bundle
    bundle = load_bundle(args.model)
    model         = bundle["model"]
    calibrator    = bundle.get("calibrator", None)
    threshold     = bundle["threshold"]
    feature_names = bundle.get("feature_names", [])
    model_name    = bundle.get("model_name", "RF")
    val_auc       = bundle.get("val_auc", float("nan"))
    test_auc      = bundle.get("test_auc", float("nan"))

    # ══════════════════════════════════════════════════════════════════════════
    #  ADAPTIVE BASELINE MODE
    # ══════════════════════════════════════════════════════════════════════════
    if args.adaptive:
        if not args.images:
            print("[ERROR] --adaptive requires --images.", file=sys.stderr)
            return 2
        if args.dates and len(args.dates) != len(args.images):
            print(
                f"[ERROR] --dates count ({len(args.dates)}) must match "
                f"--images count ({len(args.images)}).",
                file=sys.stderr,
            )
            return 2

        # Fill in synthetic dates if not provided
        dates = args.dates if args.dates else [f"D{i+1}" for i in range(len(args.images))]

        # Import adaptive baseline from temporal manager
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.temporal.temporal_manager import AdaptiveBaseline  # noqa: PLC0415

        ab = AdaptiveBaseline(
            model         = model,
            calibrator    = calibrator,
            feature_names = feature_names,
            alert_threshold   = args.alert_threshold   or ALERT_THRESHOLD,
            promote_threshold = args.promote_threshold or PROMOTE_THRESHOLD,
        )

        for img_path, date_str in zip(args.images, dates):
            ab.step(img_path, date_str)

        summary = ab.summary()
        any_red = summary["red_alerts"] > 0

        if args.json:
            print(json.dumps({
                "mode":            "adaptive_baseline",
                "model":           model_name,
                "alert_threshold":   ab.alert_threshold,
                "promote_threshold": ab.promote_threshold,
                "val_auc":  float(val_auc)  if not (val_auc  != val_auc)  else None,
                "test_auc": float(test_auc) if not (test_auc != test_auc) else None,
                **summary,
            }, indent=2))
        else:
            BAR = 20
            print()
            print("══════════════════════════════════════════════════════")
            print("  KEMET1 — Adaptive Baseline Inference")
            print("══════════════════════════════════════════════════════")
            print(f"  Model              : {model_name}")
            print(f"  Alert threshold    : {ab.alert_threshold:.2f}  (yellow)")
            print(f"  Promote threshold  : {ab.promote_threshold:.2f}  (red / baseline moves)")
            if not (val_auc != val_auc):
                print(f"  Val AUC            : {val_auc:.4f}")
            print()

            for step in summary["history"]:
                if step["score"] is None:
                    print(f"  [INIT]  {step['current_date']}  — initial baseline registered")
                    continue
                score  = step["score"]
                alert  = step["alert"]
                moved  = step["baseline_moved"]
                bar    = ("█" * int(score * BAR)).ljust(BAR, "░")
                tip    = "(pos)" if step["t1_is_pos"] else "(neg)"
                b_date = step["baseline_date"]
                c_date = step["current_date"]
                if alert == "red":
                    icon = "⚠ RED   "
                elif alert == "yellow":
                    icon = "⚡ YELLOW"
                else:
                    icon = "✓ clear "
                moved_tag = "  ← baseline moved" if moved else ""
                print(
                    f"  {icon}  {b_date} {tip} → {c_date}"
                    f"  p={score:.3f} {bar}{moved_tag}"
                )

            print()
            print("──────────────────────────────────────────────────────")
            print(f"  Steps: {summary['steps']}  |  "
                  f"RED: {summary['red_alerts']}  |  "
                  f"YELLOW: {summary['yellow_alerts']}")
            print(f"  Final baseline: {summary['current_baseline_date']}  "
                  f"(encroached={summary['baseline_is_pos']})")
            if any_red:
                print("  RESULT: ⚠  ENCROACHMENT CONFIRMED")
            else:
                print("  RESULT: ✓  No confirmed encroachment")
            print("══════════════════════════════════════════════════════")
            print()

        return 1 if any_red else 0

    # ══════════════════════════════════════════════════════════════════════════
    #  STANDARD MODE (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.t1_paths:
        print("[ERROR] Provide --t1/--t2 pairs (standard mode) or --adaptive --images (adaptive mode).",
              file=sys.stderr)
        return 2

    if len(args.t1_paths) != len(args.t2_paths):
        print(
            f"[ERROR] --t1 and --t2 must be given the same number of times "
            f"(got {len(args.t1_paths)} vs {len(args.t2_paths)}).",
            file=sys.stderr,
        )
        return 2

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
        t1_is_pos_flags = [tip for _, _, tip in pairs]
        scores = apply_temporal_consistency(scores, threshold, t1_is_pos_flags)

    # Decisions
    decisions = [s >= threshold for s in scores]
    encroachment_detected = any(decisions)

    if args.json:
        result = {
            "mode": "standard",
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
