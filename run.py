"""
Food Security ML Pipeline — CLI Entry Point

Single-run modes:
    python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif
    python run.py --gee                  # Download from GEE
    python run.py --test                 # Run with synthetic data
    python run.py --start-from 5 ...    # Resume from a specific step

Multi-temporal mode (recommended for ongoing monitoring):
    # First call — provide both the baseline AND the new image
    python run.py --temporal --new-image 2024.tif --new-date 2024 ^
                  --first-t1 2023.tif --first-date 2023

    # Every subsequent call — just add the new image; history is automatic
    python run.py --temporal --new-image 2025.tif --new-date 2025

    # Check what the pipeline will do without running it
    python run.py --temporal --new-image 2025.tif --new-date 2025 --dry-run

    # Wipe history and start fresh
    python run.py --temporal-reset
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food Security ML Pipeline — Detect Buildings on Agricultural Land",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard single run
  python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif

  # Multi-temporal: first run (establishes baseline)
  python run.py --temporal --new-image images/2024.tif --new-date 2024 ^
                --first-t1 images/2023.tif --first-date 2023

  # Multi-temporal: add 2025 image (pipeline decides comparisons automatically)
  python run.py --temporal --new-image images/2025.tif --new-date 2025

  # Download from Google Earth Engine
  python run.py --gee

  # Run with synthetic test data (no model weights needed)
  python run.py --test
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--t1",             type=str, help="Path to T1 (before) GeoTIFF")
    mode_group.add_argument("--gee",            action="store_true", help="Download from GEE")
    mode_group.add_argument("--test",           action="store_true", help="Synthetic test data")
    mode_group.add_argument("--temporal",       action="store_true", help="Multi-temporal mode")
    mode_group.add_argument("--temporal-reset", action="store_true", help="Clear temporal history")

    # Single-run args
    parser.add_argument("--t2",         type=str, help="Path to T2 GeoTIFF (with --t1)")
    parser.add_argument("--start-from", type=int, default=1, choices=range(1, 9),
                        help="Resume from step N (1-8)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    # Temporal args
    parser.add_argument("--new-image",  type=str, help="New image to add (--temporal)")
    parser.add_argument("--new-date",   type=str, help="Date label for new image, e.g. '2025'")
    parser.add_argument("--first-t1",   type=str, help="Baseline T1 image (first temporal run only)")
    parser.add_argument("--first-date", type=str, help="Date label for baseline T1")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Show what comparisons would run without executing them")

    args = parser.parse_args()

    if args.t1 and not args.t2:
        parser.error("--t2 is required when using --t1")
    if args.temporal:
        if not args.new_image:
            parser.error("--new-image is required with --temporal")
        if not args.new_date:
            parser.error("--new-date is required with --temporal")

    return args


def run_test_mode():
    from src.utils.geo_utils import create_synthetic_geotiff
    from config.settings import RAW_DIR

    print("\n" + "=" * 60)
    print("  RUNNING IN TEST MODE (synthetic data)")
    print("=" * 60 + "\n")

    t1_path = create_synthetic_geotiff(RAW_DIR / "T1" / "T1_synthetic.tif")
    t2_path = create_synthetic_geotiff(RAW_DIR / "T2" / "T2_synthetic.tif")
    print(f"Created synthetic T1: {t1_path}")
    print(f"Created synthetic T2: {t2_path}")

    from pipeline import FoodSecurityPipeline
    pipe    = FoodSecurityPipeline()
    results = pipe.run_full(t1_path=t1_path, t2_path=t2_path)
    print("\n✅ Test mode completed successfully!")
    return results


def run_temporal_mode(args):
    from src.temporal.temporal_manager import load_state, get_comparison_plan

    state = load_state()
    plan  = get_comparison_plan(args.new_image, args.new_date, state)

    print("\n" + "=" * 60)
    print("  TEMPORAL MODE")
    print("=" * 60)

    if plan["is_first_run"]:
        if not args.first_t1 or not args.first_date:
            print(
                "\n  ⚠  This is the first run — no history yet.\n"
                "  Provide --first-t1 and --first-date to set the baseline.\n"
                "\n  Example:\n"
                "    python run.py --temporal --new-image 2024.tif --new-date 2024 \\\n"
                "                  --first-t1 2023.tif --first-date 2023\n"
            )
            sys.exit(1)
        print(f"  First run — baseline: {args.first_date}")
        print(f"  Comparison: {args.first_date} → {args.new_date}")
    elif plan["mode"] == "rolling":
        comp = plan["comparisons"][0]
        print(f"  Mode: ROLLING WINDOW (no prior change detected)")
        print(f"  Comparing: {comp['t1_date']} → {comp['t2_date']}")
    elif plan["mode"] == "dual":
        print(f"  Mode: DUAL (prior change detected)")
        for c in plan["comparisons"]:
            tag = "cumulative (total)" if c["label"] == "cumulative" else "incremental (new only)"
            print(f"    [{tag}] {c['t1_date']} → {c['t2_date']}")

    print("=" * 60 + "\n")

    if args.dry_run:
        print("  [DRY RUN] No pipeline executed.\n")
        return

    from pipeline import FoodSecurityPipeline
    pipe = FoodSecurityPipeline()
    output = pipe.run_temporal(
        new_image_path = args.new_image,
        new_date       = args.new_date,
        first_t1_path  = args.first_t1,
        first_t1_date  = args.first_date,
    )

    regions  = output["regions"]
    n_new    = sum(1 for r in regions if r.get("encroachment_type") == "new_encroachment")
    n_exist  = sum(1 for r in regions if r.get("encroachment_type") == "existing_encroachment")
    total_ha = sum(r.get("area_ha", 0) for r in regions)

    print("\n" + "=" * 60)
    print("  TEMPORAL RESULTS")
    print("=" * 60)
    print(f"  Mode:               {output['mode']}")
    print(f"  New encroachment:   {n_new} regions")
    print(f"  Existing (prior):   {n_exist} regions")
    print(f"  Total area lost:    {total_ha:.2f} ha")
    print("=" * 60 + "\n")

    if output["results"]:
        last_paths = output["results"][-1]["result"].get("step_08", {}).get("paths", {})
        if last_paths:
            print("  Output files:")
            for k, v in last_paths.items():
                print(f"    {k}: {v}")
    print()


def main():
    args = parse_args()

    if args.temporal_reset:
        from src.temporal.temporal_manager import reset_state
        reset_state()
        print("✅ Temporal history cleared.")
        return

    if args.test:
        run_test_mode()
        return

    if args.temporal:
        run_temporal_mode(args)
        return

    from pipeline import FoodSecurityPipeline
    pipe = FoodSecurityPipeline()

    if args.gee:
        results = pipe.run_full(use_gee=True, start_from=args.start_from)
    else:
        results = pipe.run_full(
            t1_path=args.t1, t2_path=args.t2, start_from=args.start_from
        )

    if "step_08" in results:
        report   = results["step_08"].get("report", {})
        print("\n" + "=" * 60)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Regions detected:    {report.get('total_regions', 'N/A')}")
        print(f"  Encroachment area:   {report.get('encroachment_ha', 'N/A')} ha")
        print(f"  Yellow alert area:   {report.get('yellow_alert_ha', 'N/A')} ha")
        print("=" * 60)
        paths = results["step_08"].get("paths", {})
        print("\n  Output Files:")
        for fmt, path in paths.items():
            print(f"    {fmt}: {path}")
        print()


if __name__ == "__main__":
    main()
