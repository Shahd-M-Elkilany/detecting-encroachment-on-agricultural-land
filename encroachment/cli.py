"""Command-line interface: ``kemet1 <command>``.

kemet1 extract   tiles -> data/features.csv
kemet1 train     data/features.csv -> models/encroachment_logreg.joblib
kemet1 evaluate  full evaluation protocol -> results/evaluation.json
kemet1 figures   results -> docs/figures/*.png
kemet1 score     one site -> HTML report
kemet1 report    many sites -> HTML reports + summary map
kemet1 upload    send sites to the team backend
kemet1 serve     run the web app
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def cmd_extract(args) -> None:
    import numpy as np
    import pandas as pd
    from scipy import ndimage

    from .alarm import mean_ndvi_loss
    from .features import FEATURE_NAMES, conversion_mask, pair_features, read_pair

    rows = []
    labels = data.load_labels(args.labels)
    for i, rec in enumerate(labels, 1):
        pair = read_pair(*data.tile_paths(rec["site"], args.tiles))
        mask = conversion_mask(pair.before, pair.after)
        components, _ = ndimage.label(mask)
        largest = np.bincount(components.ravel())[1:].max(initial=0) * pair.pixel_area_ha
        lat, lon = pair.centre
        row = {
            "site": rec["site"],
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "label": int(rec["label"] == "pos"),
            "pct_converted": round(float(mask.mean()), 4),
            "max_cluster_ha": round(float(largest), 2),
            "mean_ndvi_loss": round(mean_ndvi_loss(pair.before, pair.after), 6),
        }
        row.update(zip(FEATURE_NAMES, pair_features(pair.before, pair.after).tolist(), strict=True))
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(labels)}")
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False, float_format="%.6g")
    from .features import label_rule

    agree = sum(label_rule(r["pct_converted"], r["max_cluster_ha"]) == bool(r["label"]) for r in rows)
    print(f"wrote {len(rows)} sites to {args.out}; the labelling rule reproduces {agree} of {len(rows)} labels")


def cmd_train(args) -> None:
    import numpy as np

    from . import model as model_lib
    from .evaluation import dev_test_split

    table = data.load_features(args.features)
    X, y = data.xy(table)
    dev, test = dev_test_split(y)
    fitted = model_lib.fit(X[dev], y[dev])
    model_lib.save(
        fitted,
        args.out,
        {
            "trained_on": f"{len(dev)} development sites ({int(y[dev].sum())} positive); "
            f"{len(test)} test sites held out",
            "test_sites": sorted(table["site"].to_numpy()[test].tolist()),
            "model": "StandardScaler + LogisticRegression(C=1.0)",
        },
    )
    print(f"saved {args.out} (trained on {len(dev)} sites, {int(np.sum(y[dev]))} positive)")


def cmd_evaluate(args) -> None:
    from . import alarm
    from .evaluation import full_report

    report, _, thresholds = full_report(data.load_features(args.features))
    _write_json(report, args.out)
    t = report["test"]
    print(
        f"test AUC {t['auc']:.3f} (95% CI {t['auc_ci95'][0]:.2f}-{t['auc_ci95'][1]:.2f})  "
        f"random-CV AUC {report['robustness_all_sites']['logreg']['random_5fold_x10']['mean']:.3f}  "
        f"spatial-CV AUC {report['robustness_all_sites']['logreg']['spatial_5fold_mean']:.3f}"
    )
    if (thresholds["medium"], thresholds["high"]) != (alarm.MEDIUM_THRESHOLD, alarm.HIGH_THRESHOLD):
        print(f"note: thresholds chosen now {thresholds} differ from encroachment/alarm.py", file=sys.stderr)


def cmd_figures(args) -> None:
    from .figures import make_all

    for path in make_all(args.out):
        print("wrote", path)
    if args.tiles.is_dir():
        from .figures import fig_examples

        print("wrote", fig_examples(args.tiles, args.out))


def _scorer():
    from .analysis import Scorer

    return Scorer()


def cmd_score(args) -> None:
    from .report import write_site_report

    before, after = (args.before, args.after) if args.before else data.tile_paths(args.site, args.tiles)
    result = _scorer().score_files(before, after, args.site)
    print(json.dumps(result.summary(), indent=1))
    print("report:", write_site_report(result, args.out))


def _sites(args) -> list[str]:
    labels = data.load_labels()
    if args.sites:
        return args.sites
    return [r["site"] for r in labels if args.all or r["label"] == "pos"]


def cmd_report(args) -> None:
    from .report import summary_html, write_site_report

    scorer, summaries = _scorer(), []
    for site in _sites(args):
        result = scorer.score_files(*data.tile_paths(site, args.tiles), site)
        write_site_report(result, args.out)
        summaries.append(result.summary())
        print(f"{site:8s} {result.risk:6s} p={result.probability:.3f} changed={result.changed_ha:.1f} ha")
    (args.out / "index.html").write_text(summary_html(summaries), encoding="utf-8")
    _write_json(summaries, args.out / "summary.json")
    print(f"wrote {len(summaries)} reports and index.html to {args.out}")


def cmd_upload(args) -> None:
    from .backend import GEOCODE_DELAY_S, submit

    scorer, log = _scorer(), []
    for i, site in enumerate(_sites(args)):
        if i:
            time.sleep(GEOCODE_DELAY_S)
        try:
            outcome = submit(scorer.score_files(*data.tile_paths(site, args.tiles), site), dry_run=args.dry_run)
        except Exception as exc:
            outcome = {"site": site, "status": "error", "error": str(exc)}
        log.append(outcome)
        print(site, outcome["status"], outcome.get("payload", {}).get("risk_level", ""))
    _write_json(log, Path("outputs/upload_log.json"))


def cmd_serve(args) -> None:
    from .web.app import main

    main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kemet1", description="Farmland encroachment detection from Sentinel-2")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="compute the feature table from the tiles")
    p.add_argument("--tiles", type=Path, default=data.TILES_DIR)
    p.add_argument("--labels", type=Path, default=data.LABELS)
    p.add_argument("--out", type=Path, default=data.FEATURES)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("train", help="fit the model on the development sites")
    p.add_argument("--features", type=Path, default=data.FEATURES)
    p.add_argument("--out", type=Path, default=data.MODEL)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="run the evaluation protocol")
    p.add_argument("--features", type=Path, default=data.FEATURES)
    p.add_argument("--out", type=Path, default=data.RESULTS / "evaluation.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("figures", help="draw the figures used in the README")
    p.add_argument("--out", type=Path, default=data.ROOT / "docs" / "figures")
    p.add_argument("--tiles", type=Path, default=data.TILES_DIR, help="also draw example sites if present")
    p.set_defaults(func=cmd_figures)

    for name, func, helptext in (
        ("report", cmd_report, "score sites and write HTML reports"),
        ("upload", cmd_upload, "submit sites to the team backend"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("sites", nargs="*", help="site names (default: all positive sites)")
        p.add_argument("--all", action="store_true", help="all 300 sites, not only positives")
        p.add_argument("--tiles", type=Path, default=data.TILES_DIR)
        if name == "report":
            p.add_argument("--out", type=Path, default=Path("outputs/reports"))
        else:
            p.add_argument("--dry-run", action="store_true", help="geocode and render, but send nothing")
        p.set_defaults(func=func)

    p = sub.add_parser("score", help="score one site and write its HTML report")
    p.add_argument("site", nargs="?", default="site")
    p.add_argument("--before", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--tiles", type=Path, default=data.TILES_DIR)
    p.add_argument("--out", type=Path, default=Path("outputs/reports"))
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("serve", help="run the web app (PORT env var, default 8000)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.func is cmd_score and bool(args.before) != bool(args.after):
        raise SystemExit("--before and --after go together")
    args.func(args)


if __name__ == "__main__":
    main()
