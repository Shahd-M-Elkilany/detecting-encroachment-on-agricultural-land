"""Score the 8-step pipeline against the same labels and test sites as the classifier.

The pipeline outputs maps, not probabilities. Its site-level score here is the
area it marks red (confirmed encroachment regions, in hectares). That area is
ranked against the labels exactly like the classifier's probability: ROC AUC on
all 300 sites and on the 45 held-out test sites, plus precision and recall at a
red-area threshold chosen on the 255 development sites.

    python evaluate_pipeline.py score --tiles ../data/tiles   # needs the tiles; writes ../results/pipeline_scores.csv
    python evaluate_pipeline.py report                         # from the CSV; writes ../results/pipeline_evaluation.json
    python evaluate_pipeline.py figures [--tiles ../data/tiles] # ROC of both models; example maps if tiles are given
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORES = ROOT / "results" / "pipeline_scores.csv"
REPORT = ROOT / "results" / "pipeline_evaluation.json"
FIELDS = ["site", "regions", "red_ha", "building_ha", "yellow_ha", "changed_pct", "agri_pct"]


def score(
    tiles: Path,
    out: Path = SCORES,
    sites: list[str] | None = None,
    change_dir: Path | None = None,
    variant: str = "pct_x1",
) -> Path:
    """Run the pipeline on every site. With ``change_dir`` (ChangeFormer outputs from
    changeformer/run_changeformer.py), Step 05 uses ChangeFormer's change probability."""
    sys.path.insert(0, str(HERE))
    from run_full_pipeline import analyse, load_pair, quiet

    if sites is None:
        sites = [row["site"] for row in json.loads((ROOT / "data" / "labels.json").read_text())]
    done = {}
    if out.exists():
        with out.open() as f:
            done = {r["site"]: r for r in csv.DictReader(f)}
    out.parent.mkdir(parents=True, exist_ok=True)
    quiet()
    for site in sites:
        if site in done:
            continue
        t1, t2, meta = load_pair(tiles / f"{site}_before_2024.tif", tiles / f"{site}_after_2025.tif")
        prob = None
        if change_dir is not None:
            prob = np.load(change_dir / f"{site}.npz")[variant].astype(np.float32) / 255
        done[site] = {"site": site, **analyse(t1, t2, meta, change_prob=prob)["summary"]}
        quiet()
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(done[s] for s in sorted(done, key=lambda s: int(s[4:])))
        print(f"{site}: {done[site]['red_ha']} ha red, {done[site]['regions']} regions", flush=True)
    return out


def report(scores: Path = SCORES, out: Path = REPORT) -> dict:
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import roc_auc_score

    from encroachment.data import load_features
    from encroachment.evaluation import bootstrap_auc, counts, dev_test_split

    table = load_features()
    with scores.open() as f:
        rows = {r["site"]: r for r in csv.DictReader(f)}
    missing = [s for s in table["site"] if s not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} sites have no pipeline score yet, e.g. {missing[:3]}")
    y = table["label"].to_numpy().astype(int)
    red = np.array([float(rows[s]["red_ha"]) for s in table["site"]])
    regions = np.array([int(rows[s]["regions"]) for s in table["site"]])
    dev, test = dev_test_split(y)

    # Threshold on red area, chosen on development sites: the one with the best F1.
    def f1_of(c: dict) -> float:
        return 2 * c["precision"] * c["recall"] / max(c["precision"] + c["recall"], 1e-9)

    grid = np.unique(red[dev])
    f1 = [f1_of(counts(y[dev], red[dev], t)) for t in grid]
    threshold = float(grid[int(np.argmax(f1))])
    lo, hi = bootstrap_auc(y[test], red[test])
    result = {
        "score": "red area in hectares (confirmed encroachment regions from Step 08)",
        "mode": "pre-computed index tiles: Steps 02-03 skipped, Steps 05-07 use their spectral rules",
        "sites": int(len(y)),
        "auc_all_sites": round(float(roc_auc_score(y, red)), 4),
        "auc_all_sites_region_count": round(float(roc_auc_score(y, regions)), 4),
        "development": {
            "auc": round(float(roc_auc_score(y[dev], red[dev])), 4),
            "threshold_ha": round(threshold, 4),
            "at_threshold": counts(y[dev], red[dev], threshold),
        },
        "test": {
            "sites": int(len(test)),
            "positives": int(y[test].sum()),
            "auc": round(float(roc_auc_score(y[test], red[test])), 4),
            "auc_ci95": [round(lo, 4), round(hi, 4)],
            "at_threshold": counts(y[test], red[test], threshold),
        },
        "sites_with_any_red_region": {
            "positives": int(((red > 0) & (y == 1)).sum()),
            "negatives": int(((red > 0) & (y == 0)).sum()),
        },
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _test_scores():
    """Test-site labels, classifier probabilities and pipeline red areas."""
    sys.path.insert(0, str(ROOT))
    from encroachment import data
    from encroachment import model as model_lib
    from encroachment.evaluation import dev_test_split

    table = data.load_features()
    X, y = data.xy(table)
    dev, test = dev_test_split(y)
    prob = model_lib.fit(X[dev], y[dev]).predict_proba(X[test])[:, 1]
    with SCORES.open() as f:
        red_ha = {r["site"]: float(r["red_ha"]) for r in csv.DictReader(f)}
    red = np.array([red_ha[s] for s in table["site"].to_numpy()[test]])
    return y[test], prob, red


def fig_two_models(out: Path) -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve

    from encroachment.figures import GREY, NEW, OLD, _save, _style

    plt = _style()
    y, prob, red = _test_scores()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for scores, colour, name in (
        (red, OLD, "8-step pipeline (red area)"),
        (prob, NEW, "tile classifier (probability)"),
    ):
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{name}  AUC {roc_auc_score(y, scores):.2f}")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="Both models, 45 held-out test sites")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, out, "two_models_roc.png")


def fig_pipeline_examples(
    tiles: Path, out: Path, sites=(("site264", "positive"), ("site150", "negative"), ("site299", "positive"))
) -> Path:
    """The pipeline's red / yellow / green map next to the 2025 image for the README example sites."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    from matplotlib.patches import Rectangle
    from run_full_pipeline import analyse, load_pair, quiet
    from src.step_08_final_output.generate_output import _build_colored_map, _compute_red_score

    from encroachment import alarm
    from encroachment.analysis import Scorer
    from encroachment.figures import MUTED, _save, _style
    from encroachment.imagery import false_colour

    plt = _style()
    scorer = Scorer()
    fig, axes = plt.subplots(len(sites), 2, figsize=(8.2, 4.0 * len(sites)))
    threshold = json.loads(REPORT.read_text())["development"]["threshold_ha"]

    def verdict(flagged: bool, label: str) -> str:
        if label == "positive":
            return "caught" if flagged else "missed"
        return "false alarm" if flagged else "correctly quiet"

    for row, (site, label) in zip(axes, sites, strict=True):
        before_path, after_path = tiles / f"{site}_before_2024.tif", tiles / f"{site}_after_2025.tif"
        t1, t2, meta = load_pair(before_path, after_path)
        quiet()
        res = analyse(t1, t2, meta)
        s04, s05, s06, s07 = res["s04"], res["s05"], res["s06"], res["s07"]
        _, after_rgb = false_colour(t1, t2)
        grey = (after_rgb.mean(axis=-1) * 0.8 + 0.1) * 255
        base = np.repeat(grey[..., None], 3, axis=-1).astype(np.uint8)
        red_score = _compute_red_score(s05["change_confidence"], s04["spectral_signal"], s05["change_map"])
        coloured = _build_colored_map(base, s06["agri_mask"], s04["yellow_mask"], red_score, s07["building_mask"])
        prob = scorer.score_files(before_path, after_path, site).probability
        row[0].imshow(after_rgb, interpolation="nearest")
        row[1].imshow(coloured, interpolation="nearest")
        for region in res["regions"]:
            x1, y1, x2, y2 = region["bbox_px"]
            row[1].add_patch(Rectangle((x1 - 2, y1 - 2), x2 - x1 + 4, y2 - y1 + 4, fill=False, ec="#ff2020", lw=1.6))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        red_ha = res["summary"]["red_ha"]
        row[0].set_title(
            f"{site} ({label}) · classifier {prob:.2f}: {verdict(prob >= alarm.MEDIUM_THRESHOLD, label)}", fontsize=10.5
        )
        row[1].set_title(f"8-step map · {red_ha:.1f} ha red: {verdict(red_ha >= threshold, label)}", fontsize=10.5)
    fig.text(
        0.5,
        0.005,
        "Left: 2025 false colour (R = NDBI, G = NDVI, B = MNDWI). Right, over a grey 2025 image: green = farmland in "
        "2024, yellow = NDVI fell by more than 0.15,\nred = changed farmland with a built-up signal (boxes: regions). "
        f"Flags: classifier at Medium risk (≥ {alarm.MEDIUM_THRESHOLD:.2f}), pipeline at ≥ {threshold:.1f} ha red.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, out, "pipeline_examples.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="run the pipeline on every labelled site (needs the tiles)")
    s.add_argument("--tiles", type=Path, default=ROOT / "data" / "tiles")
    s.add_argument("--out", type=Path, default=SCORES)
    s.add_argument("--changeformer", type=Path, default=None, help="folder of ChangeFormer .npz maps (Step 05)")
    s.add_argument("--variant", default="pct_x1", help="which ChangeFormer map to use")
    sub.add_parser("report", help="compare the scores with the labels")
    f = sub.add_parser("figures", help="draw the README figures for the pipeline")
    f.add_argument("--tiles", type=Path, default=None, help="tile folder; enables the example maps")
    f.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.tiles, args.out, change_dir=args.changeformer, variant=args.variant)
    elif args.cmd == "figures":
        sys.path.insert(0, str(ROOT))
        print(fig_two_models(args.out))
        if args.tiles:
            print(fig_pipeline_examples(args.tiles, args.out))
    else:
        print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
