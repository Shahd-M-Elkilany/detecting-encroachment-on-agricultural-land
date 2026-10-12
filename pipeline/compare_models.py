"""Compare every KEMET1 model against the labels on the same development and test sites.

Methods, each reduced to one score per site:

- logistic regression (v2 classifier) and Random Forest (v1): probability; development
  scores are 5x repeated 5-fold out-of-fold predictions, test scores come from a model
  fitted on all 255 development sites;
- 8-step pipeline with the spectral rule in Step 05: red area (ha);
- ChangeFormer alone (LEVIR-CD weights on Sentinel-2 RGB): mean change probability or
  changed farmland area; the input variant and score are chosen on development sites;
- 8-step pipeline with ChangeFormer in Step 05: red area (ha).

For every method: ROC AUC on development sites, ROC AUC on the 45 test sites with a
bootstrap 95 % CI, and test precision / recall at the threshold that maximises F1 on
the development sites.

    python compare_models.py        # writes ../results/model_comparison.json and the figure
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "model_comparison.json"


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _best_f1_threshold(y, s, counts) -> float:
    def f1(c):
        return 2 * c["precision"] * c["recall"] / max(c["precision"] + c["recall"], 1e-9)

    grid = np.unique(s)
    return float(grid[int(np.argmax([f1(counts(y, s, t)) for t in grid]))])


def scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Labels, development and test indices, per-method scores, and the ChangeFormer variant search."""
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import roc_auc_score

    from encroachment import data
    from encroachment import model as model_lib
    from encroachment.evaluation import dev_test_split, out_of_fold

    table = data.load_features()
    sites = table["site"].to_numpy()
    X, y = data.xy(table)
    dev, test = dev_test_split(y)
    methods: dict[str, np.ndarray] = {}

    for name, kind in (("Logistic regression (v2)", "logreg"), ("Random Forest (v1)", "random_forest")):
        s = np.zeros(len(y))
        s[dev] = out_of_fold(X[dev], y[dev], kind)
        s[test] = model_lib.fit(X[dev], y[dev], kind).predict_proba(X[test])[:, 1]
        methods[name] = s

    def by_site(rows, field, variant=None):
        values = {r["site"]: float(r[field]) for r in rows if variant is None or r["variant"] == variant}
        return np.array([values[s] for s in sites])

    methods["8-step pipeline, spectral rule"] = by_site(_read(RESULTS / "pipeline_scores.csv"), "red_ha")

    search = {}
    cf_path = RESULTS / "changeformer_scores.csv"
    if cf_path.exists():
        rows = _read(cf_path)

        def choose(prefix_ok):
            candidates = {}
            for variant in sorted({r["variant"] for r in rows if prefix_ok(r["variant"])}):
                if len({r["site"] for r in rows if r["variant"] == variant}) < len(sites):
                    continue
                for field in ("mean_prob", "changed_farmland_ha"):
                    s = by_site(rows, field, variant)
                    auc = float(roc_auc_score(y[dev], s[dev])) if np.ptp(s[dev]) > 0 else 0.5
                    candidates[f"{variant} / {field}"] = (auc, s)
            return candidates

        candidates = choose(lambda v: not v.startswith("oscd_"))
        if candidates:
            search = {k: round(v[0], 4) for k, v in candidates.items()}
            chosen = max(candidates, key=lambda k: candidates[k][0])
            search["chosen_on_development"] = chosen
            methods["ChangeFormer (LEVIR-CD weights)"] = candidates[chosen][1]
        tuned = choose(lambda v: v.startswith("oscd_"))
        if tuned:
            chosen = max(tuned, key=lambda k: tuned[k][0])
            search["oscd_finetuned"] = {k: round(v[0], 4) for k, v in tuned.items()}
            search["oscd_finetuned"]["chosen_on_development"] = chosen
            methods["ChangeFormer fine-tuned on OSCD"] = tuned[chosen][1]

    pipe_candidates = {}
    for path in sorted(RESULTS.glob("pipeline_changeformer_*.csv")):
        rows = _read(path)
        if len(rows) == len(sites):
            s = by_site(rows, "red_ha")
            auc = float(roc_auc_score(y[dev], s[dev])) if np.ptp(s[dev]) > 0 else 0.5
            pipe_candidates[path.stem.removeprefix("pipeline_changeformer_")] = (auc, s)
    if pipe_candidates:
        chosen = max(pipe_candidates, key=lambda k: pipe_candidates[k][0])
        search["pipeline_step05"] = {k: round(v[0], 4) for k, v in pipe_candidates.items()}
        search["pipeline_step05"]["chosen_on_development"] = chosen
        methods["8-step pipeline, ChangeFormer in Step 05"] = pipe_candidates[chosen][1]
    return y, dev, test, methods, search


def test_only_changeformer(y, test) -> dict:
    """ChangeFormer at 4x upsampling, run on the 45 test sites only (fixed in advance, not tuned)."""
    from sklearn.metrics import roc_auc_score

    from encroachment import data
    from encroachment.evaluation import bootstrap_auc

    path = RESULTS / "changeformer_scores.csv"
    if not path.exists():
        return {}
    sites = data.load_features()["site"].to_numpy()[test]
    out = {}
    for variant in sorted({r["variant"] for r in _read(path)}):
        rows = {r["site"]: r for r in _read(path) if r["variant"] == variant}
        if not variant.endswith("x4") or not all(s in rows for s in sites):
            continue
        for field in ("mean_prob", "changed_farmland_ha"):
            s = np.array([float(rows[x][field]) for x in sites])
            if np.ptp(s) == 0:
                out[f"{variant} / {field}"] = {
                    "test_auc": 0.5,
                    "test_auc_ci95": [0.5, 0.5],
                    "sites_with_nonzero_score": 0,
                }
                continue
            lo, hi = bootstrap_auc(y[test], s)
            out[f"{variant} / {field}"] = {
                "test_auc": round(float(roc_auc_score(y[test], s)), 4),
                "test_auc_ci95": [round(lo, 4), round(hi, 4)],
                "sites_with_nonzero_score": int((s > 0).sum()),
            }
    return out


def compare(out: Path = OUT) -> dict:
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import roc_auc_score

    from encroachment.evaluation import bootstrap_auc, counts

    y, dev, test, methods, search = scores()
    result = {
        "labels": "automatic NDVI/NDBI rule (the ArcGIS-derived labels in data/labels.json)",
        "development_sites": int(len(dev)),
        "test_sites": int(len(test)),
        "test_positives": int(y[test].sum()),
        "changeformer_variant_search_dev_auc": search,
        "methods": {},
    }
    for name, s in methods.items():
        threshold = _best_f1_threshold(y[dev], s[dev], counts)
        lo, hi = bootstrap_auc(y[test], s[test])
        flat = np.ptp(s[test]) == 0
        result["methods"][name] = {
            "dev_auc": round(float(roc_auc_score(y[dev], s[dev])), 4) if np.ptp(s[dev]) else 0.5,
            "test_auc": 0.5 if flat else round(float(roc_auc_score(y[test], s[test])), 4),
            "test_auc_ci95": [0.5, 0.5] if flat else [round(lo, 4), round(hi, 4)],
            "test_at_dev_threshold": counts(y[test], s[test], threshold),
            "sites_with_nonzero_score": int((s > 0).sum()),
        }
    result["changeformer_4x_upsampled_test_sites_only"] = test_only_changeformer(y, test)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def figure(out_dir: Path = ROOT / "docs" / "figures") -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve

    sys.path.insert(0, str(ROOT))
    from encroachment.figures import GREY, _save, _style

    colours = {
        "Logistic regression (v2)": "#2a78d6",
        "Random Forest (v1)": "#eb6834",
        "8-step pipeline, spectral rule": "#1baf7a",
        "ChangeFormer (LEVIR-CD weights)": "#6250d6",
        "ChangeFormer fine-tuned on OSCD": "#d03b3b",
        "8-step pipeline, ChangeFormer in Step 05": "#eda100",
    }
    plt = _style()
    y, _, test, methods, _ = scores()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name, s in methods.items():
        if np.ptp(s[test]) == 0:
            ax.plot([0, 1], [0, 1], color=colours.get(name, GREY), lw=2, label=f"{name}  AUC 0.50 (no signal)")
            continue
        fpr, tpr, _ = roc_curve(y[test], s[test])
        ax.plot(
            fpr, tpr, color=colours.get(name, GREY), lw=2, label=f"{name}  AUC {roc_auc_score(y[test], s[test]):.2f}"
        )
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="All models, 45 held-out test sites")
    ax.legend(loc="lower right", fontsize=8.5)
    return _save(fig, out_dir, "all_models_roc.png")


def fig_changeformer_examples(
    s2_dir: Path,
    cf_dir: Path,
    out_dir: Path = ROOT / "docs" / "figures",
    sites=(("site264", "positive"), ("site150", "negative"), ("site299", "positive")),
    variants=("pct_x1", "pct_x2", "pct_x4"),
) -> Path:
    """Sentinel-2 RGB before and after, and ChangeFormer's change probability at each upsampling."""
    import rasterio

    from encroachment.figures import MUTED, _save, _style

    plt = _style()
    fig, axes = plt.subplots(len(sites), 2 + len(variants), figsize=(3.0 * (2 + len(variants)), 3.2 * len(sites)))
    for row, (site, label) in zip(axes, sites, strict=True):
        imgs = []
        for tag in ("before_2024", "after_2025"):
            with rasterio.open(s2_dir / f"{site}_{tag}.tif") as r:
                imgs.append(r.read()[[2, 1, 0]].astype(np.float32))
        both = np.concatenate([i.reshape(3, -1) for i in imgs], 1)
        lo, hi = np.percentile(both, 2, 1), np.percentile(both, 98, 1)
        maps = np.load(cf_dir / f"{site}.npz")
        for ax, img, year in zip(row[:2], imgs, ("2024", "2025"), strict=True):
            ax.imshow(np.clip(np.transpose((img - lo[:, None, None]) / (hi - lo)[:, None, None], (1, 2, 0)), 0, 1))
            ax.set_title(f"{site} ({label}) · {year}" if year == "2024" else year, fontsize=10)
        for ax, v in zip(row[2:], variants, strict=True):
            prob = maps[v].astype(np.float32) / 255 if v in maps.files else np.zeros(imgs[0].shape[1:])
            ax.imshow(prob, cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"ChangeFormer, {v.split('_')[1]} · {(prob > 0.5).mean() * 686:.1f} ha", fontsize=10)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.text(
        0.5,
        0.005,
        "Sentinel-2 true colour, 10 m. Change probability from the LEVIR-CD ChangeFormerV6 checkpoint (black 0, yellow 1); "
        "x2 and x4 upsample the images before inference. Area = pixels above 0.5.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, out_dir, "changeformer_examples.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare every model against the labels")
    parser.add_argument("--s2", type=Path, help="Sentinel-2 tiles (enables the ChangeFormer example figure)")
    parser.add_argument("--changeformer", type=Path, help="ChangeFormer .npz maps")
    args = parser.parse_args()
    print(json.dumps(compare(), indent=2))
    print(figure())
    if args.s2 and args.changeformer:
        print(fig_changeformer_examples(args.s2, args.changeformer))
