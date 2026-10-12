"""Figures for the README, drawn from data/features.csv and results/evaluation.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import alarm, data
from . import model as model_lib
from .evaluation import dev_test_split, spatial_folds

NEW, OLD, GREY, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
FOLDS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")
RISK = {"High": "#d03b3b", "Medium": "#eda100"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "legend.frameon": False,
        }
    )
    return plt


def _save(fig, out: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_comparison(report: dict, out: Path) -> Path:
    plt = _style()
    rob, test = report["robustness_all_sites"], report["test"]
    rows = [
        (
            "Held-out test sites (45)",
            (test["auc"], *test["auc_ci95"]),
            (test["random_forest_v1_auc"], *test["random_forest_v1_auc_ci95"]),
        ),
        ("Random 5-fold CV ×10, all sites", None, None),
        ("Spatial 5-fold CV (regions held out)", None, None),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for i, (_label, new, old) in enumerate(rows):
        y = len(rows) - 1 - i
        for kind, colour, dy, fixed in (("logreg", NEW, 0.13, new), ("random_forest", OLD, -0.13, old)):
            if fixed is not None:
                mean, lo, hi = fixed
            elif i == 1:
                s = rob[kind]["random_5fold_x10"]
                mean, lo, hi = s["mean"], s["mean"] - s["std"], s["mean"] + s["std"]
            else:
                folds = rob[kind]["spatial_5fold"]
                mean, lo, hi = float(np.mean(folds)), min(folds), max(folds)
                ax.scatter(folds, [y + dy] * len(folds), s=14, color=colour, alpha=0.35, zorder=2)
            ax.plot([lo, hi], [y + dy, y + dy], color=colour, lw=2, zorder=2)
            ax.scatter([mean], [y + dy], s=52, color=colour, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(hi + 0.008, y + dy, f"{mean:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0.5, 1.04)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC AUC  (whiskers: 95 % CI, ±1 SD, or fold range)")
    ax.set_title("v2 logistic model vs v1 Random Forest, same features and sites")
    ax.scatter([], [], color=NEW, label="v2 logistic regression")
    ax.scatter([], [], color=OLD, label="v1 Random Forest")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out, "model_comparison.png")


def _test_scores(table):
    X, y = data.xy(table)
    dev, test = dev_test_split(y)
    new = model_lib.fit(X[dev], y[dev]).predict_proba(X[test])[:, 1]
    old = model_lib.fit(X[dev], y[dev], "random_forest").predict_proba(X[test])[:, 1]
    return y[test], new, old


def fig_roc_pr(table, out: Path) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    plt = _style()
    y, new, old = _test_scores(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for scores, colour, name in ((old, OLD, "v1 Random Forest"), (new, NEW, "v2 logistic")):
        fpr, tpr, _ = roc_curve(y, scores)
        axes[0].plot(fpr, tpr, color=colour, lw=2, label=f"{name}  AUC {roc_auc_score(y, scores):.2f}")
        prec, rec, _ = precision_recall_curve(y, scores)
        axes[1].plot(rec, prec, color=colour, lw=2, label=f"{name}  AP {average_precision_score(y, scores):.2f}")
    axes[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (4, 3)))
    axes[1].axhline(y.mean(), color=GREY, lw=1, ls=(0, (4, 3)))
    axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC, 45 held-out test sites")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall, same sites", ylim=(0, 1.03))
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=9)
    return _save(fig, out, "roc_pr.png")


def fig_thresholds(table, out: Path) -> Path:
    plt = _style()
    y, new, _ = _test_scores(table)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    for label, row, colour, name in ((0, 0, GREY, "not flagged by the label rule"), (1, 1, NEW, "flagged")):
        p = new[y == label]
        ax.scatter(
            p,
            row + rng.uniform(-0.18, 0.18, len(p)),
            s=34,
            color=colour,
            alpha=0.85,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=f"{name} ({len(p)})",
        )
    for name, t in (("Medium", alarm.MEDIUM_THRESHOLD), ("High", alarm.HIGH_THRESHOLD)):
        ax.axvline(t, color=RISK[name], lw=1.6, ls=(0, (5, 3)))
        ax.text(t + 0.01, 1.42, f"{name} ≥ {t:.2f}", color=RISK[name], fontsize=9, va="center")
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("probability of encroachment (v2 model)")
    ax.set_title("Where the 45 test sites fall relative to the alarm thresholds")
    ax.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.28), ncol=2)
    return _save(fig, out, "test_scores.png")


def fig_site_map(table, out: Path) -> Path:
    plt = _style()
    lat, lon, y = table["lat"].to_numpy(), table["lon"].to_numpy(), table["label"].to_numpy()
    folds = spatial_folds(lat, lon)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for f in range(folds.max() + 1):
        m = folds == f
        ax.scatter(lon[m & (y == 0)], lat[m & (y == 0)], s=22, facecolor="none", edgecolor=FOLDS[f], linewidth=1.1)
        ax.scatter(
            lon[m & (y == 1)],
            lat[m & (y == 1)],
            s=26,
            color=FOLDS[f],
            edgecolor=SURFACE,
            linewidth=0.6,
            label=f"region {f + 1}: {m.sum()} sites, {int(y[m].sum())} flagged",
        )
    ax.scatter([], [], s=22, facecolor="none", edgecolor=GREY, label="hollow = not flagged")
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_title("The 300 Nile Delta sites and the five regions used for spatial CV")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return _save(fig, out, "site_map.png")


def make_all(out: Path) -> list[Path]:
    table = data.load_features()
    report = json.loads((data.RESULTS / "evaluation.json").read_text())
    return [fig_comparison(report, out), fig_roc_pr(table, out), fig_thresholds(table, out), fig_site_map(table, out)]
