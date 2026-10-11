"""The README must quote the numbers that are in the results files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load(path):
    return json.loads((ROOT / path).read_text())


def row(label):
    return next(line for line in README.splitlines() if line.startswith("|") and label in line)


def test_headline_aucs():
    clean = load("data/ba_clean_results.json")
    assert f"| {clean['val_auc']:.3f} |" in row("Validation, seed 42")
    assert f"| {clean['test_auc']:.3f} |" in row("Held-out test, seed 42")
    assert f"{clean['ci_lo']:.2f} to {clean['ci_hi']:.2f}" in row("Validation, seed 42")


def test_geographic_holdout():
    geo = load("data/geo_holdout.json")
    assert f"| {geo['train_south_test_north']['auc']:.3f} |" in row("train south / test north")
    assert f"| {geo['train_north_test_south']['auc']:.3f} |" in row("train north / test south")


def test_repeated_splits_row():
    spread = load("results/repeated_splits.json")["test_auc"]
    line = row("30 random splits")
    assert f"{spread['mean']:.3f} ± {spread['std']:.3f}" in line
    assert f"{spread['min']:.2f} to {spread['max']:.2f}" in line


def test_alarm_table_and_thresholds():
    import alarm

    test = load("results/repeated_splits.json")["shipped_model_seed42"]["test"]
    cases = {
        f"fusion ≥ {alarm.YELLOW_THRESHOLD:.2f}": test["alarm"]["medium"],
        f"High (fusion ≥ {alarm.ALERT_THRESHOLD:.2f})": test["alarm"]["high"],
        "fusion ≥ 0.40": test["previous_high_threshold_0.40"],
    }
    for label, r in cases.items():
        expected = f"| {r['precision']:.2f} | {r['recall']:.2f} | {r['tp']} / {r['fp']} / {r['fn']} |"
        assert expected in row(label), label
    assert test["alarm"]["medium"]["threshold"] == alarm.YELLOW_THRESHOLD
    assert test["alarm"]["high"]["threshold"] == alarm.ALERT_THRESHOLD


def test_label_counts():
    labels = load("data/ba_labels.json")
    positives = sum(r["label"] == "pos" for r in labels)
    assert f"{positives} positive\n  and {len(labels) - positives} negative" in README
