"""Repository hygiene: README numbers match the results, no personal paths, CLI wiring."""

import json
import re
import subprocess

from encroachment import alarm, data
from encroachment.cli import build_parser

README = (data.ROOT / "README.md").read_text(encoding="utf-8")
REPORT = json.loads((data.RESULTS / "evaluation.json").read_text())


def row(label):
    return next(line for line in README.splitlines() if line.startswith("|") and label in line)


def numbers(line):
    return re.findall(r"\d\.\d{2,3}", line)


def test_headline_numbers_match_the_evaluation():
    rob, test = REPORT["robustness_all_sites"], REPORT["test"]
    assert f"{rob['logreg']['random_5fold_x10']['mean']:.2f}" in numbers(row("Random 5-fold CV"))
    assert f"{rob['random_forest']['random_5fold_x10']['mean']:.2f}" in numbers(row("Random 5-fold CV"))
    assert f"{rob['logreg']['spatial_5fold_mean']:.2f}" in numbers(row("Spatial 5-fold CV"))
    assert f"{rob['random_forest']['spatial_5fold_mean']:.2f}" in numbers(row("Spatial 5-fold CV"))
    assert f"{test['auc']:.2f}" in numbers(row("Held-out test sites"))
    assert f"{test['random_forest_v1_auc']:.2f}" in numbers(row("Held-out test sites"))


def test_alarm_table_matches_the_evaluation():
    for tier in ("high", "medium"):
        c = REPORT["test"][tier]
        line = row(f"| {tier.capitalize()} ")
        assert f"{c['tp']} / {c['fp']} / {c['fn']}" in line
        assert f"{c['precision']:.2f}" in line and f"{c['recall']:.2f}" in line
    assert f"{alarm.HIGH_THRESHOLD:.2f}" in row("| High ") and f"{alarm.MEDIUM_THRESHOLD:.2f}" in row("| Medium ")


def test_no_personal_file_paths_are_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=data.ROOT, capture_output=True, text=True).stdout.split()
    offenders = [
        name
        for name in tracked
        if name.endswith((".html", ".py", ".md", ".json", ".js", ".csv", ".yml", ".toml"))
        and not name.startswith("tests/")
        and (
            "C:\\Users\\" in (text := (data.ROOT / name).read_text(encoding="utf-8", errors="ignore"))
            or "C:/Users/" in text
        )
    ]
    assert offenders == []


def test_cli_commands():
    parser = build_parser()
    for command in ("extract", "train", "evaluate", "figures", "serve"):
        assert parser.parse_args([command]).command == command
    args = parser.parse_args(["report", "site0", "site3"])
    assert args.sites == ["site0", "site3"] and not args.all
    assert parser.parse_args(["upload", "--dry-run"]).dry_run
