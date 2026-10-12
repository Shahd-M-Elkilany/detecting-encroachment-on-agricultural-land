"""The 8-step pipeline: runs end to end, finds a planted conversion, and its evaluation reproduces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import make_pair, write_geotiff

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
pytest.importorskip("cv2")

if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))


@pytest.fixture(scope="module")
def runner():
    import run_full_pipeline

    run_full_pipeline.quiet()
    return run_full_pipeline


def _files(tmp_path, before, after, name="siteX"):
    return (
        write_geotiff(tmp_path / f"{name}_before_2024.tif", before),
        write_geotiff(tmp_path / f"{name}_after_2025.tif", after),
    )


def test_red_regions_cover_the_planted_conversion(runner, tmp_path):
    before, after = make_pair(size=160, block=(60, 100))  # 16 ha turned built-up
    t1, t2, meta = runner.load_pair(*_files(tmp_path, before, after))
    result = runner.analyse(t1, t2, meta)
    assert result["summary"]["regions"] >= 1
    assert 8 <= result["summary"]["red_ha"] <= 20
    for region in result["regions"]:
        x1, y1, x2, y2 = region["bbox_px"]
        assert 55 <= x1 and x2 <= 105 and 55 <= y1 and y2 <= 105


def test_unchanged_pair_has_no_red_regions(runner, tmp_path):
    before, _ = make_pair(size=120)
    after = before + np.random.default_rng(1).normal(0, 0.005, before.shape).astype(np.float32)
    result = runner.analyse(*runner.load_pair(*_files(tmp_path, before, after)))
    assert result["summary"]["red_ha"] == 0


def test_full_run_writes_the_step_08_outputs(runner, tmp_path, tile_files):
    out = runner.run_pipeline(*tile_files, "siteX", out_dir=tmp_path / "out")
    report = json.loads(Path(out["paths"]["report_json"]).read_text())
    assert report["total_regions"] == out["summary"]["regions"] >= 1
    for key in ("colored_png", "colored_tif", "geojson"):
        assert Path(out["paths"][key]).stat().st_size > 0


def test_classifier_mode_matches_the_package(tile_files):
    import importlib.util

    from encroachment.analysis import Scorer

    spec = importlib.util.spec_from_file_location("team_pipeline", PIPELINE / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    step5 = module.FoodSecurityPipeline()._step_05_kemet1(*tile_files, [])
    expected = Scorer().score_files(*tile_files).probability
    assert step5["kemet1_score"] == pytest.approx(expected)
    assert step5["change_map"].sum() > 0


def test_synthetic_test_mode_runs():
    done = subprocess.run(
        [sys.executable, "run.py", "--test"],
        cwd=PIPELINE,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PIPELINE_LOG_LEVEL": "WARNING"},
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Test mode completed" in done.stdout


def test_pipeline_evaluation_reproduces(tmp_path):
    import evaluate_pipeline

    committed = json.loads((ROOT / "results" / "pipeline_evaluation.json").read_text())
    assert evaluate_pipeline.report(out=tmp_path / "evaluation.json") == committed


def test_readmes_quote_the_pipeline_evaluation():
    result = json.loads((ROOT / "results" / "pipeline_evaluation.json").read_text())
    test = result["test"]
    expected = [
        f"{result['auc_all_sites']:.2f}",
        f"{test['auc']:.2f} (95 % CI {test['auc_ci95'][0]:.2f} to {test['auc_ci95'][1]:.2f})",
        f"{test['at_threshold']['precision']:.2f} / {test['at_threshold']['recall']:.2f}",
        f"≥ {result['development']['threshold_ha']:.1f} ha red",
    ]
    for readme in (ROOT / "README.md", PIPELINE / "README.md"):
        text = readme.read_text(encoding="utf-8")
        for snippet in expected:
            assert snippet in text, (readme.name, snippet)
