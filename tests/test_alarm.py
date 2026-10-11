"""One alarm definition, used by every script."""
import numpy as np
import pytest

import alarm
import batch_upload
import run_inference
import upload_case


def test_thresholds_are_ordered_and_shared():
    assert 0 < alarm.YELLOW_THRESHOLD < alarm.ALERT_THRESHOLD < 1
    for module in (run_inference, upload_case, batch_upload):
        assert module.ALERT_THRESHOLD == alarm.ALERT_THRESHOLD, module.__name__
        assert module.YELLOW_THRESHOLD == alarm.YELLOW_THRESHOLD, module.__name__


def test_web_app_uses_the_shared_definition():
    app = pytest.importorskip("app")
    assert app.risk_level is alarm.risk_level and app.fusion_score is alarm.fusion_score


def test_risk_levels_at_the_boundaries():
    assert alarm.risk_level(alarm.ALERT_THRESHOLD) == "High"
    assert alarm.risk_level(alarm.ALERT_THRESHOLD - 1e-6) == "Medium"
    assert alarm.risk_level(alarm.YELLOW_THRESHOLD) == "Medium"
    assert alarm.risk_level(alarm.YELLOW_THRESHOLD - 1e-6) == "Low"
    assert alarm.risk_level(0.0) == "Low" and alarm.risk_level(1.0) == "High"


def test_spectral_score_counts_only_vegetation_loss():
    before = np.zeros((6, 10, 10), np.float32)
    after = before.copy()
    before[0] = 0.6
    after[0] = 0.6
    after[0, :5] = 0.2   # half the tile loses 0.4 NDVI
    after[0, 5:] = 0.9   # the other half gains; gains must not cancel losses
    assert alarm.spectral_score(before, after) == pytest.approx(0.2)
    assert alarm.spectral_score(before, before) == 0.0


def test_fusion_score_is_the_documented_weighted_sum(tile_pair):
    before, after = tile_pair
    spectral = alarm.spectral_score(before, after)
    assert alarm.fusion_score(0.5, before, after) == pytest.approx(0.65 * 0.5 + 0.35 * spectral)
    assert alarm.RF_WEIGHT + alarm.SPECTRAL_WEIGHT == pytest.approx(1.0)


def test_report_script_scores_with_the_shared_function():
    """run_inference.py used a different spectral formula; it must now call alarm.py."""
    import inspect

    source = inspect.getsource(run_inference.run)
    assert "_alarm.fusion_score(" in source and "_alarm.spectral_score(" in source
    assert "* 1.5" not in source


def test_api_risk_level_follows_the_thresholds(tile_pair, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import app
    from conftest import write_geotiff

    before, after = tile_pair
    b, a = write_geotiff(tmp_path / "b.tif", before), write_geotiff(tmp_path / "a.tif", after)
    with open(b, "rb") as fb, open(a, "rb") as fa:
        body = TestClient(app.app).post("/predict", files={"before": fb, "after": fa}).json()
    assert body["risk_level"] == alarm.risk_level(body["fusion_score"])
