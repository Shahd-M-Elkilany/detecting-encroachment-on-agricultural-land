"""Site analysis, HTML reports, backend payloads and the web API, on synthetic tiles."""

import json

import pytest

from encroachment import alarm, backend, report
from encroachment.analysis import Scorer


@pytest.fixture(scope="module")
def scorer():
    return Scorer()


def test_scoring_a_pair(scorer, tile_files):
    result = scorer.score_files(*tile_files)
    assert result.name == "siteX"
    assert 0 <= result.probability <= 1 and result.risk == alarm.risk_level(result.probability)
    assert result.changed_ha == pytest.approx(9.0) and len(result.clusters) == 1
    summary = result.summary()
    assert summary["clusters"] == 1 and json.dumps(summary)


def test_no_change_scores_lower(scorer, tile_files):
    changed = scorer.score_files(*tile_files)
    same = scorer.score_files(tile_files[0], tile_files[0], "same")
    assert same.clusters == [] and same.probability < changed.probability


def test_site_report(scorer, tile_files, tmp_path):
    result = scorer.score_files(*tile_files)
    path = report.write_site_report(result, tmp_path)
    html = path.read_text()
    assert path.name == "siteX_report.html"
    assert "$" not in html.split("<script>")[0], "every template field was filled"
    assert f"{result.risk} risk" in html and "9.0 ha" in html and "data:image/png;base64," in html


def test_summary_page(scorer, tile_files):
    s = scorer.score_files(*tile_files).summary()
    html = report.summary_html([s])
    assert "siteX_report.html" in html and "KEMET1 - 1 sites" in html


def test_backend_payload(scorer, tile_files):
    result = scorer.score_files(*tile_files)
    payload = backend.build_payload(result, "Gharbia", "Tanta")
    assert payload["risk_level"] == result.risk and payload["ai_confidence"] == round(result.probability, 4)
    assert payload["area_lost_m2"] == pytest.approx(90_000)
    assert (payload["before_date"], payload["after_date"]) == ("2024-01-01", "2025-01-01")
    assert set(payload) == {
        "lat",
        "long",
        "governorate",
        "markaz",
        "ai_confidence",
        "before_image_url",
        "after_image_url",
        "before_date",
        "after_date",
        "area_lost_m2",
        "risk_level",
    }


def test_address_parsing():
    assert backend.parse_address({"state": "Gharbia Governorate", "state_district": "Tanta"}) == ("Gharbia", "Tanta")
    assert backend.parse_address({}) == ("", "")


def test_dry_run_sends_nothing(scorer, tile_files, monkeypatch):
    monkeypatch.setattr(backend, "reverse_geocode", lambda lat, lon: ("Gharbia", "Tanta"))

    def refuse(*a, **k):
        raise AssertionError("dry run must not touch the network")

    import requests

    monkeypatch.setattr(requests, "post", refuse)
    out = backend.submit(scorer.score_files(*tile_files), dry_run=True)
    assert out["status"] == "dry-run" and out["payload"]["governorate"] == "Gharbia"


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from encroachment.web.app import app

    return TestClient(app)


def _post(client, before, after):
    with open(before, "rb") as b, open(after, "rb") as a:
        return client.post("/predict", files={"before": ("b.tif", b), "after": ("a.tif", a)})


def test_web_page_and_health(client):
    assert "<html" in client.get("/").text.lower()
    health = client.get("/health").json()
    assert health["thresholds"] == {"medium": alarm.MEDIUM_THRESHOLD, "high": alarm.HIGH_THRESHOLD}


def test_web_predict(client, tile_files):
    r = _post(client, *tile_files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] == alarm.risk_level(body["probability"])
    assert body["rf_prob"] == body["fusion_score"] == body["probability"], "fields the web page reads"
    assert body["area_m2"] == pytest.approx(90_000) and body["area_feddan"] == pytest.approx(21.43, abs=0.01)
    assert body["before_img"] and body["after_img"]


def test_web_rejects_bad_uploads(client, tile_pair, tmp_path):
    from conftest import write_geotiff

    before, after = tile_pair
    b = write_geotiff(tmp_path / "b.tif", before)
    assert _post(client, b, write_geotiff(tmp_path / "s.tif", after[:, :50, :50])).status_code == 422
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff")
    assert _post(client, junk, junk).status_code == 422
