"""End-to-end check of the web API with synthetic GeoTIFFs."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import write_geotiff  # noqa: E402


@pytest.fixture(scope="module")
def client():
    import app

    return TestClient(app.app)


def post(client, before_path, after_path):
    with open(before_path, "rb") as b, open(after_path, "rb") as a:
        return client.post("/predict", files={"before": ("b.tif", b, "image/tiff"), "after": ("a.tif", a, "image/tiff")})


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "<html" in response.text.lower()


def test_predict_returns_a_scored_site(client, tile_pair, tmp_path):
    before, after = tile_pair
    response = post(client, write_geotiff(tmp_path / "b.tif", before), write_geotiff(tmp_path / "a.tif", after))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk_level"] in {"High", "Medium", "Low"}
    assert 0 <= body["rf_prob"] <= 1
    assert body["clusters"] == 1 and body["area_m2"] == pytest.approx(90_000)
    assert body["area_feddan"] == pytest.approx(90_000 / 4200, abs=0.01)
    assert 29 < body["lat"] < 32 and 29 < body["lon"] < 34, "centre is reported in degrees"
    assert body["before_img"] and body["after_img"]


def test_unchanged_site_scores_lower(client, tile_pair, tmp_path):
    before, after = tile_pair
    b = write_geotiff(tmp_path / "b.tif", before)
    changed = post(client, b, write_geotiff(tmp_path / "a.tif", after)).json()
    same = post(client, b, b).json()
    assert same["clusters"] == 0 and same["area_m2"] == 0
    assert same["fusion_score"] < changed["fusion_score"]


def test_mismatched_and_malformed_uploads_are_rejected(client, tile_pair, tmp_path):
    before, after = tile_pair
    b = write_geotiff(tmp_path / "b.tif", before)
    small = write_geotiff(tmp_path / "small.tif", after[:, :60, :60])
    assert post(client, b, small).status_code == 422
    three_band = write_geotiff(tmp_path / "rgb.tif", before[:3])
    assert post(client, three_band, three_band).status_code == 422
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff")
    assert post(client, junk, junk).status_code == 422
