"""Submit scored sites as cases to the KEMET team backend.

The payload format is the one the backend expects (unchanged from v1): the
site centre, governorate and markaz from OpenStreetMap reverse geocoding, the
model's confidence and risk level, the converted area and two images.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .analysis import SiteResult
from .imagery import tile_png

BASE_URL = os.environ.get("KEMET_BACKEND_URL", "https://kemet-grad-project-backend-production.up.railway.app")
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "KEMET1-uploader/2.0"
GEOCODE_DELAY_S = 1.1  # Nominatim allows one request per second


def reverse_geocode(lat: float, lon: float, retries: int = 3) -> tuple[str, str]:
    """(governorate, markaz) for a point, or ("", "") if the lookup fails."""
    import requests

    for attempt in range(retries):
        try:
            r = requests.get(
                NOMINATIM,
                params={"format": "json", "lat": lat, "lon": lon, "zoom": 10, "accept-language": "en"},
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if "error" not in data:
                return parse_address(data.get("address", {}))
        except Exception as exc:  # network errors: retry, then give up quietly
            print(f"  geocode attempt {attempt + 1}/{retries} failed: {exc}")
        time.sleep(2)
    return "", ""


def parse_address(addr: dict) -> tuple[str, str]:
    gov = addr.get("state") or addr.get("province") or addr.get("county") or addr.get("region") or ""
    markaz = (
        addr.get("state_district")
        or addr.get("county")
        or addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or ""
    )
    return gov.replace(" Governorate", "").replace(" Muhafazat", "").strip(), markaz


def build_payload(result: SiteResult, governorate: str = "", markaz: str = "") -> dict:
    lat, lon = result.pair.centre
    year = lambda y, d: f"{y}-01-01" if y.isdigit() else d  # noqa: E731
    return {
        "lat": round(lat, 6),
        "long": round(lon, 6),
        "governorate": governorate,
        "markaz": markaz,
        "ai_confidence": round(result.probability, 4),
        "before_image_url": "",
        "after_image_url": "",
        "before_date": year(result.pair.before_year, "2024-01-01"),
        "after_date": year(result.pair.after_year, "2025-01-01"),
        "area_lost_m2": round(result.changed_ha * 10_000, 2),
        "risk_level": result.risk,
    }


def _upload_image(png_b64: str, name: str) -> str:
    import base64

    import requests

    r = requests.post(
        f"{BASE_URL}/cases/upload-image", files={"file": (name, base64.b64decode(png_b64), "image/png")}, timeout=30
    )
    r.raise_for_status()
    body = r.json()
    rel = body.get("url") or body.get("path") or body.get("imageUrl") or body.get("image_url") or body.get("filename")
    if rel is None:
        raise ValueError(f"no image URL in upload response: {body}")
    return rel if rel.startswith("http") else BASE_URL + ("" if rel.startswith("/") else "/") + rel


def submit(result: SiteResult, dry_run: bool = False, out_dir: Path | None = None) -> dict:
    """Geocode, render, upload both images and create the case. Returns the payload (and response)."""
    import requests

    lat, lon = result.pair.centre
    payload = build_payload(result, *reverse_geocode(lat, lon))
    before = tile_png(result.pair.before, f"BEFORE ({result.pair.before_year})", reference=result.pair.after)
    after = tile_png(
        result.pair.after,
        f"AFTER ({result.pair.after_year}) - {result.risk}",
        clusters=result.clusters,
        reference=result.pair.before,
    )
    if dry_run:
        return {"site": result.name, "status": "dry-run", "payload": payload}
    payload["before_image_url"] = _upload_image(before, f"{result.name}_before.png")
    payload["after_image_url"] = _upload_image(after, f"{result.name}_after.png")
    r = requests.post(f"{BASE_URL}/cases", json=payload, timeout=30)
    r.raise_for_status()
    return {"site": result.name, "status": "ok", "http": r.status_code, "payload": payload, "response": r.json()}
