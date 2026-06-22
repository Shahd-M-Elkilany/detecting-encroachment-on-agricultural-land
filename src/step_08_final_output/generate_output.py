"""
Step 08 — Final Output
Generates all deliverables:
  • Colored encroachment map (PNG + GeoTIFF)
  • Per-region before/after chips with lat/lon bounding box
  • Yellow alert (spectral degradation) + weighted Red alert (change + spectral)
  • Reverse geocoded location names (OpenStreetMap Nominatim)
  • Total area lost in hectares
  • Interactive Folium map (HTML)
  • JSON summary report

Alert weights (from config):
  red_score = 0.65 × change_confidence + 0.35 × spectral_signal
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from config.settings import FINAL_OUTPUT_CONFIG, OUTPUT_DIR
from src.utils.logger import get_logger
from src.utils.geo_utils import bbox_to_latlon, pixel_area_m2, write_geotiff

logger = get_logger("step_08")
CFG = FINAL_OUTPUT_CONFIG

# Alert colour palette (BGR for OpenCV)
COLOR_RED    = (0,   0,   255)   # confirmed encroachment
COLOR_YELLOW = (0,   200, 255)   # spectral degradation
COLOR_GREEN  = (0,   180, 0)     # stable agricultural land


# ── Public entry point ───────────────────────────────────────────────────────

def run(
    t2_rgb:             np.ndarray,          # [H,W,3] uint8 — T2 RGB image
    change_map:         np.ndarray,          # [H,W] uint8 binary
    agri_mask:          np.ndarray,          # [H,W] uint8 binary
    building_mask:      np.ndarray,          # [H,W] uint8 binary
    polygons:           List[Dict],          # from Step 07
    meta:               Dict[str, Any],      # rasterio meta (transform, crs, …)
    # Extended inputs from new steps
    change_confidence:  Optional[np.ndarray] = None,   # [H,W] float Step 05
    spectral_signal:    Optional[np.ndarray] = None,   # [H,W] float Step 04
    yellow_mask:        Optional[np.ndarray] = None,   # [H,W] uint8 Step 04
    t1_rgb:             Optional[np.ndarray] = None,   # [H,W,3] uint8 for chips
) -> Dict[str, Any]:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = meta.get("transform")

    # ── 1. Weighted red-alert score ─────────────────────────────────────────
    red_score = _compute_red_score(change_confidence, spectral_signal, change_map)

    # ── 2. Build colored map ────────────────────────────────────────────────
    colored = _build_colored_map(
        t2_rgb, agri_mask, yellow_mask, red_score, building_mask
    )

    # ── 3. Extract per-region detections ────────────────────────────────────
    regions = _extract_regions(
        red_score, building_mask, agri_mask, transform,
        t1_rgb, t2_rgb, yellow_mask, spectral_signal
    )
    logger.info(f"Detected {len(regions)} encroachment regions")

    # ── 4. Reverse geocode each region ──────────────────────────────────────
    regions = _geocode_regions(regions)

    # ── 5. Total area lost ──────────────────────────────────────────────────
    pixel_m2  = pixel_area_m2(transform) if transform else 100.0
    total_ha  = float(building_mask.sum()) * pixel_m2 / 10_000
    yellow_ha = float((yellow_mask > 0).sum()) * pixel_m2 / 10_000 if yellow_mask is not None else 0.0
    logger.info(f"Total encroachment area: {total_ha:.2f} ha")
    logger.info(f"Spectral degradation area: {yellow_ha:.2f} ha")

    # ── 6. Save colored map ─────────────────────────────────────────────────
    import cv2
    colored_path  = OUTPUT_DIR / "final_colored.png"
    geotiff_path  = OUTPUT_DIR / "final_colored.tif"
    cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    write_geotiff(geotiff_path, np.transpose(colored, (2, 0, 1)), meta)

    # ── 7. Save per-region chips ────────────────────────────────────────────
    chips_dir = OUTPUT_DIR / "region_chips"
    chips_dir.mkdir(exist_ok=True)
    _save_region_chips(regions, chips_dir, t1_rgb, t2_rgb, transform)

    # ── 8. GeoJSON ──────────────────────────────────────────────────────────
    geojson_path = OUTPUT_DIR / "encroachment.geojson"
    _save_geojson(regions, geojson_path)

    # ── 9. Interactive Folium map ────────────────────────────────────────────
    map_path = OUTPUT_DIR / "interactive_map.html"
    _build_folium_map(regions, colored, meta, map_path)

    # ── 10. JSON report ─────────────────────────────────────────────────────
    report = {
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_regions":      len(regions),
        "encroachment_ha":    round(total_ha, 4),
        "yellow_alert_ha":    round(yellow_ha, 4),
        "regions":            regions,
        "alert_weights": {
            "change_detection": CFG["red_alert_change_weight"],
            "spectral":         CFG["red_alert_spectral_weight"],
        },
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Outputs written to {OUTPUT_DIR}")
    return {
        "report":  report,
        "regions": regions,
        "paths": {
            "colored_png":      str(colored_path),
            "colored_tif":      str(geotiff_path),
            "geojson":          str(geojson_path),
            "interactive_map":  str(map_path),
            "report_json":      str(report_path),
            "chips_dir":        str(chips_dir),
        },
    }


# ── Alert scoring ────────────────────────────────────────────────────────────

def _compute_red_score(
    change_confidence: Optional[np.ndarray],
    spectral_signal:   Optional[np.ndarray],
    change_map:        np.ndarray,
) -> np.ndarray:
    """
    red_score = 0.65 × change_confidence + 0.35 × spectral_signal
    Falls back to binary change_map when confidence maps are unavailable.
    """
    cw = CFG["red_alert_change_weight"]
    sw = CFG["red_alert_spectral_weight"]

    H, W = change_map.shape

    if change_confidence is None:
        change_confidence = change_map.astype(np.float32)
    if spectral_signal is None:
        spectral_signal = np.zeros((H, W), dtype=np.float32)

    # Clip to [0,1]
    cc = np.clip(change_confidence, 0, 1)
    ss = np.clip(spectral_signal,   0, 1)

    red_score = cw * cc + sw * ss
    return red_score.astype(np.float32)


# ── Colored map ──────────────────────────────────────────────────────────────

def _build_colored_map(
    t2_rgb:        np.ndarray,
    agri_mask:     np.ndarray,
    yellow_mask:   Optional[np.ndarray],
    red_score:     np.ndarray,
    building_mask: np.ndarray,
) -> np.ndarray:
    colored = t2_rgb.copy()
    threshold = CFG["red_alert_threshold"]

    # Green overlay — stable agri land
    stable = (agri_mask > 0) & (building_mask == 0)
    colored[stable] = (colored[stable] * 0.5 + np.array(COLOR_GREEN) * 0.5).astype(np.uint8)

    # Yellow overlay — spectral degradation
    if yellow_mask is not None:
        ym = (yellow_mask > 0) & (building_mask == 0)
        colored[ym] = (colored[ym] * 0.4 + np.array(COLOR_YELLOW) * 0.6).astype(np.uint8)

    # Red overlay — confirmed encroachment
    red_pixels = (red_score >= threshold) & (building_mask > 0)
    colored[red_pixels] = (colored[red_pixels] * 0.3 + np.array(COLOR_RED) * 0.7).astype(np.uint8)

    return colored


# ── Region extraction ────────────────────────────────────────────────────────

def _extract_regions(
    red_score:     np.ndarray,
    building_mask: np.ndarray,
    agri_mask:     np.ndarray,
    transform,
    t1_rgb:        Optional[np.ndarray],
    t2_rgb:        np.ndarray,
    yellow_mask:   Optional[np.ndarray],
    spectral_signal: Optional[np.ndarray],
) -> List[Dict]:
    import cv2
    threshold = CFG["red_alert_threshold"]
    confirmed  = ((red_score >= threshold) & (building_mask > 0)).astype(np.uint8)

    contours, _ = cv2.findContours(confirmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []

    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px < 16:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        pad = CFG.get("chip_padding_px", 64)
        H_img, W_img = t2_rgb.shape[:2]

        r_min = max(0, y - pad)
        r_max = min(H_img, y + h + pad)
        c_min = max(0, x - pad)
        c_max = min(W_img, x + w + pad)

        # lat/lon bounding box
        coords = (
            bbox_to_latlon(r_min, c_min, r_max, c_max, transform)
            if transform else {}
        )

        # Score for this region
        region_score = float(red_score[y:y+h, x:x+w].mean())

        # Yellow overlap (spectral contribution to this region)
        yellow_overlap = 0.0
        if yellow_mask is not None:
            region_yellow = yellow_mask[y:y+h, x:x+w]
            yellow_overlap = float(region_yellow.mean())

        pixel_m2 = pixel_area_m2(transform) if transform else 100.0
        area_ha  = area_px * pixel_m2 / 10_000

        regions.append({
            "id":              i,
            "bbox_px":         [x, y, x + w, y + h],
            "chip_px":         [c_min, r_min, c_max, r_max],  # padded
            "area_ha":         round(area_ha, 4),
            "red_score":       round(region_score, 4),
            "yellow_overlap":  round(yellow_overlap, 4),
            "coordinates":     coords,
            "location":        None,   # filled by geocoder
        })

    # Sort largest first
    regions.sort(key=lambda r: r["area_ha"], reverse=True)
    return regions


# ── Reverse geocoding ────────────────────────────────────────────────────────

def _geocode_regions(regions: List[Dict]) -> List[Dict]:
    """
    Reverse geocode the centre of each region using OSM Nominatim.
    Rate-limited to 1 request/second to respect Nominatim's ToS.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut

        geolocator = Nominatim(user_agent=CFG.get("geocode_user_agent", "food_security_pipeline"))

        for region in regions:
            coords = region.get("coordinates", {})
            lat = coords.get("center_lat")
            lon = coords.get("center_lon")
            if lat is None or lon is None:
                continue
            try:
                location = geolocator.reverse(
                    (lat, lon), language="en", zoom=10, timeout=5
                )
                if location:
                    addr = location.raw.get("address", {})
                    region["location"] = {
                        "display_name": location.address,
                        "governorate":  addr.get("state") or addr.get("county", ""),
                        "district":     addr.get("city") or addr.get("town") or addr.get("village", ""),
                        "country":      addr.get("country", ""),
                        "lat":          round(lat, 6),
                        "lon":          round(lon, 6),
                    }
            except GeocoderTimedOut:
                logger.warning(f"Geocoding timeout for region {region['id']}")
            except Exception as e:
                logger.warning(f"Geocoding failed for region {region['id']}: {e}")
            time.sleep(1.1)   # Nominatim rate limit

    except ImportError:
        logger.warning("geopy not installed — skipping reverse geocoding. Run: pip install geopy")

    return regions


# ── Per-region chips ─────────────────────────────────────────────────────────

def _save_region_chips(
    regions: List[Dict],
    chips_dir: Path,
    t1_rgb: Optional[np.ndarray],
    t2_rgb: np.ndarray,
    transform,
) -> None:
    import cv2

    for region in regions:
        c_min, r_min, c_max, r_max = region["chip_px"]
        coords = region.get("coordinates", {})
        rid = region["id"]

        t2_chip = t2_rgb[r_min:r_max, c_min:c_max]

        if t1_rgb is not None:
            t1_chip = t1_rgb[r_min:r_max, c_min:c_max]
            # Side-by-side before | after
            chip_h = max(t1_chip.shape[0], t2_chip.shape[0])
            chip_w = t1_chip.shape[1] + t2_chip.shape[1] + 4

            canvas = np.zeros((chip_h + 40, chip_w, 3), dtype=np.uint8)
            canvas[:t1_chip.shape[0], :t1_chip.shape[1]] = t1_chip
            canvas[:t2_chip.shape[0], t1_chip.shape[1]+4:t1_chip.shape[1]+4+t2_chip.shape[1]] = t2_chip

            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "BEFORE", (10, chip_h + 25), font, 0.6, (255,255,255), 1)
            cv2.putText(canvas, "AFTER",  (t1_chip.shape[1]+14, chip_h + 25), font, 0.6, (0,100,255), 1)

            # Draw red rect on T2 side marking the detected building area
            x, y, x2, y2 = region["bbox_px"]
            rx1 = (x - c_min) + t1_chip.shape[1] + 4
            ry1 = y - r_min
            rx2 = (x2 - c_min) + t1_chip.shape[1] + 4
            ry2 = y2 - r_min
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        else:
            canvas = t2_chip.copy()
            x, y, x2, y2 = region["bbox_px"]
            cv2.rectangle(canvas,
                (x - c_min, y - r_min),
                (x2 - c_min, y2 - r_min),
                (0, 0, 255), 2)

        # Lat/lon annotation
        if coords:
            label = (
                f"Lat {coords['center_lat']:.4f}  Lon {coords['center_lon']:.4f}  "
                f"| {region['area_ha']:.2f} ha  red={region['red_score']:.2f}"
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0,0,0), -1)
            cv2.putText(canvas, label, (5, 16), font, 0.45, (255,255,255), 1)

        cv2.imwrite(
            str(chips_dir / f"region_{rid:03d}.png"),
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        )
        region["chip_image"] = str(chips_dir / f"region_{rid:03d}.png")


# ── GeoJSON ──────────────────────────────────────────────────────────────────

def _save_geojson(regions: List[Dict], path: Path) -> None:
    features = []
    for r in regions:
        coords = r.get("coordinates", {})
        if not coords:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coords.get("center_lon", 0), coords.get("center_lat", 0)],
            },
            "properties": {
                "id":            r["id"],
                "area_ha":       r["area_ha"],
                "red_score":     r["red_score"],
                "yellow_overlap": r["yellow_overlap"],
                "alert":         "RED" if r["red_score"] >= CFG["red_alert_threshold"] else "YELLOW",
                "location":      r.get("location"),
                "lat_min":       coords.get("lat_min"),
                "lat_max":       coords.get("lat_max"),
                "lon_min":       coords.get("lon_min"),
                "lon_max":       coords.get("lon_max"),
            },
        })
    geojson = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(geojson, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Folium interactive map ───────────────────────────────────────────────────

def _build_folium_map(
    regions: List[Dict],
    colored:  np.ndarray,
    meta:     Dict[str, Any],
    out_path: Path,
) -> None:
    try:
        import folium
        from folium.plugins import MiniMap, Fullscreen
    except ImportError:
        logger.warning("folium not installed — skipping interactive map. Run: pip install folium")
        return

    # Centre map on mean of all detected regions (or Egypt if none)
    lats = [r["coordinates"]["center_lat"] for r in regions if r.get("coordinates")]
    lons = [r["coordinates"]["center_lon"] for r in regions if r.get("coordinates")]
    center = [
        float(np.mean(lats)) if lats else 26.8,
        float(np.mean(lons)) if lons else 30.8,
    ]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
    MiniMap().add_to(m)
    Fullscreen().add_to(m)

    # Layer groups
    red_group    = folium.FeatureGroup(name="🔴 Red Alert — Encroachment")
    yellow_group = folium.FeatureGroup(name="🟡 Yellow Alert — Spectral Degradation")

    for r in regions:
        coords = r.get("coordinates", {})
        if not coords:
            continue

        lat = coords["center_lat"]
        lon = coords["center_lon"]
        loc = r.get("location") or {}
        alert_type = "RED" if r["red_score"] >= CFG["red_alert_threshold"] else "YELLOW"

        # Bounding rectangle
        sw = [coords["lat_min"], coords["lon_min"]]
        ne = [coords["lat_max"], coords["lon_max"]]
        color = "red" if alert_type == "RED" else "orange"

        folium.Rectangle(
            bounds=[sw, ne],
            color=color,
            fill=True,
            fill_opacity=0.2,
            weight=2,
        ).add_to(red_group if alert_type == "RED" else yellow_group)

        # Popup with full info
        chip_img = r.get("chip_image", "")
        popup_html = f"""
        <div style="font-family:Arial;min-width:280px">
          <b>Region #{r['id']}</b><br>
          <hr style="margin:4px 0">
          <b>Alert:</b> <span style="color:{'red' if alert_type=='RED' else 'orange'}">{alert_type}</span><br>
          <b>Red score:</b> {r['red_score']:.3f}
            (change×{CFG['red_alert_change_weight']} + spectral×{CFG['red_alert_spectral_weight']})<br>
          <b>Yellow overlap:</b> {r['yellow_overlap']:.3f}<br>
          <b>Area lost:</b> {r['area_ha']:.4f} ha<br>
          <hr style="margin:4px 0">
          <b>Lat:</b> {lat:.6f} &nbsp; <b>Lon:</b> {lon:.6f}<br>
          {f"<b>District:</b> {loc.get('district','')}<br>" if loc.get('district') else ""}
          {f"<b>Governorate:</b> {loc.get('governorate','')}<br>" if loc.get('governorate') else ""}
          {f"<b>Country:</b> {loc.get('country','')}<br>" if loc.get('country') else ""}
          {f'<b>Address:</b> <small>{loc.get("display_name","")}</small><br>' if loc.get('display_name') else ""}
          {f'<br><img src="{chip_img}" style="max-width:100%;border:1px solid #ccc">' if chip_img else ""}
        </div>
        """
        marker_color = "red" if alert_type == "RED" else "orange"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{'🔴' if alert_type=='RED' else '🟡'} {r['area_ha']:.2f} ha — {loc.get('district','unknown')}",
            icon=folium.Icon(color=marker_color, icon="exclamation-sign"),
        ).add_to(red_group if alert_type == "RED" else yellow_group)

    red_group.add_to(m)
    yellow_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Summary box
    total_ha = sum(r["area_ha"] for r in regions)
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
                padding:12px 16px;border-radius:8px;border:2px solid #666;font-family:Arial;font-size:13px">
      <b>Encroachment Summary</b><br>
      🔴 Red regions: {sum(1 for r in regions if r['red_score'] >= CFG['red_alert_threshold'])}<br>
      🟡 Yellow regions: {sum(1 for r in regions if r['red_score'] < CFG['red_alert_threshold'])}<br>
      📐 Total area lost: <b>{total_ha:.2f} ha</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(out_path))
    logger.info(f"Interactive map saved: {out_path}")
