"""Per-site HTML reports and the all-sites summary map."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from string import Template

from . import alarm
from .analysis import SiteResult
from .clusters import pixel_to_lonlat
from .imagery import panel_png

TEMPLATES = Path(__file__).parent / "templates"
RISK_COLOURS = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}


def _card(label: str, value: str, colour: str = "#e6edf3") -> str:
    return (
        f'<div class="card"><div class="lbl">{label}</div><div class="val" style="color:{colour}">{value}</div></div>'
    )


def site_report_html(result: SiteResult) -> str:
    pair = result.pair
    lat, lon = pair.centre
    west, south, east, north = pair.bounds_wgs84
    shape = pair.before.shape[1:]
    top = max((c.score for c in result.clusters), default=0.0)
    clusters, rows = [], []
    for i, c in enumerate(result.clusters):
        r0, c0, r1, c1 = c.box
        n_lat, w_lon = pixel_to_lonlat(r0, c0, shape, pair.bounds_wgs84)
        s_lat, e_lon = pixel_to_lonlat(r1, c1, shape, pair.bounds_wgs84)
        c_lat, c_lon = pixel_to_lonlat(*c.centre_px, shape, pair.bounds_wgs84)
        clusters.append(
            {
                "bounds": [[round(s_lat, 6), round(w_lon, 6)], [round(n_lat, 6), round(e_lon, 6)]],
                "lat": round(c_lat, 6),
                "lon": round(c_lon, 6),
                "area_ha": c.area_ha,
                "colour": c.colour(top),
            }
        )
        rows.append(
            f'<tr><td style="color:{c.colour(top)}">{i + 1}</td><td id="place{i}">{c_lat:.4f}, {c_lon:.4f}</td>'
            f"<td>{c.area_ha:.1f} ha</td><td>{c.score:.3f}</td></tr>"
        )
    colour = RISK_COLOURS[result.risk]
    cards = "".join(
        [
            _card("Probability", f"{result.probability:.3f}", colour),
            _card("Risk", result.risk, colour),
            _card("Changed area", f"{result.changed_ha:.1f} ha", "#ffa657"),
            _card("Largest cluster", f"{result.clusters[0].area_ha:.1f} ha" if result.clusters else "-"),
            _card("Mean NDVI loss", f"{result.ndvi_loss:.3f}"),
            _card("Tile", f"{pair.tile_area_ha:.0f} ha"),
            _card("Centre", f"{lat:.4f} N, {lon:.4f} E"),
        ]
    )
    template = Template((TEMPLATES / "site_report.html").read_text(encoding="utf-8"))
    return template.substitute(
        site=result.name,
        risk=result.risk,
        risk_colour=colour,
        probability=f"{result.probability:.3f}",
        high_threshold=alarm.HIGH_THRESHOLD,
        medium_threshold=alarm.MEDIUM_THRESHOLD,
        before_year=pair.before_year,
        after_year=pair.after_year,
        cards=cards,
        n_clusters=len(result.clusters),
        rows="".join(rows) or '<tr><td colspan="4">No clusters of 0.5 ha or more</td></tr>',
        panel=panel_png(result),
        lat=f"{lat:.6f}",
        lon=f"{lon:.6f}",
        tile_bounds=json.dumps([[south, west], [north, east]]),
        tile_ha=f"{pair.tile_area_ha:.0f}",
        clusters_json=json.dumps(clusters),
        generated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )


def write_site_report(result: SiteResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.name}_report.html"
    path.write_text(site_report_html(result), encoding="utf-8")
    return path


def summary_html(summaries: list[dict]) -> str:
    """One map of every scored site, coloured by risk, linking to the site reports."""
    for s in summaries:
        s["colour"] = RISK_COLOURS[s["risk_level"]]
    counts = {level: sum(s["risk_level"] == level for s in summaries) for level in RISK_COLOURS}
    rows = "".join(
        f'<tr onclick="map.setView([{s["lat"]},{s["lon"]}],14)"><td><a href="{s["site"]}_report.html">{s["site"]}</a></td>'
        f'<td style="color:{s["colour"]}">{s["risk_level"]}</td><td>{s["probability"]:.3f}</td><td>{s["changed_ha"]:.1f}</td></tr>'
        for s in sorted(summaries, key=lambda s: -s["probability"])
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>KEMET1 summary</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;display:flex;flex-direction:column;height:100vh;background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',system-ui,sans-serif}}
header{{padding:12px 20px;border-bottom:1px solid #21262d}}h1{{font-size:1rem;display:inline;margin-right:16px}}
.main{{display:flex;flex:1;min-height:0}}#map{{flex:1}}.sb{{width:340px;overflow-y:auto;background:#161b22}}
table{{width:100%;border-collapse:collapse;font-size:12px}}td,th{{padding:5px 8px;border-bottom:1px solid #21262d;text-align:left}}
tr{{cursor:pointer}}a{{color:#58a6ff}}</style></head><body>
<header><h1>KEMET1 - {len(summaries)} sites</h1>
<span style="color:{RISK_COLOURS["High"]}">High {counts["High"]}</span> &middot;
<span style="color:{RISK_COLOURS["Medium"]}">Medium {counts["Medium"]}</span> &middot;
<span style="color:{RISK_COLOURS["Low"]}">Low {counts["Low"]}</span></header>
<div class="main"><div id="map"></div><div class="sb"><table><thead><tr><th>Site</th><th>Risk</th><th>Prob.</th><th>ha</th></tr></thead>
<tbody>{rows}</tbody></table></div></div>
<script>var S={json.dumps(summaries)};var map=L.map('map').setView([30.6,31.0],8);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'OpenStreetMap'}}).addTo(map);
S.forEach(function(s){{L.circleMarker([s.lat,s.lon],{{radius:s.risk_level=='Low'?4:7,color:s.colour,fillOpacity:.85}})
.addTo(map).bindPopup('<b>'+s.site+'</b><br>'+s.risk_level+' ('+s.probability.toFixed(3)+')<br><a href="'+s.site+'_report.html">report</a>');}});</script>
</body></html>"""
