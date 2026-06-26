"""
batch_report.py  -  Run inference on all positive sites and build master summary.

    python batch_report.py               # all positive sites
    python batch_report.py --limit 10    # first 10 positives only
    python batch_report.py --all         # all 300 sites (pos + neg)

Outputs:
  outputs/<site>_report.html  - individual site reports
  encroachment_summary.html   - master Egypt map + sortable table
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    labels = json.load(open("data/ba_labels.json"))
    targets = labels if args.all else [r for r in labels if r["label"]=="pos"]
    if args.limit:
        targets = targets[:args.limit]

    print(f"Running inference on {len(targets)} sites...")
    from run_inference import run, BA_DIR
    results = []
    t0 = time.time()
    for i, r in enumerate(targets):
        sn = r["site"]
        bp = BA_DIR/(sn+"_before_2024.tif")
        ap = BA_DIR/(sn+"_after_2025.tif")
        if not bp.exists():
            print(f"  [{i+1}/{len(targets)}] {sn} - missing, skip")
            continue
        print(f"  [{i+1}/{len(targets)}] {sn} ...", end=" ", flush=True)
        t1 = time.time()
        prob, total_ha, main_ha, n_cl = run(bp, ap, sn)
        results.append({"site":sn,"label":r["label"],"prob":prob,
                        "total_km2":round(total_ha*0.01,4),
                        "main_km2":round(main_ha*0.01,4),"clusters":n_cl})
        print(f"prob={prob:.3f}  lost={total_ha*0.01:.3f}km2  ({time.time()-t1:.1f}s)")

    print(f"\nDone: {len(results)} sites in {time.time()-t0:.0f}s")
    Path("outputs").mkdir(exist_ok=True)
    json.dump(results, open("outputs/batch_results.json","w"), indent=2)
    build_summary(results)
    print("Summary: encroachment_summary.html")
    print("Tip: python geocode_reports.py outputs/*_report.html")


def build_summary(results):
    import rasterio, json as _j
    from rasterio.warp import transform_bounds
    from run_inference import BA_DIR

    markers = []
    for r in results:
        bp = BA_DIR/(r["site"]+"_before_2024.tif")
        try:
            with rasterio.open(bp) as s:
                wgs = transform_bounds(s.crs,"EPSG:4326",*s.bounds)
            clat=(wgs[1]+wgs[3])/2; clon=(wgs[0]+wgs[2])/2
        except Exception:
            clat,clon=30.0,31.0
        markers.append({**r,"lat":round(clat,5),"lon":round(clon,5)})

    det=[m for m in markers if m["prob"]>=0.40]
    pos=[m for m in markers if m["label"]=="pos"]
    total_lost=sum(m["total_km2"] for m in pos)

    rows="".join(
        "<tr onclick=\"flyTo(%(lat)s,%(lon)s)\" style='cursor:pointer'>"
        "<td>%(site)s</td>"
        "<td style='color:%(clr)s'>%(tick)s</td>"
        "<td>%(prob).3f</td><td>%(km2).4f</td><td>%(cl)d</td>"
        "<td><a href='outputs/%(site)s_report.html' target='_blank' style='color:#58a6ff'>↗</a></td></tr>"
        % dict(clr="#ff4444" if m["prob"]>=0.40 else "#44ff88",
               tick="ENCR" if m["prob"]>=0.40 else "neg",
               prob=m["prob"],km2=m["total_km2"],cl=m["clusters"],
               lat=m["lat"],lon=m["lon"],site=m["site"])
        for m in sorted(markers,key=lambda x:-x["prob"])
    )

    css=("*{margin:0;padding:0;box-sizing:border-box}"
         "body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;display:flex;flex-direction:column;height:100vh}"
         "header{padding:12px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:18px;flex-shrink:0;flex-wrap:wrap}"
         "h1{font-size:1rem;color:#8ee3ff}"
         ".k{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:7px 12px;font-size:10px;color:#6e7681}"
         ".kv{font-size:1.2rem;font-weight:700;display:block}"
         ".main{display:flex;flex:1;min-height:0}"
         "#map{flex:1}"
         ".sb{width:330px;background:#161b22;border-left:1px solid #30363d;overflow-y:auto;flex-shrink:0}"
         ".sb h3{padding:9px 12px;font-size:10px;color:#8ee3ff;text-transform:uppercase;letter-spacing:.07em;"
         "border-bottom:1px solid #21262d;position:sticky;top:0;background:#161b22;z-index:1}"
         "table{width:100%;border-collapse:collapse;font-size:11px}"
         "th{padding:5px 8px;color:#6e7681;text-align:left;border-bottom:1px solid #21262d;font-size:10px}"
         "td{padding:5px 8px;border-bottom:1px solid #0d1117}"
         "tr:hover td{background:#21262d}"
         "footer{padding:7px 20px;font-size:10px;color:#6e7681;border-top:1px solid #21262d;flex-shrink:0}")

    mapjs=(
        "var M="+_j.dumps(markers)+";"
        "var map=L.map('map').setView([27,30],6);"
        "var sat=L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',"
        "{maxZoom:20,subdomains:['mt0','mt1','mt2','mt3'],attribution:'Google'});"
        "var osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'OSM'});"
        "sat.addTo(map);"
        "L.control.layers({'Satellite':sat,'OSM':osm}).addTo(map);"
        "L.control.scale().addTo(map);"
        "M.forEach(function(m){"
        "var clr=m.prob>=0.40?'#ff4444':'#44ff88';var r=m.prob>=0.40?7:4;"
        "L.circleMarker([m.lat,m.lon],{radius:r,color:clr,fillColor:clr,fillOpacity:0.8,weight:1.5})"
        ".addTo(map).bindPopup('<b>'+m.site+'</b><br>RF: '+m.prob.toFixed(3)"
        "+'<br>Lost: '+m.total_km2+' km²<br>'"
        "+'<a href=\"outputs/'+m.site+'_report.html\" target=\"_blank\">Open report ↗</a>');"
        "});"
        "function flyTo(lat,lon){map.setView([lat,lon],15);}"
    )

    html="\n".join([
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<title>KEMET1 Encroachment Summary</title>",
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>",
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>",
        "<style>"+css+"</style></head><body>",
        "<header>",
        "  <h1>KEMET1 — Nile Delta Encroachment Summary</h1>",
        "  <div class='k'>Sites<span class='kv'>"+str(len(markers))+"</span></div>",
        "  <div class='k'>Encroachment<span class='kv' style='color:#ff4444'>"+str(len(det))+"</span></div>",
        "  <div class='k'>Total Lost<span class='kv' style='color:#ff8c00'>"+f"{total_lost:.2f} km²"+"</span></div>",
        "  <div class='k'>Positive Sites<span class='kv'>"+str(len(pos))+"</span></div>",
        "</header>",
        "<div class='main'><div id='map'></div>",
        "<div class='sb'><h3>Sites — sorted by RF score</h3>",
        "<table><thead><tr><th>Site</th><th>Result</th><th>Score</th><th>Lost km²</th><th>Cl</th><th></th></tr></thead>",
        "<tbody>"+rows+"</tbody></table></div></div>",
        "<footer>KEMET1 BeforeAfter RF · Sentinel-2 10m · 2024→2025 · Val AUC 0.9596</footer>",
        "<script>"+mapjs+"</script>",
        "</body></html>",
    ])
    Path("encroachment_summary.html").write_text(html,encoding="utf-8")


if __name__=="__main__":
    main()
